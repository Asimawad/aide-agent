#
# python/backend_vllm.py
import logging
import re
import time
import os
from typing import Optional, Dict, Any, Tuple, List
import openai
from omegaconf import OmegaConf
from funcy import notnone, once, select_values

from aide.backend.utils import (
    OutputType,
    opt_messages_to_list,
    backoff_create,
    ContextLengthExceededError,
)

logger = logging.getLogger("aide")



_client: openai.OpenAI = None
_vllm_config: dict = {
    "base_url": os.getenv("VLLM_BASE_URL", f"http://localhost:8000/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
}
@once
def _setup_vllm_client():
    """Sets up the OpenAI client for vLLM server."""
    global _client
    logger.info(
        f"Setting up planner vLLM client with base_url: {_vllm_config['base_url']}",
        extra={"verbose": True},
    )
    try:
        _client = openai.OpenAI(
            base_url=_vllm_config["base_url"],
            api_key=_vllm_config["api_key"],
            max_retries=0,  # Rely on backoff_create for retries
        )

    except Exception as e:
        logger.error(f"Failed to setup vLLM client: {e}")
        raise

VLLM_API_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIError,
    openai.InternalServerError,
)


# Only needed if you configure base_url/api_key via a central OmegaConf object passed to it. Remove if config is purely via env vars or static. >>>
def set_vllm_config(cfg: OmegaConf):
    """Update planner vLLM config from OmegaConf."""
    global _vllm_config
    if cfg.get("vllm"):
        _vllm_config.update(
            {
                "base_url": cfg.vllm.get("base_url", _vllm_config["base_url"]),
                "api_key": cfg.vllm.get("api_key", _vllm_config["api_key"]),
            }
        )
    logger.debug(
        f"Updated vLLM config: base_url={_vllm_config['base_url']}",
        extra={"verbose": True},
    )

def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "Qwen/Qwen2-0.5B-Instruct",
    temperature: float = 0.7,
    planner=False,
    func_spec=None,
    num_responses: int = 1,
    convert_system_to_user=False,
    max_retries=3,
    **model_kwargs: Any,
) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
    """

    Query a vLLM-hosted model, returning up to `n` completions.
    Returns (outputs, elapsed_s, prompt_tokens, completion_tokens, info).
    """
    # build messages
    logger.info("activated vllm backend...", extra={"verbose": True})
    model_kwargs = select_values(notnone, model_kwargs)
    # Prepare messages list for OpenAI API format
    def prepare_messages(sys_msg):
        return opt_messages_to_list(sys_msg, user_message, convert_system_to_user=False)

    current_system_message = system_message
    retries = 0
    while retries < max_retries:
        messages = prepare_messages(current_system_message)
        
        api_params = {
            "temperature": temperature,
            "max_tokens": model_kwargs.get("max_new_tokens", 4096),
            "top_p": model_kwargs.get("top_p", 0.95),
            "frequency_penalty": model_kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": model_kwargs.get("presence_penalty", 0.0),
        }
        filtered_api_params = {k: v for k, v in api_params.items() if v is not None}
        filtered_api_params["model"] = model
        filtered_api_params["n"] = num_responses


        try:
            _setup_vllm_client()
            t0 = time.time()
            completion = backoff_create(
                _client.chat.completions.create,
                VLLM_API_EXCEPTIONS,
                messages=messages,
                **filtered_api_params,
            )

            print(f"number of responses generated from vllm: {len(completion.choices)}")
            req_time = time.time() - t0
            if not completion or not completion.choices:
                logger.error(
                    "vLLM API call returned empty or invalid completion object."
                )
                return (
                    "ERROR: Invalid API response",
                    req_time,
                    0,
                    0,
                    {"error": "Invalid API response"},
                )
            elapsed = time.time() - t0

            # collect outputs
            outputs: List[OutputType] = []
            for choice in completion.choices:
                text = choice.message.content or ""
                outputs.append(text)

            # pad if fewer than requested
            while len(outputs) < num_responses:
                outputs.append("")

            # token usage
            prompt_toks = getattr(completion.usage, "prompt_tokens", 0)
            comp_toks   = getattr(completion.usage, "completion_tokens", 0)

            info = {
                "model": completion.model,
                "n_requested": num_responses,
                "n_returned": len(completion.choices),
                "finish_reasons": [c.finish_reason for c in completion.choices],
                "id": getattr(completion, "id", None),
                "created": getattr(completion, "created", None),
            }
            return outputs, elapsed, prompt_toks, comp_toks, info

        except ContextLengthExceededError as cle:
            logger.error(f"Context length exceeded: {cle}")
            return ["ERROR: context length"] * num_responses, 0.0, 0, 0, {"error": str(cle)}

        except Exception as e:
            retries += 1
            logger.warning(f"vLLM call failed (attempt {retries}/{max_retries}): {e}", exc_info=True)
            if retries >= max_retries:
                return [f"ERROR: {e}"] * num_responses, 0.0, 0, 0, {"error": str(e)}
            if retries == 2:
                logger.info("Dropping system_message to reduce context size")
                messages = opt_messages_to_list(None, user_message, convert_system_to_user=False)
            continue






    #         choice = completion.choices[0]
    #         output = choice.message.content or ""
    #         input_tokens = completion.usage.prompt_tokens if completion.usage else 0
    #         output_tokens = (
    #             completion.usage.completion_tokens if completion.usage else 0
    #         )
    #         info = {
    #             "model": completion.model,
    #             "finish_reason": choice.finish_reason,
    #             "id": completion.id,
    #             "created": completion.created,
    #         }
    #         logger.debug(f"No of tokens {output_tokens}")
    #         return output, req_time, input_tokens, output_tokens, info
    #     except ContextLengthExceededError as cle: # Catch specific error
    #         logger.error(f"Context Length Exceeded: {cle}. Aborting retries for this call.", exc_info=False, extra={"verbose": True}) # exc_info=False as CLE is already logged well
    #         return f"Exceeded context length limit", 0, 0, 0, {"error": str(cle)}
    #     except Exception as e:
    #         logger.error(
    #             f"vLLM query failed (attempt {retries + 1}): {e}", exc_info=True
    #         )
    #         retries += 1
    #         if retries == 2:
    #             # After 2 retries, drop the system message to shorten prompt
    #             logger.warning(
    #                 "Dropping system message after 2 failed attempts to mitigate long prompt issues."
    #             )
    #             current_system_message = None
    #         if retries >= max_retries:
    #             return f"ERROR: {e}", 0, 0, 0, {"error": str(e)}

    # # Should not reach here
    # return "ERROR: Unknown failure", 0, 0, 0, {"error": "Unknown failure"}
