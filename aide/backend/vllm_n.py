import logging
import time
import os
from typing import Optional, Dict, Any, Tuple, List
import openai
from omegaconf import OmegaConf
from funcy import notnone, once, select_values
from aide.backend.utils import OutputType, opt_messages_to_list, backoff_create

logger = logging.getLogger("aide")

_client1: openai.OpenAI = None
_vllm_config1 = {
    "base_url": os.getenv("VLLM_BASE_URL2", "http://localhost:8000/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
}

_client: openai.OpenAI = None
_vllm_config = {
    "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
}

VLLM_API_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIError,
    openai.InternalServerError,
)

@once
def _setup_vllm_client():
    global _client
    logger.info(f"Setting up planner vLLM client with base_url: {_vllm_config['base_url']}", extra={"verbose": True})
    _client = openai.OpenAI(
        base_url=_vllm_config["base_url"],
        api_key=_vllm_config["api_key"],
        max_retries=0,
    )

@once
def _setup_vllm_client1():
    global _client1
    logger.info(f"Setting up coder vLLM client with base_url: {_vllm_config1['base_url']}", extra={"verbose": True})
    _client1 = openai.OpenAI(
        base_url=_vllm_config1["base_url"],
        api_key=_vllm_config1["api_key"],
        max_retries=0,
    )

def set_vllm_config(cfg: OmegaConf):
    global _vllm_config
    if cfg.get("vllm"):
        _vllm_config.update({
            "base_url": cfg.vllm.get("base_url", _vllm_config["base_url"]),
            "api_key": cfg.vllm.get("api_key", _vllm_config["api_key"]),
        })
    logger.debug(f"Updated vLLM config: base_url={_vllm_config['base_url']}", extra={"verbose": True})

def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "Qwen/Qwen2-0.5B-Instruct",
    temperature: float = 0.7,
    planner: bool = False,
    n: int = 1,
    convert_system_to_user: bool = False,
    max_retries: int = 3,
    **model_kwargs: Any,
) -> Tuple[List[str], float, List[int], List[int], List[Dict[str, Any]]]:
    """
    Query a vLLM-hosted model and return `n` samples.
    
    Returns:
      - outputs:  List of `n` response strings
      - req_time: total round-trip time
      - input_tokens:  list of prompt token counts per sample
      - output_tokens: list of completion token counts per sample
      - infos:   list of metadata dicts per sample
    """
    logger.info("activated vllm backend...", extra={"verbose": True})
    model_kwargs = select_values(notnone, model_kwargs)

    def prepare_messages(sys_msg):
        return opt_messages_to_list(sys_msg, user_message, convert_system_to_user=False)

    current_system = system_message
    retries = 0
    while retries < max_retries:
        messages = prepare_messages(current_system)

        # build OpenAI-style params
        api_params = {
            "temperature": temperature,
            "max_tokens": model_kwargs.get("max_new_tokens"),
            "top_p": model_kwargs.get("top_p"),
            "top_k": model_kwargs.get("top_k"),
            "stop": model_kwargs.get("stop"),
            "frequency_penalty": model_kwargs.get("frequency_penalty"),
            "presence_penalty": model_kwargs.get("presence_penalty"),
            "n": n,   # request `n` completions
        }
        filtered = {k: v for k, v in api_params.items() if v is not None}
        filtered["model"] = model

        try:
            # choose planner vs coder client
            if not planner:
                _setup_vllm_client()
                t0 = time.time()
                completion = backoff_create(
                    _client.chat.completions.create,
                    VLLM_API_EXCEPTIONS,
                    messages=messages,
                    **filtered,
                )
            else:
                _setup_vllm_client1()
                t0 = time.time()
                completion = backoff_create(
                    _client1.chat.completions.create,
                    VLLM_API_EXCEPTIONS,
                    messages=messages,
                    **filtered,
                )

            req_time = time.time() - t0
            choices = completion.choices or []
            if not choices:
                raise ValueError("Empty completion.choices")

            # extract all samples
            outputs = [c.message.content or "" for c in choices]
            input_toks  = [c.usage.prompt_tokens     for c in choices]
            output_toks = [c.usage.completion_tokens for c in choices]
            infos = [
                {
                  "model":   completion.model,
                  "finish_reason": c.finish_reason,
                  "id":      c.id,
                  "created": completion.created,
                }
                for c in choices
            ]
            return outputs, req_time, input_toks, output_toks, infos

        except Exception as e:
            logger.error(f"vLLM query failed (attempt {retries+1}): {e}", exc_info=True)
            retries += 1
            if retries == 2:
                logger.warning("Dropping system message after 2 failures.")
                current_system = None
            if retries >= max_retries:
                # all retries exhausted
                err = {"error": str(e)}
                return [f"ERROR: {e}"], 0.0, [0], [0], [err]

    # should not reach here
    return ["ERROR: Unknown failure"], 0.0, [0], [0], [{"error": "Unknown failure"}]
