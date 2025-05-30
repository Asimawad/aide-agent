# python/backend_vllm.py

import logging
import time
import os
from typing import Optional, List, Tuple, Dict, Any

import openai
from funcy import notnone, once, select_values

from aide.backend.utils import (
    OutputType,
    opt_messages_to_list,
    backoff_create,
    ContextLengthExceededError,
)

logger = logging.getLogger("aide")

# two separate clients for coder vs planner
_client_coder: openai.OpenAI = None
_client_planner: openai.OpenAI = None

# defaults; tweak via env
# python/backend_vllm.py
# python/backend_vllm.py
# python/backend_vllm.py
# backend_vllm.py  – keep the /v1 suffix
_VLLM_CODER_URL  = os.getenv("VLLM_BASE_URL",  "http://localhost:8000/v1")
_VLLM_PLAN_URL   = os.getenv("VLLM_BASE_URL2", "http://localhost:8001/v1")


_VLLM_CODER_APIKEY = os.getenv("VLLM_API_KEY",     "")
_VLLM_PLAN_APIKEY  = os.getenv("VLLM_API_KEY",     "")

VLLM_API_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIError,
    openai.InternalServerError,
)

@once
def _setup_coder_client():
    global _client_coder
    logger.info(f"Setting up vLLM coder client @ {_VLLM_CODER_URL}")
    _client_coder = openai.OpenAI(
        base_url=_VLLM_CODER_URL,
        api_key=_VLLM_CODER_APIKEY,
        max_retries=0,
    )

@once
def _setup_planner_client():
    global _client_planner
    logger.info(f"Setting up vLLM planner client @ {_VLLM_PLAN_URL}")
    _client_planner = openai.OpenAI(
        base_url=_VLLM_PLAN_URL,
        api_key=_VLLM_PLAN_APIKEY,
        max_retries=0,
    )

def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    *,
    model: str = "Qwen/Qwen2-0.5B-Instruct",
    temperature: float = 0.7,
    num_responses: int = 1,
    planner: bool = False,
    max_retries: int = 3,
    **unused_kwargs: Any,
) -> Tuple[List[OutputType], float, int, int, Dict[str, Any]]:
    """
    Query a vLLM-hosted model, returning up to `n` completions.
    Returns (outputs, elapsed_s, prompt_tokens, completion_tokens, info).
    """
    # build messages
    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=False)

    # pick client
    client_setup = _setup_planner_client if planner else _setup_coder_client
    client_setup()
    client = _client_planner if planner else _client_coder
    print(client.base_url)
    retries = 0
    current_system = system_message

    while True:
        # prepare API args
        api_kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "n": num_responses,
            "stop": unused_kwargs.get("stop"),
            "max_tokens": unused_kwargs.get("max_new_tokens") or unused_kwargs.get("max_tokens"),
            "top_p": unused_kwargs.get("top_p"),
            "top_k": unused_kwargs.get("top_k"),
            "frequency_penalty": unused_kwargs.get("frequency_penalty"),
            "presence_penalty": unused_kwargs.get("presence_penalty"),
        }
        # drop None values
        api_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}

        try:
            print(f"----------------------------------------------------------\n")
            print(f"api_kwargs: {api_kwargs}")
            print(f"messages: {messages}")
            print(f"----------------------------------------------------------\n")
            print(f"model: {model}")
            print(f"----------------------------------------------------------\n")
            print(f"temperature: {temperature}")
            print(f"----------------------------------------------------------\n")
            print(f"num_responses: {num_responses}")
            print(f"----------------------------------------------------------\n")
            print(f"planner: {planner}")
# backend_vllm.py  – right before the API call
            api_kwargs["max_tokens"] = min(1024, api_kwargs.get("max_tokens") or 1024)
            # (and if you don’t need multiple samples:)
            api_kwargs.pop("n", None)

            t0 = time.time()
            completion = backoff_create(
                client.chat.completions.create,
                VLLM_API_EXCEPTIONS,
                stream=True,
                model="RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic",
                messages=[{"role": "user", "content": "ping"}],
            )
            print(f"------------------###########----------------------------------------\n")
            print(f"completion: {completion}")
            print(f"-----------------######################-----------------------------------------\n")
            elapsed = time.time() - t0

            # sanity
            if not completion or not completion.choices:
                raise RuntimeError("Empty completion")

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
