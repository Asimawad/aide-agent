# python/backend_vllm.py
import logging
import re
import time
import os
from typing import Optional, Dict, Any, Tuple
import openai 
from omegaconf import OmegaConf

from aide.backend.utils import OutputType, opt_messages_to_list, backoff_create

logger = logging.getLogger("aide") 

_client: openai.OpenAI = None 
_vllm_config: dict = { 
    "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"), 
    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"), }

VLLM_API_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIError,
    openai.InternalServerError,
)

def _setup_vllm_client(): # <<< KEEP: Correctly initializes the OpenAI client once.
    """Sets up the OpenAI client for vLLM server."""
    global _client
    if _client is None:
        logger.info(f"Setting up vLLM client with base_url: {_vllm_config['base_url']}", extra={"verbose": True})
        try:
            _client = openai.OpenAI(
                base_url=_vllm_config["base_url"],
                api_key=_vllm_config["api_key"],
                max_retries=0,  # Rely on backoff_create for retries
            )
        except Exception as e:
            logger.error(f"Failed to setup vLLM client: {e}")
            raise

# Only needed if you configure base_url/api_key via a central OmegaConf object passed to it. Remove if config is purely via env vars or static. >>>
def set_vllm_config(cfg: OmegaConf):
    """Update vLLM config from OmegaConf."""
    global _vllm_config
    if cfg.get("vllm"):
        _vllm_config.update({
            "base_url": cfg.vllm.get("base_url", _vllm_config["base_url"]),
            "api_key": cfg.vllm.get("api_key", _vllm_config["api_key"]),
        })
    logger.info(f"Updated vLLM config: base_url={_vllm_config['base_url']}", extra={"verbose": True})
 
def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "Qwen/Qwen2-0.5B-Instruct", # Model name is passed to the server API call
    temperature: float = 0.7,
    func_spec=None, # Add func_spec argument (even if ignored by vLLM server) to match signature
    convert_system_to_user=False, # Add convert_system_to_user argument for signature match
    **model_kwargs: Any,
) -> Tuple[OutputType, float, int, int, Dict[str, Any]]: # <<< KEEP: Standard backend return signature
    """
    Query a vLLM-hosted model using OpenAI-compatible API.
    """
    _setup_vllm_client() 

    # Split prompt if needed (conditionally kept)
    
    # Prepare messages list for OpenAI API format
    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=False) # 

    # Filter kwargs and map to OpenAI API names
    # Include common parameters vLLM's OpenAI endpoint accepts
    api_params = {
        "temperature": temperature,
        "max_tokens": model_kwargs.get("max_new_tokens"), # Map internal 'max_new_tokens' to API's 'max_tokens'
        "top_p": model_kwargs.get("top_p"),
        "top_k": model_kwargs.get("top_k"), # vLLM supports top_k, but might need specific value like -1 to disable
        "stop": model_kwargs.get("stop"),
        "frequency_penalty": model_kwargs.get("frequency_penalty"),
        "presence_penalty": model_kwargs.get("presence_penalty"),
        # Add other supported params as needed
    }
    # Filter out None values as API might not like them
    filtered_api_params = {k: v for k, v in api_params.items() if v is not None}

    # Add the model name (required by the API call)
    filtered_api_params["model"] = model

    # Log ignored kwargs (useful for debugging)
    ignored_kwargs = set(model_kwargs.keys()) - set(api_params.keys())
    if ignored_kwargs:
         logger.warning(f"Ignored invalid or unmapped model_kwargs for vLLM API backend: {ignored_kwargs}", extra={"verbose": True})

    # Perform API call
    t0 = time.time() 
    try:
        # Use backoff_create for retries on API errors
        completion = backoff_create(
            _client.chat.completions.create,
            VLLM_API_EXCEPTIONS, # Use the defined exceptions
            messages=messages,
            **filtered_api_params, # Pass the filtered API parameters
        )
    except Exception as e:
        # Catch potential errors during the API call itself
        logger.error(f"vLLM query failed: {e}", exc_info=True) # Log full traceback
        # Return an error structure consistent with the expected tuple signature
        return f"ERROR: {e}", time.time() - t0, 0, 0, {"error": str(e)}


    req_time = time.time() - t0 
    # Process response
    if not completion or not completion.choices:
         logger.error("vLLM API call returned empty or invalid completion object.")
         return "ERROR: Invalid API response", req_time, 0, 0, {"error": "Invalid API response"}

    choice = completion.choices[0]
    output = choice.message.content or "" # Generated text

    # Token counts (vLLM server *should* return usage stats)
    input_tokens = completion.usage.prompt_tokens if completion.usage else 0
    output_tokens = completion.usage.completion_tokens if completion.usage else 0

    # Metadata
    info = { # <<< KEEP: Useful metadata to return
        "model": completion.model, # Model name returned by the server
        "finish_reason": choice.finish_reason,
        # Add other useful info if available, e.g., completion ID
        "id": completion.id,
        "created": completion.created,
    }
    logger.info(f"No of tokens {output_tokens}")
    return output, req_time, input_tokens, output_tokens, info