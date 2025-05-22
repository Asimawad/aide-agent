# aide-ds-main/aide/backend/backend_hosted_deepseek.py
import json
import logging
import time
import os
from typing import Optional, Dict, Any, Tuple

import openai  # We'll use the openai library if the API is OpenAI-compatible

from aide.backend.utils import (
    FunctionSpec,  # Though likely not used by coder
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values

logger = logging.getLogger("aide")

# --- Configuration for the Hosted DeepSeek Coder API ---
# Fetch these from environment variables
HOSTED_DEEPSEEK_API_KEY = os.getenv("HOSTED_DEEPSEEK_API_KEY")

# Global client instance for this backend
_hosted_deepseek_client: Optional[openai.OpenAI] = None

# Define API exceptions that should trigger backoff/retries for this provider
# These are often similar to OpenAI's but check your provider's documentation
HOSTED_DEEPSEEK_API_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIError,
    openai.InternalServerError,  # Add others if specified by your provider
)


@once
def _setup_hosted_deepseek_client():
    """Sets up the OpenAI-compatible client for the hosted DeepSeek Coder service."""
    global _hosted_deepseek_client
    if not HOSTED_DEEPSEEK_API_KEY:
        logger.error(
            "HOSTED_DEEPSEEK_API_KEY environment variable not set. This backend cannot function."
        )
        # You could raise an error here or let it fail when _hosted_deepseek_client is used
        return

    logger.info(
        f"Setting up Hosted DeepSeek Coder client with base_url",
        extra={"verbose": True},
    )
    try:
        _hosted_deepseek_client = openai.OpenAI(
            base_url="https://api.deepseek.com", api_key=HOSTED_DEEPSEEK_API_KEY
        )
    except Exception as e:
        logger.error(
            f"Failed to setup Hosted DeepSeek Coder client: {e}", exc_info=True
        )
        _hosted_deepseek_client = None  # Ensure it's None on failure
        raise


def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: Optional[
        str
    ] = None,  # Model ID will be passed, ensure it's the correct one for this provider
    temperature: float = 0.6,  # Default temperature for coder
    func_spec: Optional[FunctionSpec] = None,  # Typically None for coder
    convert_system_to_user: bool = False,
    max_retries: int = 3,
    **model_kwargs: Any,
) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
    """
    Query the hosted DeepSeek Coder model.
    Assumes an OpenAI-compatible API structure.
    """
    _setup_hosted_deepseek_client()

    if _hosted_deepseek_client is None:
        err_msg = (
            "Hosted DeepSeek Coder client not initialized. Check API key and base URL."
        )
        logger.error(err_msg)
        return f"ERROR: {err_msg}", 0, 0, 0, {"error": err_msg}

    # !! REPLACE WITH THE MODEL ID YOUR PROVIDER EXPECTS FOR DEEPSEEK CODER !!
    # This should match what you set in your AIDE config for agent.code.model
    actual_model_id_for_provider = model or "deepseek-chat"

    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )

    # Prepare API parameters (common OpenAI-compatible ones)
    api_params = {
        "temperature": temperature,
        "max_tokens": model_kwargs.get(
            "max_new_tokens", model_kwargs.get("max_tokens")
        ),  # Accommodate both
        "top_p": model_kwargs.get("top_p"),
        "stop": model_kwargs.get("stop"),
        "frequency_penalty": model_kwargs.get("frequency_penalty"),
        "presence_penalty": model_kwargs.get("presence_penalty"),
        # Add other parameters if your provider supports them (e.g., seed)
    }
    filtered_api_params = {k: v for k, v in api_params.items() if v is not None}
    filtered_api_params["model"] = (
        actual_model_id_for_provider  # Use the provider-specific ID
    )

    logger.debug(
        f"Calling Hosted DeepSeek Coder API ({actual_model_id_for_provider}) with params: {filtered_api_params}",
        extra={"verbose": True},
    )

    t0 = time.time()
    completion = None
    try:
        completion = backoff_create(
            _hosted_deepseek_client.chat.completions.create,
            HOSTED_DEEPSEEK_API_EXCEPTIONS,  # Use specific exceptions for this provider if different
            messages=messages,
            **filtered_api_params,
        )
    except Exception as e:
        logger.error(f"Hosted DeepSeek Coder query failed: {e}", exc_info=True)
        return f"ERROR: API call failed: {e}", time.time() - t0, 0, 0, {"error": str(e)}

    req_time = time.time() - t0

    if not completion or not completion.choices:
        err_msg = "Hosted DeepSeek Coder API call returned empty or invalid completion object."
        logger.error(err_msg)
        return f"ERROR: {err_msg}", req_time, 0, 0, {"error": err_msg}

    choice = completion.choices[0]
    output = choice.message.content or ""  # Generated text

    input_tokens = completion.usage.prompt_tokens if completion.usage else 0
    output_tokens = completion.usage.completion_tokens if completion.usage else 0

    info = {
        "model_used_by_provider": completion.model,  # What the provider reports
        "finish_reason": choice.finish_reason,
        "id": completion.id,
        "created": completion.created,
        "provider_base_url": "hm",
    }
    logger.debug(
        f"Hosted DeepSeek Coder response tokens: In={input_tokens}, Out={output_tokens}",
        extra={"verbose": True},
    )
    return output, req_time, input_tokens, output_tokens, info
