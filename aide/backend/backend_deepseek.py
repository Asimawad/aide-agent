"""Backend for OpenRouter API"""

import logging
import os
import time
from typing import Optional, Dict, Any, Tuple

from funcy import notnone, once, select_values
import openai

from .utils import FunctionSpec, OutputType, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
    )


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

    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    if func_spec is not None:
        raise NotImplementedError(
            "We are not supporting function calling in OpenRouter for now."
        )

    # in case some backends dont support system roles, just convert everything to user
    messages = [
        {"role": "user", "content": message}
        for message in [system_message, user_message]
        if message
    ]

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body={
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    output = completion.choices[0].message.content

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info


# # aide-ds-main/aide/backend/backend_hosted_deepseek.py
# import json
# import logging
# import time
# import os
# from typing import Optional, Dict, Any, Tuple

# import openai  # We'll use the openai library if the API is OpenAI-compatible

# from aide.backend.utils import (
#     FunctionSpec,  # Though likely not used by coder
#     OutputType,
#     opt_messages_to_list,
#     backoff_create,
# )
# from funcy import notnone, once, select_values

# logger = logging.getLogger("aide")

# # --- Configuration for the Hosted DeepSeek Coder API ---
# # Fetch these from environment variables
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# # !! REPLACE THE DEFAULT VALUE WITH YOUR PROVIDER'S ACTUAL BASE URL !!

# # Global client instance for this backend
# _hosted_deepseek_client: Optional[openai.OpenAI] = None

# # Define API exceptions that should trigger backoff/retries for this provider
# # These are often similar to OpenAI's but check your provider's documentation
# HOSTED_DEEPSEEK_API_EXCEPTIONS = (
#     openai.APIConnectionError,
#     openai.RateLimitError,
#     openai.APITimeoutError,
#     openai.APIError,
#     openai.InternalServerError,  # Add others if specified by your provider
# )


# @once
# def _setup_hosted_deepseek_client():
#     """Sets up the OpenAI-compatible client for the hosted DeepSeek Coder service."""
#     global _hosted_deepseek_client
#     if not OPENROUTER_API_KEY:
#         logger.error(
#             "HOSTED_DEEPSEEK_API_KEY environment variable not set. This backend cannot function."
#         )
#         # You could raise an error here or let it fail when _hosted_deepseek_client is used
#         return

#     logger.info(
#         f"Setting up Hosted DeepSeek Coder client with base_url",
#         extra={"verbose": True},
#     )
#     try:
#         _hosted_deepseek_client = openai.OpenAI(
#             base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY
#         )
#     except Exception as e:
#         logger.error(
#             f"Failed to setup Hosted DeepSeek Coder client: {e}", exc_info=True
#         )
#         _hosted_deepseek_client = None  # Ensure it's None on failure
#         raise


# def query(
#     system_message: Optional[str] = None,
#     user_message: Optional[str] = None,
#     model: Optional[
#         str
#     ] = None,  # Model ID will be passed, ensure it's the correct one for this provider
#     temperature: float = 0.6,  # Default temperature for coder
#     func_spec: Optional[FunctionSpec] = None,  # Typically None for coder
#     convert_system_to_user: bool = False,
#     max_retries: int = 3,
#     **model_kwargs: Any,
# ) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
#     """
#     Query the hosted DeepSeek Coder model.
#     Assumes an OpenAI-compatible API structure.
#     """
#     _setup_hosted_deepseek_client()

#     if _hosted_deepseek_client is None:
#         err_msg = (
#             "Hosted DeepSeek Coder client not initialized. Check API key and base URL."
#         )
#         logger.error(err_msg)
#         return f"ERROR: {err_msg}", 0, 0, 0, {"error": err_msg}

#     # !! REPLACE WITH THE MODEL ID YOUR PROVIDER EXPECTS FOR DEEPSEEK CODER !!
#     # This should match what you set in your AIDE config for agent.code.model
#     actual_model_id_for_provider = model or "YOUR_HOSTED_DEEPSEEK_CODER_MODEL_ID"
#     if "YOUR_HOSTED_DEEPSEEK_CODER_MODEL_ID" == actual_model_id_for_provider:
#         logger.warning(
#             "Using placeholder model ID for hosted DeepSeek. Ensure 'model' kwarg is passed or placeholder is updated."
#         )

#     messages = opt_messages_to_list(
#         system_message, user_message, convert_system_to_user=convert_system_to_user
#     )

#     # Prepare API parameters (common OpenAI-compatible ones)
#     api_params = {
#         "temperature": temperature,
#         "max_tokens": model_kwargs.get(
#             "max_new_tokens", model_kwargs.get("max_tokens")
#         ),  # Accommodate both
#         "top_p": model_kwargs.get("top_p"),
#         "stop": model_kwargs.get("stop"),
#         "frequency_penalty": model_kwargs.get("frequency_penalty"),
#         "presence_penalty": model_kwargs.get("presence_penalty"),
#         # Add other parameters if your provider supports them (e.g., seed)
#     }
#     filtered_api_params = {k: v for k, v in api_params.items() if v is not None}
#     filtered_api_params["model"] = (
#         actual_model_id_for_provider  # Use the provider-specific ID
#     )

#     logger.debug(
#         f"Calling Hosted DeepSeek Coder API ({actual_model_id_for_provider}) with params: {filtered_api_params}",
#         extra={"verbose": True},
#     )

#     t0 = time.time()
#     completion = None
#     try:
#         completion = backoff_create(
#             _hosted_deepseek_client.chat.completions.create,
#             HOSTED_DEEPSEEK_API_EXCEPTIONS,  # Use specific exceptions for this provider if different
#             messages=messages,
#             **filtered_api_params,
#         )
#     except Exception as e:
#         logger.error(f"Hosted DeepSeek Coder query failed: {e}", exc_info=True)
#         return f"ERROR: API call failed: {e}", time.time() - t0, 0, 0, {"error": str(e)}

#     req_time = time.time() - t0

#     if not completion or not completion.choices:
#         err_msg = "Hosted DeepSeek Coder API call returned empty or invalid completion object."
#         logger.error(err_msg)
#         return f"ERROR: {err_msg}", req_time, 0, 0, {"error": err_msg}

#     choice = completion.choices[0]
#     output = choice.message.content or ""  # Generated text

#     input_tokens = completion.usage.prompt_tokens if completion.usage else 0
#     output_tokens = completion.usage.completion_tokens if completion.usage else 0

#     info = {
#         "model_used_by_provider": completion.model,  # What the provider reports
#         "finish_reason": choice.finish_reason,
#         "id": completion.id,
#         "created": completion.created,
#         "provider_base_url": "hm",
#     }
#     logger.debug(
#         f"Hosted DeepSeek Coder response tokens: In={input_tokens}, Out={output_tokens}",
#         extra={"verbose": True},
#     )
#     return output, req_time, input_tokens, output_tokens, info


# # aide-ds-main/aide/backend/backend_hosted_deepseek.py
# import json
# import logging
# import time
# import os
# from typing import Optional, Dict, Any, Tuple

# import openai # We'll use the openai library if the API is OpenAI-compatible

# from aide.backend.utils import (
#     FunctionSpec, # Though likely not used by coder
#     OutputType,
#     opt_messages_to_list,
#     backoff_create,
# )
# from funcy import notnone, once, select_values

# logger = logging.getLogger("aide")

# # --- Configuration for the Hosted DeepSeek Coder API ---
# # Fetch these from environment variables
# HOSTED_DEEPSEEK_API_KEY = os.getenv("HOSTED_DEEPSEEK_API_KEY")
# # !! REPLACE THE DEFAULT VALUE WITH YOUR PROVIDER'S ACTUAL BASE URL !!
# HOSTED_DEEPSEEK_BASE_URL = os.getenv("HOSTED_DEEPSEEK_BASE_URL", "YOUR_PROVIDER_BASE_URL_HERE/v1")

# # Global client instance for this backend
# _hosted_deepseek_client: Optional[openai.OpenAI] = None

# # Define API exceptions that should trigger backoff/retries for this provider
# # These are often similar to OpenAI's but check your provider's documentation
# HOSTED_DEEPSEEK_API_EXCEPTIONS = (
#     openai.APIConnectionError,
#     openai.RateLimitError,
#     openai.APITimeoutError,
#     openai.APIError,
#     openai.InternalServerError, # Add others if specified by your provider
# )

# @once
# def _setup_hosted_deepseek_client():
#     """Sets up the OpenAI-compatible client for the hosted DeepSeek Coder service."""
#     global _hosted_deepseek_client
#     if not HOSTED_DEEPSEEK_API_KEY:
#         logger.error("HOSTED_DEEPSEEK_API_KEY environment variable not set. This backend cannot function.")
#         # You could raise an error here or let it fail when _hosted_deepseek_client is used
#         return

#     if "YOUR_PROVIDER_BASE_URL_HERE" in HOSTED_DEEPSEEK_BASE_URL:
#         logger.error("HOSTED_DEEPSEEK_BASE_URL is not correctly set in backend_hosted_deepseek.py. Please replace placeholder.")
#         # Consider raising an error
#         return

#     logger.info(
#         f"Setting up Hosted DeepSeek Coder client with base_url: {HOSTED_DEEPSEEK_BASE_URL}",
#         extra={"verbose": True},
#     )
#     try:
#         _hosted_deepseek_client = openai.OpenAI(
#             base_url=HOSTED_DEEPSEEK_BASE_URL,
#             api_key=HOSTED_DEEPSEEK_API_KEY,
#             max_retries=0,  # Rely on backoff_create for retries
#         )
#     except Exception as e:
#         logger.error(f"Failed to setup Hosted DeepSeek Coder client: {e}", exc_info=True)
#         _hosted_deepseek_client = None # Ensure it's None on failure
#         raise

# def query(
#     system_message: Optional[str] = None,
#     user_message: Optional[str] = None,
#     model: Optional[str] = None, # Model ID will be passed, ensure it's the correct one for this provider
#     temperature: float = 0.6, # Default temperature for coder
#     func_spec: Optional[FunctionSpec] = None, # Typically None for coder
#     convert_system_to_user: bool = False,
#     max_retries: int = 3,
#     **model_kwargs: Any,
# ) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
#     """
#     Query the hosted DeepSeek Coder model.
#     Assumes an OpenAI-compatible API structure.
#     """
#     _setup_hosted_deepseek_client()

#     if _hosted_deepseek_client is None:
#         err_msg = "Hosted DeepSeek Coder client not initialized. Check API key and base URL."
#         logger.error(err_msg)
#         return f"ERROR: {err_msg}", 0, 0, 0, {"error": err_msg}

#     # !! REPLACE WITH THE MODEL ID YOUR PROVIDER EXPECTS FOR DEEPSEEK CODER !!
#     # This should match what you set in your AIDE config for agent.code.model
#     actual_model_id_for_provider = model or "YOUR_HOSTED_DEEPSEEK_CODER_MODEL_ID"
#     if "YOUR_HOSTED_DEEPSEEK_CODER_MODEL_ID" == actual_model_id_for_provider:
#         logger.warning("Using placeholder model ID for hosted DeepSeek. Ensure 'model' kwarg is passed or placeholder is updated.")


#     messages = opt_messages_to_list(
#         system_message, user_message, convert_system_to_user=convert_system_to_user
#     )

#     # Prepare API parameters (common OpenAI-compatible ones)
#     api_params = {
#         "temperature": temperature,
#         "max_tokens": model_kwargs.get("max_new_tokens", model_kwargs.get("max_tokens")), # Accommodate both
#         "top_p": model_kwargs.get("top_p"),
#         "stop": model_kwargs.get("stop"),
#         "frequency_penalty": model_kwargs.get("frequency_penalty"),
#         "presence_penalty": model_kwargs.get("presence_penalty"),
#         # Add other parameters if your provider supports them (e.g., seed)
#     }
#     filtered_api_params = {k: v for k, v in api_params.items() if v is not None}
#     filtered_api_params["model"] = actual_model_id_for_provider # Use the provider-specific ID

#     logger.debug(f"Calling Hosted DeepSeek Coder API ({actual_model_id_for_provider}) with params: {filtered_api_params}", extra={"verbose": True})

#     t0 = time.time()
#     completion = None
#     try:
#         completion = backoff_create(
#             _hosted_deepseek_client.chat.completions.create,
#             HOSTED_DEEPSEEK_API_EXCEPTIONS, # Use specific exceptions for this provider if different
#             messages=messages,
#             **filtered_api_params,
#         )
#     except Exception as e:
#         logger.error(f"Hosted DeepSeek Coder query failed: {e}", exc_info=True)
#         return f"ERROR: API call failed: {e}", time.time() - t0, 0, 0, {"error": str(e)}

#     req_time = time.time() - t0

#     if not completion or not completion.choices:
#         err_msg = "Hosted DeepSeek Coder API call returned empty or invalid completion object."
#         logger.error(err_msg)
#         return f"ERROR: {err_msg}", req_time, 0, 0, {"error": err_msg}

#     choice = completion.choices[0]
#     output = choice.message.content or "" # Generated text

#     input_tokens = completion.usage.prompt_tokens if completion.usage else 0
#     output_tokens = completion.usage.completion_tokens if completion.usage else 0

#     info = {
#         "model_used_by_provider": completion.model, # What the provider reports
#         "finish_reason": choice.finish_reason,
#         "id": completion.id,
#         "created": completion.created,
#         "provider_base_url": HOSTED_DEEPSEEK_BASE_URL,
#     }
#     logger.debug(f"Hosted DeepSeek Coder response tokens: In={input_tokens}, Out={output_tokens}", extra={"verbose": True})
#     return output, req_time, input_tokens, output_tokens, info
