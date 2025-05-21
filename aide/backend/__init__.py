import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple
import json
import openai
from aide.utils.config import (
    load_cfg,
)  # Assuming this path is correct relative to your project structure

from . import (
    backend_deepseek,
    backend_local,
    backend_openai,
    backend_vllm,
    backend_ollama,
)
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

# --- Configuration & Globals ---
# It's good practice to load config once or pass it around explicitly.
# If cfg can change during runtime for different calls, consider passing it to query.
# For now, assuming cfg.inference_engine is a global default for the query function.
_DEFAULT_CONFIG = load_cfg()
logger = logging.getLogger("aide.backend")  # Standard logger name

# Mapping from provider names to their query function implementations
_PROVIDER_QUERY_FUNCTIONS: Dict[
    str, Callable[..., Tuple[OutputType, float, int, int, Dict[str, Any]]]
] = {
    "openai": backend_openai.query,
    "vllm": backend_vllm.query,
    "deepseek": backend_deepseek.query,
    "ollama": backend_ollama.query,  # Assuming Ollama uses OpenAI's API
    "hf": backend_local.query,  # Renamed "HF" to "hf" for consistency
}

# Configuration for determining provider based on model name prefixes
# Order matters: more specific prefixes should come before more general ones if overlap is possible.
_MODEL_PREFIX_TO_PROVIDER_MAP: Dict[str, Tuple[str, ...]] = {
    "openai": ("gpt-", "o4-", "o3-"),
    "deepseek": ("deepseek", "wojtek/"),
    # Add other providers and their prefixes here
    # e.g., "another_provider": ("another-prefix-",)
}
_DEFAULT_MODEL_PROVIDER = "vllm"  # Fallback provider if no prefix matches


# --- Helper Functions ---


def _determine_provider_from_model_name(model_name: str) -> str:
    """Determines the provider based on the model name's prefix."""
    for provider, prefixes in _MODEL_PREFIX_TO_PROVIDER_MAP.items():
        if any(model_name.startswith(p) for p in prefixes):
            return provider
    return _DEFAULT_MODEL_PROVIDER


def _resolve_provider(
    model_name: str,
    requested_inference_engine: Optional[str] = None,
    default_inference_engine: Optional[str] = _DEFAULT_CONFIG.inference_engine,
) -> str:
    """
    Resolves the final provider name, considering explicit engine requests and model name.
    """
    # Use requested_inference_engine if provided, otherwise default from config
    current_inference_engine = (
        requested_inference_engine
        if requested_inference_engine is not None
        else default_inference_engine
    )

    # VLLM engine override:
    # If VLLM is the chosen engine AND the model isn't one that should stick to OpenAI's API directly
    # (like o3, o4, gpt- which might have special handling or features not via a generic VLLM endpoint)
    is_openai_direct_model = any(
        model_name.startswith(p)
        for p in _MODEL_PREFIX_TO_PROVIDER_MAP.get("openai", ())
    )  # Use map dynamically
    is_deepseek_direct_model = any(
        model_name.startswith(p)
        for p in _MODEL_PREFIX_TO_PROVIDER_MAP.get("deepseek", ())
    )  # Use map dynamically

    if (
        current_inference_engine == "vllm"
        and not is_openai_direct_model
        and not is_deepseek_direct_model
    ):
        # If it's VLLM engine and NOT an explicit OpenAI or DeepSeek API model, force VLLM
        return "vllm"
    elif current_inference_engine == "ollama":
        # If the inference engine is explicitly set to OpenAI, use it
        return "ollama"
    # Default determination based on model name
    return _determine_provider_from_model_name(model_name)


def _prepare_model_kwargs(
    provider_name: str,
    model_name: str,
    max_tokens: Optional[int],
    reasoning_effort: Optional[str],
    original_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepares the final keyword arguments for the model query function."""
    final_kwargs = original_kwargs.copy()
    final_kwargs["model"] = model_name  # Ensure model name is always in kwargs

    # Max tokens parameter name varies by provider/model type
    if max_tokens is not None:
        if provider_name == "openai":
            if model_name.startswith(("o3-", "o4-")):
                final_kwargs["max_completion_tokens"] = max_tokens
            elif model_name.startswith("gpt-"):
                final_kwargs["max_tokens"] = max_tokens
            # Potentially other OpenAI models might have different names, but these are common
        elif provider_name in (
            "hf",
            "vllm",
            "deepseek",
        ):  # Assuming these use 'max_new_tokens'
            final_kwargs["max_new_tokens"] = max_tokens
        # else:
        # Add other provider-specific max_tokens names if necessary
        # logger.warning(f"Max tokens specified, but unsure of param name for provider '{provider_name}'.")

    # Reasoning effort is specific to certain OpenAI models
    if provider_name == "openai" and model_name.startswith(("o3-", "o4-")):
        if reasoning_effort:
            final_kwargs["reasoning_effort"] = reasoning_effort

    return final_kwargs


def query(
    system_message: Optional[PromptType],
    user_message: Optional[PromptType],
    model: str,
    max_tokens: Optional[int] = None,
    func_spec: Optional[FunctionSpec] = None,
    convert_system_to_user: bool = False,
    inference_engine: Optional[str] = None,
    planner: bool = False,
    current_step: int = 0,  # Already present, good!
    reasoning_effort: Optional[str] = None,
    **model_kwargs: Any,
) -> OutputType:
    provider_name = _resolve_provider(
        model, requested_inference_engine=inference_engine
    )
    final_model_kwargs = _prepare_model_kwargs(
        provider_name, model, max_tokens, reasoning_effort, model_kwargs
    )

    compiled_system_message = (
        compile_prompt_to_md(system_message) if system_message else None
    )
    compiled_user_message = compile_prompt_to_md(user_message) if user_message else None

    # Enhanced Logging
    log_identifier = f"BACKEND_QUERY_STEP{current_step}_MODEL_{model.replace('/', '_').replace('-', '_')}_PROVIDER_{provider_name}"

    logger.info(f"{log_identifier}: Dispatching query.", extra={"verbose": True})
    # For verbose logs, print the full prompts and func_spec
    # Use json.dumps for better readability of dicts/lists in log files
    if compiled_system_message:
        try:
            # If system_message was a dict, compile_prompt_to_md makes it a string.
            # If you want to log the original dict structure for dict prompts:
            # system_log_content = json.dumps(system_message, indent=2) if isinstance(system_message, (dict, list)) else compiled_system_message
            system_log_content = compiled_system_message  # Current behavior
        except (
            TypeError
        ):  # Handle non-serializable if original dict had complex objects
            system_log_content = str(system_message)
        logger.debug(
            f"{log_identifier}_SYSTEM_MESSAGE_START\n{system_log_content}\n{log_identifier}_SYSTEM_MESSAGE_END",
            extra={"verbose": True},
        )

    if compiled_user_message:
        try:
            # user_log_content = json.dumps(user_message, indent=2) if isinstance(user_message, (dict, list)) else compiled_user_message
            user_log_content = compiled_user_message  # Current behavior
        except TypeError:
            user_log_content = str(user_message)
        logger.debug(
            f"{log_identifier}_USER_MESSAGE_START\n{user_log_content}\n{log_identifier}_USER_MESSAGE_END",
            extra={"verbose": True},
        )

    if func_spec:
        logger.debug(
            f"{log_identifier}_FUNC_SPEC_START\n{json.dumps(func_spec.to_dict(), indent=2)}\n{log_identifier}_FUNC_SPEC_END",
            extra={"verbose": True},
        )

    logger.debug(
        f"{log_identifier}_FINAL_MODEL_KWARGS: {final_model_kwargs}",
        extra={"verbose": True},
    )

    query_func = _PROVIDER_QUERY_FUNCTIONS.get(provider_name)
    if not query_func:
        logger.error(
            f"{log_identifier}: No query function found for provider: {provider_name}",
            extra={"verbose": True},
        )
        raise ValueError(f"Unsupported provider: {provider_name}")

    # step_identifier is already used by you, which is good. Let's ensure it's passed.
    # The 'step_identifier' passed to the backend (e.g. vllm) should be unique and informative.
    # The one you generate (f"Step_{current_step}_Model_{model.replace('/', '_')}") is good.

    t0 = time.time()
    raw_responses, latency, input_tokens, output_tokens, info = (
        "ERROR_DEFAULT",
        0.0,
        0,
        0,
        {},
    )  # Defaults
    try:
        print(f"Querying {provider_name} with model {model}...")
        # Pass current_step to the specific backend if it can use it for more granular logging
        raw_responses, latency, input_tokens, output_tokens, info = query_func(
            system_message=compiled_system_message,
            user_message=compiled_user_message,
            func_spec=func_spec,
            planner=planner,
            convert_system_to_user=convert_system_to_user,
            step_identifier=f"Prov_{provider_name}_Step{current_step}_M_{model.replace('/', '_')}",  # More specific for backend
            # current_step=current_step, # If backend functions accept it
            **final_model_kwargs,
        )
    except Exception as e:
        logger.error(
            f"{log_identifier}: Error during actual provider query: {e}",
            exc_info=True,
            extra={"verbose": True},
        )
        raise  # Re-raise to be handled by agent's retry logic or crash

    query_duration = time.time() - t0
    logger.info(
        f"{log_identifier}: Query completed. Provider Latency: {latency:.3f}s, Total Dispatch: {query_duration:.3f}s, Tokens In/Out: {input_tokens}/{output_tokens}",
        extra={"verbose": True},
    )

    # Log raw response to verbose log
    if func_spec:
        try:
            response_log_content = (
                json.dumps(raw_responses, indent=2)
                if isinstance(raw_responses, (dict, list))
                else str(raw_responses)
            )
        except TypeError:
            response_log_content = str(raw_responses)
        logger.debug(
            f"{log_identifier}_RAW_FUNCTION_RESPONSE_START\n{response_log_content}\n{log_identifier}_RAW_FUNCTION_RESPONSE_END",
            extra={"verbose": True},
        )
    else:
        logger.debug(
            f"{log_identifier}_RAW_TEXT_RESPONSE_START\n{str(raw_responses)}\n{log_identifier}_RAW_TEXT_RESPONSE_END",
            extra={"verbose": True},
        )

    return raw_responses
