import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from aide.utils.config import load_cfg # Assuming this path is correct relative to your project structure

from . import (
    backend_deepseek,
    backend_local,
    backend_openai,
    backend_vllm,
)
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

# --- Configuration & Globals ---
# It's good practice to load config once or pass it around explicitly.
# If cfg can change during runtime for different calls, consider passing it to query.
# For now, assuming cfg.inference_engine is a global default for the query function.
_DEFAULT_CONFIG = load_cfg()
logger = logging.getLogger("aide.backend") # Standard logger name

# Mapping from provider names to their query function implementations
_PROVIDER_QUERY_FUNCTIONS: Dict[str, Callable[..., Tuple[OutputType, float, int, int, Dict[str, Any]]]] = {
    "openai": backend_openai.query,
    "vllm": backend_vllm.query,
    "deepseek": backend_deepseek.query,
    "hf": backend_local.query, # Renamed "HF" to "hf" for consistency
}

# Configuration for determining provider based on model name prefixes
# Order matters: more specific prefixes should come before more general ones if overlap is possible.
_MODEL_PREFIX_TO_PROVIDER_MAP: Dict[str, Tuple[str, ...]] = {
    "openai": ("gpt-", "o4-", "o3-"),
    "deepseek": ("deepseek-", "wojtek/"),
    # Add other providers and their prefixes here
    # e.g., "another_provider": ("another-prefix-",)
}
_DEFAULT_MODEL_PROVIDER = "hf" # Fallback provider if no prefix matches


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
    default_inference_engine: Optional[str] = _DEFAULT_CONFIG.inference_engine
) -> str:
    """
    Resolves the final provider name, considering explicit engine requests and model name.
    """
    # Use requested_inference_engine if provided, otherwise default from config
    current_inference_engine = requested_inference_engine if requested_inference_engine is not None else default_inference_engine

    # VLLM engine override:
    # If VLLM is the chosen engine AND the model isn't one that should stick to OpenAI's API directly
    # (like o3, o4, gpt- which might have special handling or features not via a generic VLLM endpoint)
    is_openai_direct_model = any(model_name.startswith(p) for p in _MODEL_PREFIX_TO_PROVIDER_MAP.get("openai", ())) # Use map dynamically
    is_deepseek_direct_model = any(model_name.startswith(p) for p in _MODEL_PREFIX_TO_PROVIDER_MAP.get("deepseek", ())) # Use map dynamically

    if current_inference_engine == "vllm" and not is_openai_direct_model and not is_deepseek_direct_model:
        # If it's VLLM engine and NOT an explicit OpenAI or DeepSeek API model, force VLLM
        return "vllm"
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
    final_kwargs["model"] = model_name # Ensure model name is always in kwargs

    # Max tokens parameter name varies by provider/model type
    if max_tokens is not None:
        if provider_name == "openai":
            if model_name.startswith(("o3-", "o4-")):
                final_kwargs["max_completion_tokens"] = max_tokens
            elif model_name.startswith("gpt-"):
                final_kwargs["max_tokens"] = max_tokens
            # Potentially other OpenAI models might have different names, but these are common
        elif provider_name in ("hf", "vllm", "deepseek"): # Assuming these use 'max_new_tokens'
            final_kwargs["max_new_tokens"] = max_tokens
        # else:
            # Add other provider-specific max_tokens names if necessary
            # logger.warning(f"Max tokens specified, but unsure of param name for provider '{provider_name}'.")

    # Reasoning effort is specific to certain OpenAI models
    if provider_name == "openai" and model_name.startswith(("o3-", "o4-")):
        if reasoning_effort:
            final_kwargs["reasoning_effort"] = reasoning_effort

    return final_kwargs


# --- Main Query Dispatcher ---

def query(
    system_message: Optional[PromptType],
    user_message: Optional[PromptType],
    model: str,
    max_tokens: Optional[int] = None,
    func_spec: Optional[FunctionSpec] = None,
    convert_system_to_user: bool = False,
    inference_engine: Optional[str] = None, # Allow overriding global cfg.inference_engine
    planner: bool = False, # Passed through to backend
    current_step: int = 0, # Used for logging/tracing
    reasoning_effort: Optional[str] = None,
    **model_kwargs: Any,
) -> OutputType:
    """
    General LLM query dispatcher for various backends.

    Args:
        system_message: Uncompiled system message.
        user_message: Uncompiled user message.
        model: String identifier for the model (e.g., "gpt-4-turbo", "deepseek-coder").
        max_tokens: Maximum number of tokens to generate.
        func_spec: Optional FunctionSpec for function calling.
        convert_system_to_user: If True, system message is sent as a user message.
        inference_engine: Explicitly set inference engine (e.g., "vllm", "hf").
                          Overrides default from `cfg.inference_engine`.
        planner: Flag passed to the backend query function.
        current_step: Current step number, used for generating a step_identifier.
        reasoning_effort: Specific parameter for some OpenAI models.
        **model_kwargs: Additional keyword arguments for the specific model/backend.

    Returns:
        The output from the queried backend (string or dict for function call).
    """
    provider_name = _resolve_provider(model, requested_inference_engine=inference_engine)
    final_model_kwargs = _prepare_model_kwargs(
        provider_name, model, max_tokens, reasoning_effort, model_kwargs
    )

    # Compile messages
    compiled_system_message = compile_prompt_to_md(system_message) if system_message else None
    compiled_user_message = compile_prompt_to_md(user_message) if user_message else None

    # Logging
    logger.info(
        f"Dispatching query to provider: '{provider_name}', model: '{model}'"
    )
    if logger.isEnabledFor(logging.DEBUG): # Avoid expensive operations if not debugging
        logger.debug(f"  Final model kwargs: {final_model_kwargs}")
        if compiled_system_message:
            logger.info(f"  System message: {compiled_system_message[:500]}...", extra={"verbose": True})
        if compiled_user_message:
            logger.info(f"  User message: {compiled_user_message[:500]}...", extra={"verbose": True})
        if func_spec:
            logger.info(f"  Function spec: {func_spec.to_dict()}",extra={"verbose": True})

    # Get the appropriate query function
    query_func = _PROVIDER_QUERY_FUNCTIONS.get(provider_name)
    if not query_func:
        logger.error(f"No query function found for provider: {provider_name}")
        raise ValueError(f"Unsupported provider: {provider_name}")

    step_identifier = f"Step_{current_step}_Model_{model.replace('/', '_')}" # More descriptive identifier

    # Execute the query
    logger.info(f"[Timing] Pre-query for {step_identifier}")
    t0 = time.time()
    try:
        raw_responses, latency, input_tokens, output_tokens, info = query_func(
            system_message=compiled_system_message,
            user_message=compiled_user_message,
            func_spec=func_spec,
            planner=planner, # Pass through
            convert_system_to_user=convert_system_to_user,
            step_identifier=step_identifier,
            **final_model_kwargs,
        )
    except Exception as e:
        logger.error(f"Error during query to provider {provider_name} for model {model}: {e}", exc_info=True)
        # Depending on desired behavior, you might re-raise or return an error object
        raise # Re-raise for now to maintain original behavior for errors
    
    query_duration = time.time() - t0
    logger.info(f"[Timing] Post-query for {step_identifier}. Backend latency: {latency:.3f}s, Total dispatch: {query_duration:.3f}s")
    logger.info(f"  Tokens In/Out: {input_tokens}/{output_tokens}")


    if logger.isEnabledFor(logging.DEBUG): # Log full response only in debug
        if func_spec:
            logger.debug(f"  Function call response: {raw_responses}")
        else:
            # Be careful with logging potentially very long raw string responses
            logger.debug(f"  String response: {str(raw_responses)[:1000]}...")
    
    logger.info(f"Query for {step_identifier} complete.")
    return raw_responses