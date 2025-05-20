import logging
from . import (
    backend_local,
    backend_openai,
    backend_vllm,
    backend_deepseek,
)
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from aide.utils.config import load_cfg
cfg = load_cfg()
logger = logging.getLogger("aide")


def determine_provider(model: str) -> str:
    
    if model.startswith("gpt-") or model.startswith("o4-") or model.startswith("o3-"):
        return "openai"
    elif model.startswith("deepseek-"):
        return "deepseek"
    else:
        return "HF"


provider_to_query_func = {
    "openai": backend_openai.query,
    "vllm": backend_vllm.query,
    "deepseek": backend_deepseek.query,
    "HF": backend_local.query,
}


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    inference_engine:str|None = cfg.inference_engine,
    excute: bool = False,
    current_step:int=0,
    reasoning_effort: str | None = None,
    **model_kwargs,  # his mean that you can pass model specific keyword arguments as kwargs
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """
    if inference_engine == "vllm" and not (model.startswith("o3-") or model.startswith("o4-") or model.startswith("gpt-")):
        provider = "vllm"
    else:
        provider = determine_provider(model) 

    # Handle provider-specific parameters
    if provider == "openai" and (model.startswith("o3-") or model.startswith("o4-")):
        # Handle reasoning models (o3- and o4-)
        if max_tokens is not None:
            model_kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            model_kwargs["reasoning_effort"] = reasoning_effort
    elif provider == "openai" and model.startswith("gpt-"):
        # Standard GPT models
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
    elif provider == "HF" or provider == "vllm":
        # Standard parameters for other models
        if max_tokens is not None:
            model_kwargs["max_new_tokens"] = max_tokens

    # Ensure model is set in the kwargs
    model_kwargs["model"] = model
    
    logger.info(f"Querying model with arguments: {model_kwargs}, Model: {model}, Provider: {provider}", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        logger.info(f"System message: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        logger.info(f"User message: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"Function spec: {func_spec.to_dict()}", extra={"verbose": True})

    query_func = provider_to_query_func[provider]
    logger.info(f"Using model {model} with backend {provider}", extra={"verbose": True})
    step_id = f"Draft_{current_step}" # Example
    # output, req_time, in_tok_count, out_tok_count, info 
    raw_responses, latency, input_token_count, output_token_count, info = query_func(
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        convert_system_to_user=convert_system_to_user,
        step_identifier=step_id,
        **model_kwargs,
    )
    logger.info(f"Response: {raw_responses}", extra={"verbose": True})
    logger.info("Query complete", extra={"verbose": True})
    return raw_responses
