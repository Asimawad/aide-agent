"""Backend for OpenAI API."""

import time
import json
import logging
import time
from dotenv import load_dotenv
import os
from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

# load environment variables
load_dotenv()
_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

from rich.console import Console

console = Console()


@once
def _setup_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = input("Please enter your OpenAI API key: ")
    global _client
    _client = openai.OpenAI(max_retries=0)


def filter_model_kwargs(model: str, kwargs: dict) -> dict:
    """
    Filter and adapt kwargs based on the model being used to prevent invalid parameters.
    Handles parameter renaming and removal for specific models.
    """
    # Remove None values
    filtered_kwargs = select_values(notnone, kwargs)

    # Define valid parameters and renames for each model family
    MODEL_PARAM_SPECS = [
        {
            "prefixes": ("o3-", "o4-"),
            "valid_params": {
                "model",
                "n",
                "stream",
                "stop",
                "max_completion_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
                "reasoning_effort",
            },
            "renames": {"max_completion_tokens": "max_tokens"},
            "remove": {"temperature", "top_p"},  # Not supported
        },
        {
            "prefixes": ("gpt-"),
            "valid_params": {
                "model",
                "top_p",
                "n",
                "stream",
                "stop",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
                "response_format",
                "seed",
                "temperature",
            },
            "renames": {},
            "remove": set(),
        },
        {
            "prefixes": (),
            "valid_params": {
                "model",
                "top_p",
                "n",
                "stream",
                "stop",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
            },
            "renames": {},
            "remove": set(),
        },
    ]

    # Find the spec for this model
    for spec in MODEL_PARAM_SPECS:
        if any(model.startswith(prefix) for prefix in spec["prefixes"]):
            valid_params = spec["valid_params"]
            renames = spec["renames"]
            remove = spec["remove"]
            break
    else:
        # Default to the last spec if no prefix matches
        valid_params = MODEL_PARAM_SPECS[-1]["valid_params"]
        renames = MODEL_PARAM_SPECS[-1]["renames"]
        remove = MODEL_PARAM_SPECS[-1]["remove"]

    # Remove unsupported parameters
    for param in remove:
        if param in filtered_kwargs:
            filtered_kwargs.pop(param)
            logger.debug(
                f"Removed '{param}' parameter for model {model} as it's not supported",
                extra={"verbose": True},
            )

    # Filter and rename parameters
    result = {}
    for k, v in filtered_kwargs.items():
        if k in valid_params:
            result[k] = v
        elif k in renames and renames[k] in valid_params:
            result[renames[k]] = v
            logger.debug(
                f"Renamed '{k}' to '{renames[k]}' for model {model}",
                extra={"verbose": True},
            )
        else:
            logger.debug(
                f"Ignored invalid parameter '{k}' for model {model}",
                extra={"verbose": True},
            )

    # Log which parameters were removed
    removed_params = set(filtered_kwargs.keys()) - set(result.keys())
    if removed_params:
        logger.debug(
            f"Removed invalid parameters for model {model}: {removed_params}",
            extra={"verbose": True},
        )
    return result


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    excute: bool = False,
    step_identifier=None,
    planner=False,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    logger.info("activated openai backend...")

    t0 = time.time()
    _setup_openai_client()
    logger.debug(f"[Timing] _setup_openai_client took: {time.time() - t0:.3f}s")
    t0 = time.time()
    # Prepare messages list for OpenAI API format
    model = model_kwargs.get("model", "")
    filtered_kwargs = filter_model_kwargs(model, model_kwargs)

    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    logger.debug(
        f"Calling OpenAI API with model {model} and parameters: {filtered_kwargs}",
        extra={"verbose": True},
    )

    t0 = time.time()
    try:
        completion = backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        error_info = {"error": str(e), "model": model}
        return (
            f"ERROR: OpenAI API call failed: {str(e)}",
            time.time() - t0,
            0,
            0,
            error_info,
        )

    req_time = time.time() - t0

    choice = completion.choices[0]
    logger.debug(f"[Timing] backoff_create end: {time.time() - t0:.3f}s")
    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
            # console.rule(f"[green]Excution Feedback")
            logger.debug(f"Response of the feedback is {output}")
            logger.debug("\n")
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e
    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
        "execution_summaries": "None",
    }

    return output, req_time, in_tokens, out_tokens, info
