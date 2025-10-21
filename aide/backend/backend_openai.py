"""Backend for OpenAI API with multi-response support."""

import json
import logging
import os
import time
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv
from funcy import notnone, once, select_values

from aide.backend.utils import (
    ContextLengthExceededError,
    FunctionSpec,
    OutputType,  # Union[str, dict]
    backoff_create,
    opt_messages_to_list,
)

logger = logging.getLogger("aide")

# Load environment variables
load_dotenv()
_client: openai.OpenAI = None  # will be initialized in _setup_openai_client

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    """Initialize the OpenAI client, prompting for API key if necessary."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found; prompting user.")
        os.environ["OPENAI_API_KEY"] = input("Please enter your OpenAI API key: ")
    global _client
    _client = openai.OpenAI(max_retries=0)


def filter_model_kwargs(model: str, kwargs: dict) -> dict:
    """
    Filter and adapt kwargs based on the model.
    Ensures only supported parameters are passed, renaming/removing as needed.
    """
    # Drop None values
    filtered = select_values(notnone, kwargs)

    # Define per-model parameter specs
    SPEC_LIST = [
        # Anthropic-style via OpenAI (o3-, o4-): no 'n', use max_completion_tokens → max_tokens
        {
            "prefixes": ("o3-", "o4-"),
            "valid": {
                "model",
                "stream",
                "stop",
                "max_completion_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
                "reasoning_effort",
                "n",
            },
            "renames": {"max_completion_tokens": "max_tokens"},
            "remove": {"temperature", "top_p"},
        },
        # GPT family: supports 'n'
        {
            "prefixes": ("gpt-",),
            "valid": {
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
        # Fallback: assume supports 'n'
        {
            "prefixes": (),
            "valid": {
                "model",
                "top_p",
                "n",
                "stream",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
            },
            "renames": {},
            "remove": set(),
        },
    ]

    # Pick spec
    spec = SPEC_LIST[-1]
    for s in SPEC_LIST:
        if any(model.startswith(pref) for pref in s["prefixes"]):
            spec = s
            break

    # Remove unsupported params
    for p in spec["remove"]:
        if p in filtered:
            filtered.pop(p)
            logger.debug(f"Removed '{p}' for model {model}")

    # Build result with valid and renamed keys
    result: dict = {}
    for k, v in filtered.items():
        if k in spec["valid"]:
            result[k] = v
        elif k in spec["renames"] and spec["renames"][k] in spec["valid"]:
            result[spec["renames"][k]] = v
            logger.debug(f"Renamed '{k}'→'{spec['renames'][k]}' for model {model}")

    # Log dropped params
    dropped = set(filtered.keys()) - set(result.keys())
    if dropped:
        logger.debug(f"Dropped params for {model}: {dropped}")

    return result


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    logger.info("activated openai backend...")

    """
    Send a chat completion request, possibly returning multiple outputs.

    Returns:
      - outputs: list of OutputType (str or dict) of length == number requested
      - elapsed_time: seconds spent on API call
      - prompt_tokens: tokens consumed by prompt
      - completion_tokens: tokens produced by completion(s)
      - info: metadata dict
    """
    t0 = time.time()
    _setup_openai_client()
    model_kwargs["n"] = model_kwargs.get("num_responses", 1)
    model = model_kwargs.get("model", "")
    filtered = filter_model_kwargs(model, model_kwargs)

    # Determine how many responses we asked for
    num_req = filtered.get("n", 1)

    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )
    if func_spec:
        filtered["tools"] = [func_spec.as_openai_tool_dict]
        filtered["tool_choice"] = func_spec.openai_tool_choice_dict

    logger.debug(f"Calling OpenAI model={model} params={filtered}", extra={"verbose": True})

    try:
        completion = backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered,
        )

    except ContextLengthExceededError as cle:
        logger.error(f"ContextLengthExceededError: {cle}")
        err_list = ["ERROR: context length exceeded"] * num_req
        return err_list, time.time() - t0, 0, 0, {"model": model, "error": str(cle)}
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}", exc_info=True)
        err_list = [f"ERROR: {e}"] * num_req
        return err_list, time.time() - t0, 0, 0, {"model": model, "error": str(e)}

    elapsed = time.time() - t0
    choices = completion.choices or []
    outputs: List[OutputType] = []

    for idx, choice in enumerate(choices):
        if func_spec is None:
            content = choice.message.content
            if content is None:
                logger.warning(f"Choice {idx} has no content")
                outputs.append("ERROR: no content")
            else:
                outputs.append(content)
        else:
            calls = choice.message.tool_calls or []
            if not calls or calls[0].function.name != func_spec.name:
                logger.warning(f"Choice {idx} missing expected tool call")
                outputs.append({"error": "tool call missing or mismatched"})
            else:
                try:
                    args = json.loads(calls[0].function.arguments)
                    outputs.append(args)
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error: {je}")
                    outputs.append(
                        {"error": "invalid JSON in tool args", "raw": calls[0].function.arguments}
                    )

    # If fewer outputs than requested, pad with errors
    while len(outputs) < num_req:
        logger.warning(f"Padding response: expected {num_req}, got {len(outputs)}")
        outputs.append("ERROR: missing response")

    # Token counts (aggregate over all choices)
    prompt_toks = getattr(completion.usage, "prompt_tokens", 0)
    comp_toks = getattr(completion.usage, "completion_tokens", 0)

    info: Dict[str, Any] = {
        "model_used": completion.model,
        "created": getattr(completion, "created", None),
        "num_requested": num_req,
        "num_returned": len(choices),
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
    }

    return outputs, elapsed, prompt_toks, comp_toks, info
