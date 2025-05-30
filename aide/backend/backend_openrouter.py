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
