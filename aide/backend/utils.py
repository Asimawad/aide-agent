
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple, Type # Added Type
import re
import jsonschema
from dataclasses_json import DataClassJsonMixin
import openai # Ensure openai is imported for openai.BadRequestError

import backoff
PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

logger = logging.getLogger("aide")

# --- Custom Exception ---
class ContextLengthExceededError(Exception):
    """Custom exception for context length errors."""
    pass


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: Tuple[Type[Exception], ...], *args, **kwargs 
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        if isinstance(e, openai.BadRequestError):
            error_body = getattr(e, 'body', None)
            error_message_from_body = ""
            if isinstance(error_body, dict):
                error_message_from_body = error_body.get('message', '')
            
            error_message_attr = getattr(e, 'message', '') # For some openai.APIError instances
            full_error_str = str(e)

            is_context_length_error = (
                "maximum context length" in error_message_from_body.lower() or
                "maximum context length" in error_message_attr.lower() or
                "maximum context length" in full_error_str.lower() or
                "context_length_exceeded" in full_error_str.lower() # Some APIs use this code
            )

            if is_context_length_error:
                logger.error(f"Context length exceeded: {full_error_str}. Not retrying with backoff.")
                # Re-raise as a custom error to stop backoff and allow specific handling upstream
                raise ContextLengthExceededError(f"Context length exceeded: {full_error_str}") from e
        

        logger.info(f"Backoff-eligible exception: {e}. Retrying...")
        return False #
    except Exception as e: 
        logger.error(f"Non-retryable exception during create_fn: {e}", exc_info=True)
        raise 

def opt_messages_to_list(
    system_message: str | None,
    user_message: str | None,
    convert_system_to_user: bool = False,
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        if convert_system_to_user:
            messages.append({"role": "user", "content": system_message})
        else:
            messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


# def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
#     if isinstance(prompt, str):
#         return prompt.strip() + "\n"
#     elif isinstance(prompt, list):
#         return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

#     out = []
#     header_prefix = "#" * _header_depth
#     for k, v in prompt.items():
#         out.append(f"{header_prefix} {k}\n")
#         out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
#     return "\n".join(out)

# aide/backend/utils.py

def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1, is_list_item: bool = False) -> str:
    if isinstance(prompt, str):
        if is_list_item:
            return f"- {prompt.strip()}" # Format as list item if it's a string directly in a list
        return prompt.strip() + "\n"
    
    elif isinstance(prompt, list):
        list_items_compiled = []
        for item in prompt:
            if isinstance(item, str):
                list_items_compiled.append(compile_prompt_to_md(item, _header_depth, is_list_item=True))
            else:
                compiled_item = compile_prompt_to_md(item, _header_depth) # Keep same header depth or indent?
                list_items_compiled.append(compiled_item)
        
        return "\n".join(list_items_compiled) # No need to add extra "\n" if items handle it

    elif isinstance(prompt, dict):
        out = []
        header_prefix = "#" * _header_depth
        indentation = "  " * (_header_depth -1) if is_list_item and _header_depth > 1 else ""

        for k, v in prompt.items():
            out.append(f"{indentation}{header_prefix} {k}\n")
            compiled_value = compile_prompt_to_md(v, _header_depth=_header_depth + 1)
            indented_compiled_value = "\n".join([f"{indentation}{line}" for line in compiled_value.splitlines()])
            out.append(indented_compiled_value)
        return "\n".join(out).strip() + "\n" # Ensure a trailing newline for dicts
    else:
        logger.warning(f"compile_prompt_to_md received an unexpected type: {type(prompt)}. Value: {prompt}")
        return str(prompt) + "\n"

def _split_prompt(
    system_message: Optional[str], user_message: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Split a long system_message into system and user parts if user_message is None."""
    if user_message or not system_message:
        return system_message, user_message


    task_match = re.search(
        r"(# Task description[\s\S]*?)(# Instructions|$)(user|$)",
        system_message,
        re.DOTALL,
    )
    if task_match:
        system_part = system_message[: task_match.start()]  # Up to task description
        user_part = task_match.group(1).strip()  # Task description and beyond
        logger.info("Split system_message into system and user parts")
        return system_part.strip(), user_part
    else:
        # Fallback: Use system_message as-is, warn about potential issues
        logger.warning("Could not split system_message; treating as system prompt only")
        return system_message, None


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }
