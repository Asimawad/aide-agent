import logging
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple
import re
import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

import backoff

logger = logging.getLogger("aide")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


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
        # Recursively compile each item in the list
        # Each item will be formatted appropriately by the recursive call
        # (e.g., as a list item if it's a string, or as a sub-section if it's a dict)
        list_items_compiled = []
        for item in prompt:
            # Pass is_list_item=True if the item itself is a string, 
            # so it gets the "- " prefix.
            # If the item is a dict or another list, the recursive call will handle its structure.
            if isinstance(item, str):
                list_items_compiled.append(compile_prompt_to_md(item, _header_depth, is_list_item=True))
            else:
                # For dicts or nested lists within the list, don't add the bullet point here;
                # let the recursive call handle their formatting (e.g. headers for dicts).
                # We might want to indent them if they are part of a list.
                # For simplicity now, just compile them. You can add indentation logic if needed.
                compiled_item = compile_prompt_to_md(item, _header_depth) # Keep same header depth or indent?
                list_items_compiled.append(compiled_item)
        
        # Join the compiled list items.
        # If the original list items were strings, they already have "- ".
        # If they were dicts, they'll be formatted with headers.
        return "\n".join(list_items_compiled) # No need to add extra "\n" if items handle it

    elif isinstance(prompt, dict):
        out = []
        header_prefix = "#" * _header_depth
        # If this dict is an item within a list, we might want to indent its content
        indentation = "  " * (_header_depth -1) if is_list_item and _header_depth > 1 else ""

        for k, v in prompt.items():
            # Apply indentation to the header if this dict is a list item
            out.append(f"{indentation}{header_prefix} {k}\n")
            # Recursively call for the value, passing down the header depth
            # and an indication if the value itself is a list item (False in this case for dict values)
            # Add indentation to the compiled value as well
            compiled_value = compile_prompt_to_md(v, _header_depth=_header_depth + 1)
            indented_compiled_value = "\n".join([f"{indentation}{line}" for line in compiled_value.splitlines()])
            out.append(indented_compiled_value)
        return "\n".join(out).strip() + "\n" # Ensure a trailing newline for dicts
    else:
        # Handle other unexpected types if necessary, or raise an error
        logger.warning(f"compile_prompt_to_md received an unexpected type: {type(prompt)}. Value: {prompt}")
        return str(prompt) + "\n"

def _split_prompt(
    system_message: Optional[str], user_message: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Split a long system_message into system and user parts if user_message is None."""
    if user_message or not system_message:
        return system_message, user_message

    # Heuristic: Split on '# Task description' or similar to isolate task
    task_match = re.search(
        r"(# Task description[\s\S]*?)(# Instructions|$)(user|$)",
        system_message,
        re.DOTALL,
    )
    if task_match:
        system_part = system_message[: task_match.start()]  # Up to task description
        user_part = task_match.group(1).strip()  # Task description and beyond
        logger.info("Split system_message into system and user parts")
        # logger.info(f"system_part: {system_part[:300]}")
        # logger.info(f"user_part: {user_part[:300]}")
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
