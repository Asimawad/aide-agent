import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects



def trim_long_string(string: str, threshold: int = 3000, k: int = 1200) -> str: # Reduced defaults
    """
    Trims a long string to a specified threshold by showing the first k and last k characters.
    Default threshold and k values are reduced to be more conservative with token counts.
    """
    if not isinstance(string, str): # Handle non-string inputs gracefully
        return str(string)
        
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - (len(first_k_chars) + len(last_k_chars)) # More accurate truncated_len
        # Ensure k is not more than half the threshold to avoid overlap or negative truncated_len
        if k * 2 > threshold:
            # Adjust k to be smaller if string is not much larger than k*2
            adjusted_k = len(string) // 3 
            first_k_chars = string[:adjusted_k]
            last_k_chars = string[-adjusted_k:]
            truncated_len = len(string) - (len(first_k_chars) + len(last_k_chars))

        return f"{first_k_chars}\n... [{truncated_len} characters truncated] ...\n{last_k_chars}"
    else:
        return string


def is_valid_python_script(script_content: str) -> bool:
    """Check if a script content is a valid Python script."""
    if not script_content.strip(): # Empty or whitespace-only is not valid for our purpose
        return False
    try:
        compile(script_content, "<string>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception: # Catch other potential errors during compile
        return False

def format_code(code_content: str) -> str:
    """Format Python code using Black. Returns original if formatting fails."""
    if not code_content.strip():
        return ""
    try:
        # Ensure mode is explicitly set if you have specific formatting needs
        return black.format_str(code_content, mode=black.FileMode()).strip()
    except black.parsing.InvalidInput:
        return code_content.strip() # Return original if it's not valid Python for Black
    except Exception:
        return code_content.strip() # Fallback for other black errors

def extract_code(text: str, expect_single_script: bool = True) -> str:
    """
    Extracts Python code blocks from text, prioritizing well-formed blocks.
    Removes <think>...</think> blocks first.
    If expect_single_script is True, it tries to join multiple found blocks.
    """
    if not text:
        return ""

    # 1. Remove all <think>...</think> blocks first to clean up the input
    # Using re.DOTALL to make '.' match newlines as well.
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # If after removing think tags, the text is empty, return empty string.
    if not cleaned_text.strip():
        return ""

    # 2. Regex patterns to try, from more specific to more general
    #    Pattern 1: Standard fenced block with optional 'python' and newlines
    #    Captures content between ```python\n ... \n``` or ```\n ... \n```
    #    Allows for optional whitespace around newlines.
    patterns = [
        r"```(?:python)?\s*\n(.*?)\n\s*```",  # Your primary pattern, good for well-formed blocks
        r"```(?:python)?(.*?)\n\s*```",      # Allows code on the same line as ```python
        r"```\s*\n(.*?)\n\s*```",          # Generic block with newlines
        r"```(.*?)```",                    # Most lenient fenced block (code on same line, no newlines required)
    ]

    extracted_code_blocks = []

    for pattern in patterns:
        matches = re.findall(pattern, cleaned_text, re.DOTALL)
        for match_content in matches:
            # If the pattern has multiple capture groups (like some of your original fallbacks did),
            # re.findall returns tuples. We need to find the actual code.
            # For the patterns above, match_content should be the code itself.
            # Ensure it's a string if a pattern might have multiple groups.
            code_candidate = match_content if isinstance(match_content, str) else "".join(m for m in match_content if m) # Join if it's a tuple of groups
            
            if code_candidate.strip(): # Ensure it's not just whitespace
                extracted_code_blocks.append(code_candidate.strip())
        
        if extracted_code_blocks: # If any pattern found blocks, use them and stop
            break
            
    # 3. Fallback: If no fenced blocks found, try to identify code if the whole text is a script
    #    This is risky and should only be used if you are somewhat confident the remaining text IS code.
    #    A simple heuristic: does it start with common Python keywords or import statements?
    if not extracted_code_blocks and cleaned_text.strip():
        # Check if the cleaned text (without fences) itself looks like Python
        # This is a simpler check than your original broad regex fallback
        lines = cleaned_text.strip().split('\n')
        if lines and (lines[0].strip().startswith(("import ", "from ", "def ", "class ")) or 
                      any(line.strip().startswith("# Thought:") for line in lines[:5])): # Check for your convention
            # We make a judgment call here: if it starts like Python, assume the whole thing is code.
            # This avoids capturing trailing natural language if the LLM just forgot closing backticks.
            # However, it might miss code if it's preceded by significant natural language.
            potential_code = cleaned_text.strip()
            if is_valid_python_script(potential_code): # Validate before adding
                 extracted_code_blocks.append(potential_code)


    # 4. Validate and format the collected blocks
    valid_formatted_blocks = []
    for block in extracted_code_blocks:
        # It's possible a block contains non-code text if regex was too lenient
        # and then further code. Let's try to re-extract from the block if it's complex.
        # This is a bit recursive but can help clean up.
        # For simplicity here, we'll just validate what we got from the main patterns.
        if is_valid_python_script(block):
            valid_formatted_blocks.append(format_code(block))
        # else:
            # Optionally, log or handle blocks that were extracted by regex but are not valid Python.
            # print(f"Warning: Regex extracted a block, but it's not valid Python:\n---\n{block[:200]}...\n---")

    if not valid_formatted_blocks:
        return "" # No valid Python code found

    if expect_single_script:
        # For AIDE, you likely want to join them into one script.
        # Ensure there's a newline between blocks for proper syntax if they were separate.
        return "\n\n".join(valid_formatted_blocks).strip()
    else:
        # If you might want individual valid blocks (e.g., for SFT data where LLM gives multiple examples)
        # you could return the list:
        # return valid_formatted_blocks 
        # For now, sticking to single script output:
        return "\n\n".join(valid_formatted_blocks).strip()

def extract_plan(text):
    """Extract plan from the text."""
    parsed_plan = []
    if "</think>" in text:
        parts = re.split(r"</think>", text, maxsplit=1, flags=re.DOTALL)
        # parts[0] is everything before </think>, parts[1] is everything after
        text = parts[1].strip() if len(parts) > 1 else ""

    # Extract everything after 'PLAN:' including multi-line content
    matches = re.findall(r"## Plan:\s*(.*)", text, re.DOTALL)
    for plan in matches:
        parsed_plan.append(plan.strip())

    # validate the parsed plan and format it
    # Combine and return the plan as a single string
    return "\n\n".join(parsed_plan)


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


# New: extract summary before PLAN
def extract_summary_and_plan(text, task=False):
    """Extract summary from the response before the 'PLAN:' section."""
    # Remove any thinking tags
    if "</think>" in text:
        parts = re.split(r"</think>", text, maxsplit=1, flags=re.DOTALL)
        text = parts[1].strip() if len(parts) > 1 else text
    # Split on PLAN: and return the first part as summary
    if task:
        return text
    if "PLAN" in text:
        parts = text.split("PLAN")
        summary = parts[0].strip()
        plan = parts[1].strip()
        return summary, plan
    return " " , text



def extract_reflection_summary_and_revised_code(text: str) -> tuple[str, str]:
    """
    Extracts the reflection summary and the revised code snippet from an LLM response
    that is expected to have 'Reflection Summary:' and 'Revised Code Snippet:' sections.
    """
    summary = ""
    revised_code = ""

    summary_match = re.search(r"Reflection Summary:(.*?)---", text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()
    else: # Fallback if --- is missing but summary is there
        summary_match_alt = re.search(r"Reflection Summary:(.*?)Revised Code Snippet:", text, re.DOTALL | re.IGNORECASE)
        if summary_match_alt:
            summary = summary_match_alt.group(1).strip()
        else: # Try to grab anything before code as summary
            summary_match_generic = re.search(r"(.*?)Revised Code Snippet:", text, re.DOTALL | re.IGNORECASE)
            if summary_match_generic:
                summary = summary_match_generic.group(1).strip()


    # Extract code using existing extract_code, assuming it handles ```python ... ```
    # We might need to be more specific if the reflection output has other text after the code block.
    # For now, let's assume the Revised Code Snippet is the *last* major block.
    code_block_match = re.search(r"Revised Code Snippet:.*?```python\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        revised_code = code_block_match.group(1).strip()
    else: # Fallback if only one code block is present after "Revised Code Snippet:"
        code_block_match_alt = re.search(r"Revised Code Snippet:(.*)", text, re.DOTALL | re.IGNORECASE)
        if code_block_match_alt:
            potential_code_block = code_block_match_alt.group(1).strip()
            extracted_code_from_potential = extract_code(potential_code_block) # Use your existing robust extractor
            if extracted_code_from_potential:
                revised_code = extracted_code_from_potential
            else: # If no code block found, maybe the whole thing is the snippet (less likely)
                revised_code = f"# FAILED TO EXTRACT REVISED CODE SNIPPET\n# Raw after 'Revised Code Snippet:':\n{potential_code_block}"
        else:
            revised_code = "# FAILED TO FIND 'Revised Code Snippet:' SECTION"
            
    if not summary:
        summary = "SUMMARY_EXTRACTION_FAILED_FROM_REFLECTION"
        
    return summary, revised_code