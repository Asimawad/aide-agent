# Inside ./utils/self_reflection.py
import logging  # Import logging
from typing import Callable, Optional, Any  # Added Optional and Any
import re

# Assuming query, wrap_code, extract_code are accessible or passed in.
# Define necessary type hints for the functions being passed
QueryFuncType = Callable[..., Any]  # Changed to Any to handle dict or str
WrapCodeFuncType = Callable[[str], str]
ExtractCodeFuncType = Callable[
    [Any], Optional[str]
]  # Response might not be str, result can be None

from .config import load_cfg  # Assuming this is your config loader

# --- Configuration ---
cfg = load_cfg()
logger = logging.getLogger("aide")  # Use the same logger instance


def perform_two_step_reflection(
    code: str,
    analysis: str,  # This is the textual summary from the feedback LLM
    term_out: str,
    task_desc: str,
    model_name: str,  # This is acfg.code.model for coder, acfg.code.planner_model for planner
    temperature: float,
    convert_system_to_user: bool,
    query_func: QueryFuncType,
    wrap_code_func: WrapCodeFuncType,
    extract_code_func: ExtractCodeFuncType,
    current_step: int = 0,  # Added for logging context
) -> tuple[str, str]:
    """
    Performs a two-step self-reflection on the provided code.

    1. Critiques the code and proposes minimal text-based edits.
    2. Applies only those edits to the original code.
    """
    log_prefix_main = f"SELF_REFLECT_STEP{current_step}"
    logger.info(
        f"{log_prefix_main}: Starting two-step reflection.", extra={"verbose": True}
    )
    # logger.debug(f"{log_prefix_main}_INITIAL_CODE_START\n{code}\n{log_prefix_main}_INITIAL_CODE_END", extra={"verbose": True})
    # logger.debug(f"{log_prefix_main}_INITIAL_ANALYSIS_START\n{analysis}\n{log_prefix_main}_INITIAL_ANALYSIS_END", extra={"verbose": True})
    # logger.debug(f"{log_prefix_main}_INITIAL_TERM_OUT_START\n{term_out}\n{log_prefix_main}_INITIAL_TERM_OUT_END", extra={"verbose": True})

    # --- Stage 1: Critique and Edit Proposal ---
    log_prefix_critique = f"{log_prefix_main}_CRITIQUE"
    system_prompt1 = {
        "SYSTEM": "You are a senior data scientist, trying to check a code written by an junior data scientist to solve a machine learning engineering task - a Kaggle competetion",
        "How to answer the user": " Whenever you answer, always:"
        " 1. Write a “Review” section in plain text—explaining the main mistake(s).\n"  # Matched output format
        " 2. Then write an “Instructions” section: a NUMBERED list of fix instructions.\n",  # Matched output format
    }

    critique_user_prompt = {  # Renamed from critique_prompt to avoid conflict
        "Question": f"I am writing a code to solve this task: {task_desc}, and I wrote this code below.",
        "Code to Review": wrap_code_func(code),
        "Execution Output": term_out,  # Matched key used in Agent
        "Execution Feedback": analysis,  # Matched key used in Agent
        "Your Task": "Provide a Code review for mistakes and bugs, and also give steps to fix it systematically.",
        "Rules I need you to follow": (
            "RULE 1: **DO NOT WRITE ANY PYTHON CODE IN YOUR RESPONSE.**\n"
            "RULE 2: Do not suggest big changes or new ideas.\n"
            "RULE 3: Only suggest fixes for small mistakes in the code shown.\n"
            "RULE 4: Follow the 'Output Format' EXACTLY.\n"
            "RULE 5: If you see that the code is fine, Reply with one sentence only -> ```No specific errors found requiring changes.```"
        ),
        "Output Format": (
            "Your response should contain two sections: \n"
            "1. ”Review” Section: - Explaining the main mistake(s).\n"
            "2. ”Instructions” Section: write a NUMBERED list of fix instructions.\n"
            "- Each number is ONE simple step Guiding me from the start to the finish of the code.\n"
        ),
    }

    logger.info(
        f"{log_prefix_critique}: Sending request for critique and edit proposal. Model: {cfg.agent.code.planner_model}",
        extra={"verbose": True},
    )
    logger.debug(
        f"{log_prefix_critique}_SYSTEM_PROMPT_START\n{system_prompt1}\n{log_prefix_critique}_SYSTEM_PROMPT_END",
        extra={"verbose": True},
    )
    logger.debug(
        f"{log_prefix_critique}_USER_PROMPT_START\n{critique_user_prompt}\n{log_prefix_critique}_USER_PROMPT_END",
        extra={"verbose": True},
    )

    plan_raw = "Error during critique query"  # Default
    try:
        plan_raw = query_func(
            system_message=system_prompt1,
            user_message=critique_user_prompt,
            model=model_name,  # Using planner model for critique
            planner=True,  # Treat critique as a planning-like task
            temperature=temperature,
            convert_system_to_user=convert_system_to_user,
            current_step=current_step,  # Pass current_step for backend logging
        )
        logger.info(
            f"{log_prefix_critique}: Received critique response.",
            extra={"verbose": True},
        )
        logger.debug(
            f"{log_prefix_critique}_RAW_RESPONSE_START\n{plan_raw}\n{log_prefix_critique}_RAW_RESPONSE_END",
            extra={"verbose": True},
        )
    except Exception as e:
        logger.error(
            f"{log_prefix_critique}: Error during critique LLM query: {e}",
            exc_info=True,
            extra={"verbose": True},
        )
        return f"REFLECTION_CRITIQUE_ERROR: {e}", code
    reflection_plan = plan_raw.strip() if isinstance(plan_raw, str) else str(plan_raw)

    logger.debug(
        f"{log_prefix_critique}_EXTRACTED_PLAN_START\n{reflection_plan}\n{log_prefix_critique}_EXTRACTED_PLAN_END",
        extra={"verbose": True},
    )

    if "No specific errors found requiring changes." in reflection_plan:
        logger.info(
            f"{log_prefix_main}: Critique found no specific errors. Returning original code.",
            extra={"verbose": True},
        )
        return reflection_plan, code

    # --- Stage 2: Focused Code Edit ---
    log_prefix_coder = f"{log_prefix_main}_CODER_EDIT"
    system_prompt2 = {
        "SYSTEM": "You are a Kaggle Grandmaster and a precise coder. You will receive a Kaggle competition code, a review on the correctness of this code, and Instructions on how to improve its correctness.",
        "Task": "Your task is to help your team win the competition by following the code review and implementing the suggested instructions.",
        "How to reply to the user": "Whenever you answer, always:"
        " 1. Think of a ”Plan:” section in plain text as concise bullet points on how you are going to update the original code. Wrap this section between <think></think> tags; it will be used to hide your thinking from the user. "  # Note: corrected typo competetion
        " 2. Then write a Revised Code: section titled '# Applying edits based on review.' containing the full new improved code, written by following the instructions provided by the user, and your concise plan to fix the code."
        " 3. If the review on the code is ```No specific errors found requiring changes.```, Just reply with the same exact copy of the original code wrapped in ``` markdown block.",  # This case handled above
        "CRITICAL REQUIREMENT": "Always make sure that you don't provide partial solutions; your final code block should be complete, starting from imports, until saving the submission.",
    }

    coder_user_prompt = {  # Renamed from coder_prompt
        "Question": f"I am trying to improve my code to solve this task: {task_desc}, following the review and instructions I got from my teammates.",
        "Task": (
            "1. Read the 'Original Code', 'Execution Output', and 'Execution Feedback'.\n"
            "2. Read the 'Edit Instructions'. These are text instructions, NOT code.\n"
            "3. Apply ONLY the changes from 'Edit Instructions' to the 'Original Code'.\n"
            "4. Output the result EXACTLY as shown in 'Output Format'."
        ),
        "Original Code": wrap_code_func(code),
        "Execution Output": term_out,
        "Execution Feedback": analysis,
        "Edit Instructions": reflection_plan,
        "Rules": (
            "RULE 1: Apply the steps from 'Edit Instructions'.\n"
            "RULE 2: **Do NOT change any other part of the code if they are ok.**\n"
            "RULE 3: Do not reformat or restructure code.\n"
            "RULE 4: If 'Edit Instructions' accidentally contains any code examples, IGNORE THEM. Follow only the numbered text steps.\n"
            "RULE 5: Your entire output MUST follow the 'Output Format'."
        ),
        "Output Format": (
            "Your thinking and planning should be hidden from me; that is achieved by wrapping it between <think></think> tags. I just care about the code.\n"
            "Line 1: Start IMMEDIATELY with a comment '# Applying edits based on review.'\n"
            "Next Line: Start the Python code block immediately.\n"
            "```python\n"
            "[The FULL original code, with ONLY the requested edits applied]\n"
            "```\n"
            "**IMPORTANT: NO TEXT before the first '#' comment. NO TEXT after the final '```'.**"
        ),
    }

    logger.info(
        f"Sending request for code revision. Model: {model_name}",
        extra={"verbose": True},
    )  # Using model_name passed in, which is acfg.code.model
    logger.debug(
        f"SYSTEM_PROMPT_START\n{system_prompt2}\nSYSTEM_PROMPT_END",
        extra={"verbose": True},
    )
    logger.debug(
        f"USER_PROMPT_START\n{coder_user_prompt}\nUSER_PROMPT_END",
        extra={"verbose": True},
    )

    revised_code_response = "Error during coder query"  # Default
    try:
        revised_code_response = query_func(
            system_message=system_prompt2,
            user_message=coder_user_prompt,
            model=model_name,  # This should be the coder model, e.g., cfg.agent.code.model
            planner=False,
            temperature=temperature,
            convert_system_to_user=convert_system_to_user,
            current_step=current_step,  # Pass current_step
        )
        logger.info(
            f"Received code revision response.",
            extra={"verbose": True},
        )
        logger.debug(
            f"RAW_RESPONSE_START\n{revised_code_response}\nRAW_RESPONSE_END",
            extra={"verbose": True},
        )
    except Exception as e:
        logger.error(
            f"Error during coder LLM query: {e}",
            exc_info=True,
            extra={"verbose": True},
        )
        return reflection_plan, code  # Return original code on error

    revised_code = extract_code_func(revised_code_response)
    if revised_code:
        logger.debug(
            f"EXTRACTED_CODE_START\n{revised_code}\nEXTRACTED_CODE_END",
            extra={"verbose": True},
        )
    else:
        logger.warning(
            f"Code extraction failed from revision response. Raw response was: {revised_code_response}",
            extra={"verbose": True},
        )

    final_revised_code = revised_code if revised_code and revised_code.strip() else code
    if (
        final_revised_code == code and revised_code and revised_code.strip()
    ):  # It extracted something but it was same as original or whitespace
        logger.info(
            f"{log_prefix_main}: Extracted revised code is same as original or empty after stripping. Using original code.",
            extra={"verbose": True},
        )
    elif not revised_code or not revised_code.strip():
        logger.info(
            f"{log_prefix_main}: No valid revised code extracted. Using original code.",
            extra={"verbose": True},
        )
    else:
        logger.info(
            f"{log_prefix_main}: Successfully revised code.", extra={"verbose": True}
        )

    return reflection_plan, final_revised_code
