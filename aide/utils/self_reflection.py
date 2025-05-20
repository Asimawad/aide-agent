# Inside ./utils/self_reflection.py
from typing import Callable
import re
from .config import load_cfg

cfg = load_cfg
QueryFuncType = Callable[..., str]  # Simplified type hint for query
WrapCodeFuncType = Callable[[str], str]  # Simplified type hint for wrap_code
ExtractCodeFuncType = Callable[[str], str]  # Simplified type hint for extract_code

# Example Input Code (with a common error)
EXAMPLE_INPUT_CODE = """
import pandas as pd
import numpy as np

# Simulate loading data
train_df = pd.DataFrame({'feature1': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})
test_df = pd.DataFrame({'feature1': np.random.rand(50)})
sample_submission = pd.DataFrame({'id': range(50), 'target': np.zeros(50)})

# Simulate predictions
predictions = np.random.rand(len(test_df))

# Create submission DataFrame
submission_df = pd.DataFrame({'id': sample_submission['id'], 'target': predictions})

# Save submission - INCORRECT PATH
submission_df.to_csv('my_submission.csv', index=False) # <<< ERROR HERE

print("Submission file created.")
"""

# Expected Output for Stage 1 (Critique) based on EXAMPLE_INPUT_CODE
EXAMPLE_STAGE1_OUTPUT_CRITIQUE = """The primary mistake is saving the submission file with an incorrect name and potentially in the wrong directory relative to the expected './submission/' folder.
1. Line 17: Change the filename from `'my_submission.csv'` to `'submission.csv'`.
2. Line 17: Change the file path to include the target directory, making it `'./submission/submission.csv'`.
"""  # NOTE: Combined steps for clarity, could be separate. Adjust based on model performance.

# Expected Output for Stage 2 (Code Edit) based on EXAMPLE_INPUT_CODE and EXAMPLE_STAGE1_OUTPUT_CRITIQUE
EXAMPLE_STAGE2_OUTPUT_CODE = """# Applying edits based on review.
```python
import pandas as pd
import numpy as np

# Simulate loading data
train_df = pd.DataFrame({'feature1': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})
test_df = pd.DataFrame({'feature1': np.random.rand(50)})
sample_submission = pd.DataFrame({'id': range(50), 'target': np.zeros(50)})

# Simulate predictions
predictions = np.random.rand(len(test_df))

# Create submission DataFrame
submission_df = pd.DataFrame({'id': sample_submission['id'], 'target': predictions})

# Save submission - CORRECTED PATH
submission_df.to_csv('./submission/submission.csv', index=False) # <<< FIXED HERE

print("Submission file created.")
```"""
# --- End Few-Shot Example Definition ---

def perform_two_step_reflection(
    code: str,
    task_desc: str,
    model_name: str,
    temperature: float,
    query_func: QueryFuncType,
    wrap_code_func: WrapCodeFuncType,
    extract_code_func: ExtractCodeFuncType,
) -> tuple[str, str]:
    system_prompt = {"SYSTEM":"    You are a senior data scientist, and precise code critic, trying to check a code written by an junior data scientist to solve a machine learning engineering task - a Kaggle competetion" ,    }
    critique_prompt = { 
        "Question" : f"I am writing a code to solve this task : {task_desc}  , and I wrote this code below ",
        "Code to Review": wrap_code_func(code),  

        "Your Task": "Give a critique, reflect on the code and provide a Code review for possible mistakes and bugs , and also give steps to fix it systimatically",
        "Rules I need you to follow": (
            "RULE 1: **DO NOT WRITE ANY PYTHON CODE IN YOUR RESPONSE.**\n"
            "RULE 2: Do not suggest big changes or new ideas.\n"
            "RULE 3: Only suggest fixes for small mistakes in the code shown.\n"
            "RULE 4: Follow the 'Output Format' EXACTLY."
            "RULE 5: If you see that the code is fine, Reply with one sentence only-> ```No specific errors found requiring changes.```"
        ),

        "Output Format": (
            "If mistakes are found, your response should contain two sections: \n"
            
            "1. ”Review” Section: - Explaining the main mistake(s).\n"
            "2. ”Instructions” Section: write a NUMBERED list of fix instructions.\n"
            "- Each number is ONE simple step Guiding ne from the start to the finish of the code.\n"
            "\n"
            "If no mistakes are found:\n"
            "- Write only this sentence: ```No specific errors found requiring changes.```."
            "\n"
        ),
    }
    plan_raw = query_func(  # Use the passed function
        system_message=system_prompt,
        user_message=critique_prompt,
        model=model_name,  # Use the passed argument
        temperature=temperature,  # Use the passed argument
        convert_system_to_user=False,  # Use the passed argument
    )

    parts = re.split(r"</think>", plan_raw, maxsplit=1, flags=re.DOTALL)
    reflection_plan = parts[1].strip() if len(parts) > 1 else ""
    if reflection_plan.strip() == "No specific errors found requiring changes.":
        return reflection_plan, code
    
    
    system_prompt2 = {"SYSTEM":"You are a Kaggle Grandmaster and a precise coder. you will receive a Kaggle competetion code, a review on the correctness of this code, and Instructions on how to improve its correctness" ,
                        "Task": "your task is to help your team win the competetion by following the code review and implementing the suggested instructions ",
                        "How to reply to the user":" Whenever you answer, always:"
                        " 1. think of a ”Plan:” section in plain text as concise bullet points on how you are going to update the original code. wrap this section between <think></think> tags, it wil be used to hide your thinking from the user, "
                        " 2. Then write a Revised Code: section titled '# Applying edits based on review.' containing the full new improved code, written by following the instructions provided by the user, and your concise plan to fix the code"
                        " 3. If the review on the code```No specific errors found requiring changes.```, Just reply with the same exact copy of the original code wrapped in ``` markdown block",
                        "CRITICAL REQUIREMENT" :"Always make sure that you don't provide a partail solutions, your final code block sholud be complete, starting from imports, untill saving the submission",
                    }
    coder_prompt = {

        "Question" : f"I am trying to improve my code to solve this task : {task_desc} , following the review and instructions I got ",

        "Task": (
            "1. Read the 'Original Code'.\n"
            "2. Read the 'Edit Instructions'. These are text instructions, NOT code.\n"
            "3. Apply ONLY the changes from 'Edit Instructions' to the 'Original Code'.\n"
            "4. Output the result EXACTLY as shown in 'Output Format'."
        ),

        "Rules": (
            "RULE 1: Apply the steps from 'Edit Instructions'.\n"
            "RULE 2: **Do NOT change any other part of the code. if they are ok**\n"
            "RULE 3: Do not reformat or restructure code.\n"
            "RULE 4: If 'Edit Instructions' accidentally contains any code examples, IGNORE THEM. Follow only the numbered text steps.\n"
            "RULE 5: Your entire output MUST follow the 'Output Format'."
        ),
        "Output Format": (
            "your thinking and planning should be hiddin from me, that is achieved by wrapping it between <think></think> tags, I just care about the code"
            "Line 1: Start IMMEDIATELY with a comment '# Applying edits based on review.'\n"
            "Next Line: Start the Python code block immediately.\n"
            "```python\n"
            "[The FULL original code, with ONLY the requested edits applied]\n"
            "```\n"
            "**IMPORTANT: NO TEXT before the first '#' comment. NO TEXT after the final '```'.**"
        ),
        "example": (
            "### EXAMPLE START ###\n"
            "Original Code:\n" 
            f"{wrap_code_func(EXAMPLE_INPUT_CODE)}\n\n"  
            "Edit Instructions:\n"
            f"{EXAMPLE_STAGE1_OUTPUT_CRITIQUE}\n\n"
            "Expected Output:\n"
            f"{EXAMPLE_STAGE2_OUTPUT_CODE}\n"
            "### EXAMPLE END ###"
                 
        ),
        
        
        
        "Original Code": wrap_code_func(code),  # Use the passed function

        "Edit Instructions": reflection_plan,  # Use the cleaned plan from Stage 1
  
    }
    coder_model = cfg.code.critic if cfg.code.model else model_name
    revised_code_response = query_func(  # Use the passed function
        system_message=system_prompt2,
        user_message=coder_prompt,
        model=coder_model ,  # Use the passed argument
        temperature=temperature,  # Use the passed argument
        convert_system_to_user=False,  # Use the passed argument
    )
    revised_code = extract_code_func(revised_code_response)  # Use the passed function

    # Return original code if extraction fails or is empty, otherwise revised
    return reflection_plan, revised_code if revised_code else code