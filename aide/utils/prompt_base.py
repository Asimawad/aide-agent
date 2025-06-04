# aide/utils/prompt_utils.py
import random
from typing import Any, Dict, List,Optional
import copy # For deepcopying system prompts
import re # For parsing plan steps

from ..backend import FunctionSpec # Assuming this is the correct relative import
from copy import deepcopy

# --- Helper for wrapping code ---
def wrap_code(code_str: str, lang: str = "python") -> str:
    if not isinstance(code_str, str):
        try:
            code_str = str(code_str)
        except:
            return f"```{lang}\n# Invalid code content provided.\n```" if lang else "```\n# Invalid content provided.\n```"
    if not code_str:
        return f"```{lang}\n# No code provided.\n```" if lang else "```\n# No content provided.\n```"
    if lang:
        return f"```{lang}\n{code_str}\n```"
    return f"```\n{code_str}\n```"

# --- review_func_spec (remains the same as your original, make sure it's what you need) ---
review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
            "code_quality": {
                "type": "number",
                "description": "give a score between 0-10 on the quality of the code, where 0 is a terrible code/ non-code at all, and 9-10 is a clean code with a great value for the evaluation metric.",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            # "metric", # Metric can be null if buggy, so not strictly required in the object itself
            "lower_is_better",
            "code_quality",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)



# --- Data for competitions (from your codebase) ---
COMPETITION_METADATA = {
    "aerial-cactus-identification": {"Task Type": "Image Classification", "Size (GB)": 0.0254},
    "denoising-dirty-documents": {"Task Type": "Image To Image", "Size (GB)": 0.06},
    "detecting-insults-in-social-commentary": {"Task Type": "Text Classification", "Size (GB)": 0.002},
    "dog-breed-identification": {"Task Type": "Image Classification", "Size (GB)": 0.75},
    "dogs-vs-cats-redux-kernels-edition": {"Task Type": "Image Classification", "Size (GB)": 0.85},
    "jigsaw-toxic-comment-classification-challenge": {"Task Type": "Text Classification", "Size (GB)": 0.06},
    "leaf-classification": {"Task Type": "Image Classification", "Size (GB)": 0.036},
    "mlsp-2013-birds": {"Task Type": "Audio Classification", "Size (GB)": 0.5851},
    "nomad2018-predict-transparent-conductors": {"Task Type": "Tabular", "Size (GB)": 0.00624},
    "plant-pathology-2020-fgvc7": {"Task Type": "Image Classification", "Size (GB)": 0.8},
    "random-acts-of-pizza": {"Task Type": "Text Classification", "Size (GB)": 0.003},
    "spooky-author-identification": {"Task Type": "Text Classification", "Size (GB)": 0.0019},
    "tabular-playground-series-dec-2021": {"Task Type": "Tabular", "Size (GB)": 0.7},
    "tabular-playground-series-may-2022": {"Task Type": "Tabular", "Size (GB)": 0.57},
    "text-normalization-challenge-english-language": {"Task Type": "Seq->Seq", "Size (GB)": 0.01},
    "text-normalization-challenge-russian-language": {"Task Type": "Seq->Seq", "Size (GB)": 0.01},
}
PACKAGE_CATEGORIES = {
    "common": ["numpy", "pandas", "scikit-learn"],
    "tabular": ["xgboost", "lightgbm", "scikit-learn"],
    "image": ["torch", "torchvision", "Pillow", "opencv-python"],
    "text": ["torch", "transformers", "nltk"],
    "audio": ["torch", "librosa"],
    "graph": ["networkx"],
    "optimization": ["optuna"]
}

def get_competition_environment_text(competition_name: str, is_small_model_target: bool = True) -> str:
    text_parts = [f"Competition: '{competition_name}'."]
    if competition_name in COMPETITION_METADATA:
        current_comp_data = COMPETITION_METADATA[competition_name]
        task_type = current_comp_data["Task Type"]
        text_parts.append(f"Task Type: {task_type}.")
        suggested_pkgs_set = set(PACKAGE_CATEGORIES["common"])
        task_specific_guidance = ""
        if "image" in task_type.lower():
            suggested_pkgs_set.update(PACKAGE_CATEGORIES["image"])
            task_specific_guidance = "Key image libs: `torchvision`, `Pillow`, `opencv-python`."
        elif "tabular" in task_type.lower():
            suggested_pkgs_set.update(PACKAGE_CATEGORIES["tabular"])
            task_specific_guidance = "Key tabular libs: `sklearn` (RandomForest/LogisticRegression), `xgboost`."
        elif "text" in task_type.lower() or "seq->seq" in task_type.lower():
            suggested_pkgs_set.update(PACKAGE_CATEGORIES["text"])
            task_specific_guidance = "Key text libs: `transformers`, `nltk`."
        # Add other categories as needed
        else:
            task_specific_guidance = "Use general ML libraries."
        pkgs_list = sorted(list(suggested_pkgs_set))
        pkg_str = ", ".join([f"`{p}`" for p in pkgs_list])
        if is_small_model_target:
            text_parts.append(f"Core packages: {pkg_str} {task_specific_guidance} PyTorch preferred for NNs.")
        else: # More verbose for larger models
             text_parts.append(f"Data Size (GB): {current_comp_data.get('Size (GB)', 'N/A')}.")
             text_parts.append(f"Installed Packages: Any relevant ML packages like {pkg_str}. {task_specific_guidance} All installed. PyTorch preferred for NNs.")
    else:
        pkgs = ["numpy", "pandas", "scikit-learn", "torch"]
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])
        text_parts.append(f"Metadata not found. Core packages: {pkg_str}. PyTorch preferred for NNs.")
    return " ".join(text_parts)

# --- Static Prompt Components (Simplified for 8B, Planner remains detailed) ---


AGENT_IMPLEMENTATION_GUIDELINE_LIST_8B: List[str] = [
    "1. Deliver a complete, single-file Python script that solves the competition and saves `submission.csv`.",
    "2. The PLAN must be specific and actionable for *this* task.",
    "3. Implement the solution proposed in your PLAN precisely.",
    "4. Calculate and `print()` the validation metric (e.g., `print(f'Validation Metric: {metric_value}')`).",
    "5. CRITICAL: Save test predictions to `./submission/submission.csv` in the correct format.",
    "6. Ensure the script is error-free and clean.",
]

AGENT_RESPONSE_FORMAT_TEXT_8B: str = (
    "Respond with 'PLAN:' followed by your step-by-step plan (numbered list). Then '---' separator. Then 'CODE:' followed by a single ```python ... ``` block. "
    "NO text before 'PLAN:' or after the final '```'."
    "Example PLAN step: '1. Load `train.csv` with pandas.'\n"
    "Example CODE: ```python\n# Thought: Plan step 1: Load data.\nimport pandas as pd\ntrain_df = pd.read_csv('./input/train.csv')\n```"
)

# --- System Prompts (Use _8B variants for Agent instructions) ---


AGENT_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": "You are a Kaggle Grandmaster. You plan, implement, debug, and improve ML code.",
    "user_instructions": { # These are instructions *to the LLM* on how it should structure its own output/behavior
        "Task": "Solve Kaggle competitions by generating a PLAN and Python CODE.",
        "Output Structure": AGENT_RESPONSE_FORMAT_TEXT_8B, # Use 8B format here
        "PLAN Details": "7-10 detailed, actionable bullet points. Explain *how* each step is done.",
        "CODE Details": "Implement the PLAN. Before each code block, add a '# Thought:' comment explaining its purpose and relation to the PLAN step.",
        "Key Goal": "Produce correct, working code that generates `submission.csv`."
    },
}

# Draft/Improve/Debug System Prompts for 8B (Simplified instructions TO the LLM)
AGENT_draft_SYSTEM_PROMPT_DICT_8B: Dict[str, Any] = {
    "SYSTEM": "You are a Kaggle Grandmaster. Create a PLAN and Python CODE for an ML competition. Focus on a simple, correct, bug-free first draft. Follow output format strictly.",
    "user_instructions": {
        "Task Context": "You get a Kaggle competition description. Generate a PLAN and CODE.",
        "Output Structure": AGENT_RESPONSE_FORMAT_TEXT_8B,
        "PLAN Requirements": "7-10 detailed steps (WHAT, HOW, WHY). NO EDA. Simple. Data in `./input/`. If 'Memory' given, try a NEW approach if past failed.",
        "CODE Requirements": "Implement PLAN. MANDATORY: '# Thought:' comments linking to PLAN. Single script.",
        "Final Instructions": "Strictly follow PLAN and '# Thought:'. Aim for working, bug-free draft. If 'Memory' exists, AVOID past mistakes and try NEW solution."
    }
}
AGENT_improve_SYSTEM_PROMPT_DICT_8B: Dict[str, Any] = {
    "SYSTEM": "Kaggle Grandmaster: Given working Python code, propose ONE justified improvement. Provide PLAN and full MODIFIED CODE.",
    "user_instructions": {
        "Input": "Task Desc, Previous CODE, Memory (opt).",
        "Output Structure": AGENT_RESPONSE_FORMAT_TEXT_8B,
        "PLAN Requirements": "Rationale (2-3 sentences). Detailed Plan (3-5 steps for ONE change: WHAT, WHERE, HOW, WHY).",
        "CODE Requirements": "ENTIRE previous script, modified ONLY for the single improvement. Use '# Improvement Thought:' comments.",
        "Focus": "Single, atomic improvement. Justify. Minimal code changes. No EDA."
    }
}
AGENT_debug_SYSTEM_PROMPT_DICT_8B: Dict[str, Any] = {
    "SYSTEM": "Kaggle Grandmaster: Debug Python ML code. Analyze buggy code, traceback, summary. PLAN fix for traceback error. Provide full CORRECTED CODE.",
    "user_instructions": {
        "Input": "Task Desc, Buggy Code, Traceback, Bug Summary.",
        "Output Structure": AGENT_RESPONSE_FORMAT_TEXT_8B,
        "PLAN Requirements": "Start 'Bug Analysis:' (traceback error, line, root cause, confirm/refute summary). Then 'Fix Plan:' (minimal steps for THAT error).",
        "CODE Requirements": "ENTIRE script with ONLY the fix. Use '# Bugfix Thought:' comments.",
        "Focus": "Fix ONLY traceback error. Minimal changes."
    }
}

# --- System Prompt Getters ---
def get_agent_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_SYSTEM_PROMPT_DICT)
def get_agent_draft_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_draft_SYSTEM_PROMPT_DICT_8B)
def get_agent_improve_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_improve_SYSTEM_PROMPT_DICT_8B)
def get_agent_debug_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_debug_SYSTEM_PROMPT_DICT_8B)

# --- User Message Assemblers for Agent (Using _8B elements) ---
def get_agent_draft_user_prompt(
    task_desc: str, journal_summary: str, competition_name: str, obfuscate: bool,
    acfg_data_preview: bool, data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = f"Solve '{competition_name}' Kaggle competition. Develop PLAN and Python CODE."
    if obfuscate: introduction = "Develop PLAN and Python SCRIPT for ML problem."
    return {
        "Introduction": introduction,
        "Overall Task Description": task_desc,
        "Data Overview": data_preview_content if acfg_data_preview and data_preview_content else "Rely on plan/task desc for data.",
        "Memory": journal_summary if journal_summary and journal_summary.strip() and "No previous attempts" not in journal_summary else "No prior attempts.",
        "Instructions": {
            "Key Guidelines": AGENT_IMPLEMENTATION_GUIDELINE_LIST_8B,
            "Environment": get_competition_environment_text(competition_name, is_small_model_target=True),
            "Output Format": AGENT_RESPONSE_FORMAT_TEXT_8B,
            "Drafting Focus": "Simple, correct, bug-free code. PLAN: 7-10 detailed steps (WHAT, HOW, WHY). NO EDA. Data in `./input/`. If 'Memory', try NEW approach if past failed."
        },
    }

def get_agent_improve_user_prompt(
    task_desc: str, journal_summary: str, competition_name: str, parent_node_code: str,
) -> Dict[str, Any]:
    introduction = "Improve provided working Python solution. Propose ONE change."
    return {
        "Introduction": introduction,
        "Task Description": task_desc,
        "Previous Working Code": wrap_code(parent_node_code),
        "Memory": journal_summary if journal_summary and journal_summary.strip() and "No previous attempts" not in journal_summary else "No prior improvement ideas.",
        "Instructions": {
            "Output Format": AGENT_RESPONSE_FORMAT_TEXT_8B,
            "Improvement Focus": "PLAN: Rationale (2-3 sentences) + Detailed Plan (3-5 steps for ONE change: WHAT, WHERE, HOW, WHY). CODE: Entire script, modified ONLY for the single improvement. Use '# Improvement Thought:' comments.",
            "Environment": get_competition_environment_text(competition_name, is_small_model_target=True),
        },
    }

def get_agent_debug_user_prompt(
    task_desc: str, competition_name: str, parent_node_code: str,
    parent_node_term_out: str, parent_node_feedback: str,
    acfg_data_preview: bool, data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = "Debug failing Python script. Analyze traceback, plan fix, provide corrected code."
    prompt = {
        "Introduction": introduction,
        "Task Description": task_desc,
        "Buggy Code": wrap_code(parent_node_code),
        "Execution Traceback": wrap_code(parent_node_term_out, lang=""),
        "Initial Bug Summary": parent_node_feedback if parent_node_feedback else "Analyze traceback directly.",
        "Instructions": {
            "Output Format": AGENT_RESPONSE_FORMAT_TEXT_8B,
            "Debugging Focus": "PLAN: 'Bug Analysis:' (traceback error, line, root cause) then 'Fix Plan:' (minimal steps for THAT error). CODE: Entire script with ONLY the fix. Use '# Bugfix Thought:' comments.",
            "Environment": get_competition_environment_text(competition_name, is_small_model_target=True),
        },
    }
    if acfg_data_preview and data_preview_content:
        prompt["Data Overview"] = data_preview_content
    return prompt

# --- Planner Agent Prompts ---
# PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT: This is where we ask the PLANNER for a DETAILED plan.
# We will use your original more detailed prompt here, or a similarly detailed one.
# For the sake of this example, I'm putting back the detailed planner prompt from your original codebase.
# Ensure this matches the one you want the Planner (even if 8B) to *attempt* to follow.
PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT: Dict[str, Any] = { # This should be a DETAILED prompt for the PLANNER
    "SYSTEM": (
        "You are an expert Kaggle Grandmaster and a meticulous Technical Lead. Your primary responsibility is to "
        "create an exceptionally detailed, actionable, and high-quality strategic Master Plan for solving a given machine learning competition. "
        "This Master Plan will be the sole blueprint for a separate Coder agent. "
        "Your output MUST be a 'Task Summary' followed by a 'Plan'. You do NOT write any code."
    ),
    "user_instructions": {
        "Input Understanding": "You will receive: a 'Full Kaggle Competition Description' (including evaluation, data files, and submission format), a 'Data Overview' (file structure and CSV column info), and potentially a 'Memory of Previous Attempts'.",
        "Output Requirements (Strict Adherence Mandatory)": (
            "Your entire response MUST strictly follow this two-part structure, using the specified markdown headers:\n\n"
            "## Task Summary:\n"
            "   - Provide a concise summary (around 5-7 sentences) of the overall competition task. Clearly state: \n"
            "     a) The main objective (e.g., 'predict house sale prices').\n"
            "     b) The primary evaluation metric (e.g., 'Root Mean Squared Logarithmic Error (RMSLE)').\n"
            "     c) Key characteristics of the input data (e.g., 'tabular data with numerical and categorical features in train.csv and test.csv').\n"
            "     d) The expected submission file format (e.g., 'CSV with Id and SalePrice columns').\n"
            "     This summary orients the Coder agent about the core problem.\n\n"
            "## Plan:\n"
            "   - Construct a list of 7-10 sequential, numbered bullet points outlining the step-by-step methodology to create a *simple, correct, and bug-free first draft solution* that completes the full task from data loading to submission file generation.\n"
            "   - **Crucial Detail for Each Plan Step (WHAT, HOW, WHY - The '3 Ws'):** For *every* bullet point, you MUST explicitly detail:\n"
            "       a. **WHAT** is the specific action or goal of this step (e.g., 'Load training and test data', 'Handle missing numerical values', 'Encode categorical features').\n"
            "       b. **HOW** this action will be achieved: Be extremely specific. Mention key Python libraries (e.g., `pandas`, `numpy`, `sklearn.impute`, `sklearn.preprocessing`, `xgboost`), specific functions/classes (e.g., `pd.read_csv()`, `SimpleImputer()`, `StandardScaler()`, `OneHotEncoder()`, `XGBRegressor()`), critical parameters if non-default or for clarity (e.g., `SimpleImputer(strategy='median')`, `train_test_split(test_size=0.2, random_state=42)`), and names of key variables to be created or used (e.g., 'Load data into `train_df` and `test_df`', 'Imputed features stored in `X_train_numeric_imputed`', 'Instantiate model as `xgb_model`'). Your 'HOW' should be precise enough that a Coder agent can translate it directly into code with minimal ambiguity.\n"
            "       c. **WHY** this step is necessary or its purpose in the overall solution (a brief rationale, e.g., 'to prepare data for modeling', 'to prevent data leakage from test set into training transformations', 'to convert text categories into a machine-understandable format').\n"
            "   - **File Name Specificity:** Your plan must be specific about actual file names to be loaded (e.g., `train.csv`, `test.csv`, `sample_submission.csv`), deriving these from the 'Full Kaggle Competition Description' and 'Data Overview' provided in the user message.\n"
            "   - **Constraints for This Draft Plan:**\n"
            "       - **No EDA:** Do NOT include steps for Exploratory Data Analysis.\n"
            "       - **Simplicity:** Prioritize a straightforward, robust approach. Avoid complex ensembling, extensive hyperparameter optimization, or overly niche techniques for this initial draft. A simple model like Linear Regression, RandomForest, or a basic XGBoost/LightGBM is preferred.\n"
            "       - **Data Access:** Assume all necessary data files are located in a standard `./input/` directory (relative to the script's execution) and require no unzipping, unless the problem description explicitly states otherwise.\n"
            "       - **Memory Consideration:** If a 'Memory of Previous Attempts' is provided, analyze it. Your new plan should aim to be a *distinctly different, viable approach* if previous ones failed, or build upon successful elements while avoiding repeated mistakes. Specifically state if your approach differs from memory and why.\n\n"
            '   *Example Plan Step (House Price - details as per your original for planner):*\n'
            '   " - **1. Initial Setup & Data Loading**: WHAT: Import core libraries and load `train.csv` and `test.csv`. HOW: Use `pd.read_csv(\'./input/train.csv\')` into `train_df`. Store `test_ids = test_df[\'Id\']`. WHY: Access data, preserve IDs."\n'
            # (Include more example steps here if needed to guide the planner, but keep them consistent with asking for detail)
        ),
        "Critical Reminder": "Your role is exclusively planning and summarizing. The Coder agent relies ENTIRELY on the clarity, explicitness (especially the 'HOW' for each step, including function/library/variable specifics), and logical correctness of your 'Task Summary' and 'Plan'. DO NOT generate any Python code."
    }
}

# CODER system prompt (simplified for 8B)
PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT_8B: Dict[str, Any] = {
    "SYSTEM": "Expert Kaggle Coder: Implement the provided 'Plan' from Tech Lead. Output ONLY a single Python script. Use '# Thought:' comments.",
    "user_instructions": {
        "Input": "Task Summary, Plan to Implement, Memory (opt), Environment.",
        "Task": "Translate ENTIRE 'Plan to Implement' into Python.",
        "Mandatory Commenting": "Before code blocks, add '# Thought:' comment: strategy, purpose, PLAN step(s) addressed.",
        "Output Requirement": "ONLY ```python ... ```. NO extra text.",
        "Draft Guidelines": "Simple plan implementation. Import necessary libs. Load data from `./input/`. Print validation metric. CRITICAL: Save predictions to `./submission/submission.csv`. Error-free."
    }
}

def get_planner_agent_plan_system_prompt() -> Dict[str, Any]:
    # This prompt asks the PLANNER to be detailed.
    return copy.deepcopy(PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT)

def get_planner_agent_code_system_prompt() -> Dict[str, Any]:
    # This prompt is for the CODER and is simplified.
    return copy.deepcopy(PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT_8B)

# User prompts for PlannerAgent (similar structure, content passed to LLM is managed by Agent)
# get_planner_agent_draft_plan_user_prompt: content passed from Agent, system prompt guides planner output.
# get_planner_agent_draft_code_user_prompt: receives planner's (detailed) output, its own system prompt is simplified.
# These can remain largely the same as in the previous 8B draft I provided,
# as their main role is to structure the information flow. The system prompts are key.
def get_planner_agent_draft_plan_user_prompt(
    task_desc: str, journal_summary: str, competition_name: str,
    acfg_data_preview: bool, data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = f"For ML competition '{competition_name}', provide 'Task Summary' and detailed 'Plan'."
    return {
        "Introduction": introduction,
        "Overall Task Description": task_desc,
        "Data Overview": data_preview_content if acfg_data_preview and data_preview_content else "Rely on task desc for data.",
        "Environment": get_competition_environment_text(competition_name, is_small_model_target=True), # Targetting 8B
        "Memory": journal_summary if journal_summary and journal_summary.strip() and "No previous attempts" not in journal_summary else "No prior attempts. Propose fresh, simple solution.",
        "Instructions": "Follow system prompt for 'Task Summary' and 'Plan' (detailed WHAT, HOW, WHY). Aim for NEW, simple, bug-free draft."
    }

def get_planner_agent_draft_code_user_prompt(
    task_summary_from_planner: str, plan_from_planner: str, journal_summary: str,
    competition_name: str, acfg_data_preview: bool, data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = f"Implement PLAN for '{competition_name}'. Use '# Thought:' comments."
    # For the coder, the plan_from_planner is the full detailed plan.
    # The Coder's SYSTEM prompt instructs it on how to behave.
    return {
        "Introduction": introduction,
        "Context from Technical Lead": {
            "Task Summary": task_summary_from_planner or "No task summary.",
            "Plan to Implement": plan_from_planner or "CRITICAL: No plan." # This is the FULL detailed plan
        },
        "Data Overview": data_preview_content if acfg_data_preview and data_preview_content else "Rely on plan.",
        "Environment": get_competition_environment_text(competition_name, is_small_model_target=True),
        "Memory": journal_summary if journal_summary and journal_summary.strip() and "No previous attempts" not in journal_summary else "No prior bug history.",
        "Your Task": "Write Python script from 'Plan to Implement'. Output only code block."
    }

# Improve and Debug prompts for PlannerAgent (user prompts are similar, system prompts guide planner/coder)
def get_planner_agent_improve_plan_user_prompt(*args, **kwargs) -> Dict[str, Any]: # Pass through for now
    return get_agent_improve_user_prompt(*args, **kwargs) # Can be adapted if planner needs different info for "improve plan"
def get_planner_agent_improve_code_user_prompt(*args, **kwargs) -> Dict[str, Any]:
    return get_agent_improve_user_prompt(*args, **kwargs) # Coder gets same info structure for "improve code"
def get_planner_agent_debug_plan_user_prompt(*args, **kwargs) -> Dict[str, Any]:
    return get_agent_debug_user_prompt(*args, **kwargs)
def get_planner_agent_debug_code_user_prompt(*args, **kwargs) -> Dict[str, Any]:
    return get_agent_debug_user_prompt(*args, **kwargs)


# --- CodeChainAgent Prompts (Simplified for 8B) ---
segments_order = [
    "Setup & Imports", "Data Loading", "Data Preprocessing",
    "Modeling", "Training & Validation", "Prediction & Submission"
]

_BASE_CODER_CHAIN_SYSTEM_MESSAGE_8B = (
    "Expert Python Coder: Implement specific ML segment. Input: Task Summary, Relevant Master Plan Steps, Prior Code (truncated), Current Segment Instructions. "
    "Output: ONLY Python code for current segment. Use '# Thought:' comments linking to Master Plan. Integrate. Focus on correctness for THIS segment."
)

def _create_coder_chain_segment_system_prompt_8b(objective: str) -> Dict[str, Any]:
    return {
        "SYSTEM": _BASE_CODER_CHAIN_SYSTEM_MESSAGE_8B,
        "user_instructions": {
            "Current Segment Objective": objective,
            "Reminders": "Output ONLY ```python ... ```. Use '# Thought:'. Integrate. New imports for THIS segment go in YOUR block."
        }
    }

# Create _8B variants for each segment system prompt
CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SETUP_8B = _create_coder_chain_segment_system_prompt_8b("Initial setup: imports, `set_seed`, PyTorch `DEVICE`.")
CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_DATA_LOADING_8B = _create_coder_chain_segment_system_prompt_8b("Define path constants & load primary data to DataFrames.")
CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_PREPROCESSING_8B = _create_coder_chain_segment_system_prompt_8b("Preprocessing: impute, encode, scale, features, Dataset/DataLoader, train/val split.")
CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_MODELING_8B = _create_coder_chain_segment_system_prompt_8b("Define & instantiate ML model. Move to `DEVICE`.")
CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_TRAINING_8B = _create_coder_chain_segment_system_prompt_8b("Training loop: loss, optimizer, epochs, validation, print metrics.")
CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SUBMISSION_8B = _create_coder_chain_segment_system_prompt_8b("Test prediction & save `./submission/submission.csv`.")

CHAINED_CODER_SYSTEM_PROMPT_GETTERS = {
    "Setup & Imports": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SETUP_8B),
    "Data Loading": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_DATA_LOADING_8B),
    "Data Preprocessing": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_PREPROCESSING_8B),
    "Modeling": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_MODELING_8B),
    "Training & Validation": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_TRAINING_8B),
    "Prediction & Submission": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SUBMISSION_8B),
}

# --- Helper Functions for Coder Chain User Prompts (Token Reduction) ---
def _extract_relevant_plan_steps(master_plan_text: str, segment_name: str, segments_order: List[str], all_plan_lines: List[str]) -> str:
    """
    Extracts plan steps relevant to the current segment from a list of all plan lines.
    Improved heuristic: looks for segment name keywords in plan steps, or falls back to rough slicing.
    """
    if not all_plan_lines:
        return "No detailed plan steps provided to extract from."

    segment_keywords = [kw.lower() for kw in segment_name.replace("&", " ").replace("_", " ").split() if len(kw) > 2]
    relevant_indices = []
    for i, line in enumerate(all_plan_lines):
        if any(keyword in line.lower() for keyword in segment_keywords):
            relevant_indices.append(i)
    
    extracted_steps = []
    if relevant_indices:
        # Try to get a contiguous block around the first keyword match, or just the matched lines
        # This is still heuristic. A better way is if planner structures plan with segment subheadings.
        start_rel_idx = relevant_indices
        # Try to get 1-2 steps if possible for context
        end_rel_idx = min(start_rel_idx + 2, len(all_plan_lines)) 
        extracted_steps = all_plan_lines[start_rel_idx:end_rel_idx]
    
    if not extracted_steps: # Fallback to rough slicing if keyword search yields nothing
        num_total_steps = len(all_plan_lines)
        try:
            current_segment_idx = segments_order.index(segment_name)
        except ValueError:
            current_segment_idx = 0 # Default to first segment if name not found

        # Distribute steps somewhat evenly, giving more to preprocessing/training
        # This is a very rough heuristic and needs careful thought based on typical plan structures.
        # Example:
        if segment_name == "Setup & Imports": start_idx, count = 0, 1
        elif segment_name == "Data Loading": start_idx, count = 1, 1
        elif segment_name == "Data Preprocessing": start_idx, count = 2, max(1, num_total_steps // 4)
        elif segment_name == "Modeling": start_idx, count = 2 + max(1, num_total_steps // 4), 1
        elif segment_name == "Training & Validation": start_idx = 3 + max(1, num_total_steps // 4) ; count = max(1, num_total_steps - start_idx -1) # most of remaining
        elif segment_name == "Prediction & Submission": start_idx = num_total_steps -1 if num_total_steps > 0 else 0; count = 1
        else: start_idx, count = 0,1

        start_idx = min(start_idx, num_total_steps - 1 if num_total_steps > 0 else 0)
        end_idx_exclusive = min(start_idx + count, num_total_steps)
        extracted_steps = all_plan_lines[start_idx:end_idx_exclusive]

    if not extracted_steps and all_plan_lines: # If still no steps, take the first one
        extracted_steps = [all_plan_lines]
    elif not all_plan_lines:
         return f"No plan steps available for segment: {segment_name}."


    return f"Relevant Master Plan Steps for Segment '{segment_name}':\n" + "\n".join(extracted_steps)

def _truncate_code_so_far(code_so_far: str, max_lines: int = 30, max_chars_per_line: int = 80) -> str:
    if not code_so_far or not code_so_far.strip():
        return "# No code generated in prior segments."
    lines = code_so_far.splitlines()
    important_lines = []
    other_lines = []
    for line in lines:
        stripped_line = line.strip()
        # Prioritize imports, function/class defs, and key variable assignments for context
        if stripped_line.startswith(("import ", "from ", "def ", "class ")) or \
           re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*torch\.device", stripped_line) or \
           re.match(r"^[A-Z_][A-Z0-9_]*\s*=\s*['\"./]", stripped_line): # Path constants
            important_lines.append(line[:max_chars_per_line] + '...' if len(line) > max_chars_per_line else line)
        else:
            other_lines.append(line[:max_chars_per_line] + '...' if len(line) > max_chars_per_line else line)
            
    combined_lines = important_lines
    remaining_slots = max_lines - len(combined_lines)
    
    if remaining_slots > 0 and other_lines: # Add recent non-definition lines from the end
        combined_lines.extend(other_lines[-remaining_slots:])
    elif remaining_slots < 0: # Too many important lines, truncate them
        combined_lines = combined_lines[:max_lines]


    if len(combined_lines) < len(lines):
        return "\n".join(combined_lines) + f"\n# ... (previous code truncated, {len(lines) - len(combined_lines)} more lines not shown) ..."
    return "\n".join(combined_lines)

def _get_master_plan_lines(master_plan_text: str) -> List[str]:
    """Helper to split master plan into a list of non-empty, stripped lines, attempting to identify plan steps."""
    if not master_plan_text: return []
    # Regex to capture lines starting with typical list markers (number. , - , *)
    # It also captures the content after the marker.
    potential_steps = re.findall(r"^\s*(?:(\d+\.)|([-\*]))\s+(.*)", master_plan_text, re.MULTILINE)
    # Flatten the tuples from findall (marker, content) and keep only content if marker found.
    # Or, if no markers, split by newline and filter empty.
    if potential_steps:
        #  potential_steps would be like [('1.', '', 'Step one content'), ('', '-', 'Step two content')]
        # We want the last non-empty group which is the content.
        return [match[-1].strip() for match in potential_steps if match[-1].strip()]
    else: # Fallback if no clear list markers
        return [line.strip() for line in master_plan_text.splitlines() if line.strip()]


def _get_base_coder_chain_user_prompt_args_8b(
    task_summary: str, master_plan_text: str, current_code_so_far: str,
    competition_name: str, data_preview_content: Optional[str], current_segment_name: str
) -> Dict[str, Any]:
    all_plan_lines = _get_master_plan_lines(master_plan_text)
    relevant_plan = _extract_relevant_plan_steps(master_plan_text, current_segment_name, segments_order, all_plan_lines)
    truncated_code_so_far = _truncate_code_so_far(current_code_so_far)

    return {
        "Context from Technical Lead": {
            "Task Summary": task_summary or "No task summary.",
            "Relevant Master Plan Steps for Current Segment": relevant_plan
        },
        "Python Code Generated So Far (Truncated)": wrap_code(truncated_code_so_far),
        "Environment": get_competition_environment_text(competition_name, is_small_model_target=True),
        "Data Overview": data_preview_content if data_preview_content else "Refer to Plan/Summary for data."
    }

# --- Coder Chain User Prompts (using _8b base) ---

def get_coder_chain_user_prompt_segment_setup(*args, **kwargs) -> Dict[str, Any]:
    segment_name = "Setup & Imports"
    prompt_args = _get_base_coder_chain_user_prompt_args_8b(*args, current_segment_name=segment_name, **kwargs)
    prompt_args[f"Your Current Segment: {segment_name}"] = (
        "Write Python for initial setup: library imports (pandas, numpy, sklearn, torch etc.), "
        "`set_seed(42)` function (for random, numpy, torch) & call it, PyTorch `DEVICE` definition. ONLY code block."
    )
    return prompt_args

def get_coder_chain_user_prompt_segment_data_loading(*args, **kwargs) -> Dict[str, Any]:
    segment_name = "Data Loading"
    prompt_args = _get_base_coder_chain_user_prompt_args_8b(*args, current_segment_name=segment_name, **kwargs)
    prompt_args[f"Your Current Segment: {segment_name}"] = (
        "Write Python *only* for: 1. Defining path constants (e.g., `INPUT_DIR`, `TRAIN_CSV_PATH`) for files in `./input/` (use `os.path.join` or `pathlib.Path`). "
        "2. Loading primary data (e.g., `train.csv`, `test.csv`) into pandas DataFrames (e.g., `train_df`, `test_df`). Use 'Relevant Master Plan Steps'. ONLY code."
    )
    return prompt_args

def get_coder_chain_user_prompt_segment_preprocessing(*args, **kwargs) -> Dict[str, Any]:
    segment_name = "Data Preprocessing"
    prompt_args = _get_base_coder_chain_user_prompt_args_8b(*args, current_segment_name=segment_name, **kwargs)
    prompt_args[f"Your Current Segment: {segment_name}"] = (
        "Write Python *only* for data preprocessing per 'Relevant Master Plan Steps': imputation, encoding, scaling, feature engineering, "
        "PyTorch `Dataset` class (if plan implies, with `__init__`, `__len__`, `__getitem__`), train/val split (`train_test_split`), `DataLoader`s. "
        "Define `train_loader`, `val_loader`, etc. ONLY code."
    )
    return prompt_args

def get_coder_chain_user_prompt_segment_modeling(*args, **kwargs) -> Dict[str, Any]:
    segment_name = "Modeling"
    prompt_args = _get_base_coder_chain_user_prompt_args_8b(*args, current_segment_name=segment_name, **kwargs)
    prompt_args[f"Your Current Segment: {segment_name}"] = (
        "Write Python *only* for defining & instantiating ML model per 'Relevant Master Plan Steps'. "
        "Define custom `nn.Module` if needed, instantiate (e.g., `timm.create_model`, `RandomForestClassifier`), move to `DEVICE`. Assign to `model`. ONLY code."
    )
    return prompt_args

def get_coder_chain_user_prompt_segment_training(*args, **kwargs) -> Dict[str, Any]:
    segment_name = "Training & Validation"
    prompt_args = _get_base_coder_chain_user_prompt_args_8b(*args, current_segment_name=segment_name, **kwargs)
    prompt_args[f"Your Current Segment: {segment_name}"] = (
        "Write Python *only* for model training loop per 'Relevant Master Plan Steps': define loss & optimizer, "
        "iterate epochs, train batches (forward, loss, backward, step), validate (calc loss & metric), `print()` val metrics. Update `model`. ONLY code."
    )
    return prompt_args

def get_coder_chain_user_prompt_segment_submission(*args, **kwargs) -> Dict[str, Any]:
    segment_name = "Prediction & Submission"
    prompt_args = _get_base_coder_chain_user_prompt_args_8b(*args, current_segment_name=segment_name, **kwargs)
    prompt_args[f"Your Current Segment: {segment_name}"] = (
        "Write Python *only* for test prediction & submission per 'Relevant Master Plan Steps'. Load best model (if saved), `model.eval()`, predict on test, "
        "format into DataFrame, CRITICALLY save to `./submission/submission.csv` (`index=False`). ONLY code."
    )
    return prompt_args

CHAINED_CODER_USER_PROMPT_CONSTRUCTORS = {
    "Setup & Imports": get_coder_chain_user_prompt_segment_setup,
    "Data Loading": get_coder_chain_user_prompt_segment_data_loading,
    "Data Preprocessing": get_coder_chain_user_prompt_segment_preprocessing,
    "Modeling": get_coder_chain_user_prompt_segment_modeling,
    "Training & Validation": get_coder_chain_user_prompt_segment_training,
    "Prediction & Submission": get_coder_chain_user_prompt_segment_submission,
}

# --- Self-Reflection Prompts (Simplified for 8B) ---
SEGMENT_REFLECTION_SYSTEM_PROMPT_8B: Dict[str, Any] = {
    "SYSTEM": (
        "Python Code Reviewer for Kaggle. Input: Master Plan, Task Summary, Prior Code, Initial Snippet for current segment. "
        "Review 'Initial Snippet' for correctness, plan adherence, integration. "
        "Output: 'Reflection Summary:' (brief findings) THEN '---' THEN 'Revised Code Snippet:' (```python...``` for THIS segment)."
    ),
    "user_instructions": {
        "Review Focus": "1. Correctness/Bugs. 2. Plan Adherence (Relevant Steps). 3. Integration. 4. '# Thought:' comments. 5. Segment self-containment.",
        "Output Format (Strict)": "Reflection Summary:\n[Brief analysis. If perfect, state so.]\n\n---\nRevised Code Snippet:\n```python\n# Thought: [Updated concise thought]\n[Corrected Python code for THIS SEGMENT ONLY. If no changes, repeat original.]\n```"
    }
}

def get_segment_reflection_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(SEGMENT_REFLECTION_SYSTEM_PROMPT_8B)

def get_segment_reflection_user_prompt(
    task_summary: str, master_plan_text: str, current_segment_name: str,
    code_generated_before_this_segment: str, initial_code_snippet_for_this_segment: str
) -> Dict[str, Any]:
    all_plan_lines = _get_master_plan_lines(master_plan_text)
    relevant_plan = _extract_relevant_plan_steps(master_plan_text, current_segment_name, segments_order, all_plan_lines)
    truncated_prior_code = _truncate_code_so_far(code_generated_before_this_segment, max_lines=15) # Very aggressive for reflection context

    return {
        "Context": {
            "Task Summary": task_summary,
            "Relevant Master Plan Steps": relevant_plan,
            "Current Segment": current_segment_name,
            "Prior Code (Highly Truncated)": wrap_code(truncated_prior_code),
        },
        "Snippet to Review": wrap_code(initial_code_snippet_for_this_segment),
        "Your Task": f"Review 'Snippet to Review' for '{current_segment_name}'. Provide 'Reflection Summary' & 'Revised Code Snippet' (this segment only) per system prompt."
    }

# --- Chunked Reflection (Kept original detailed prompts for now, as per discussion) ---
CHUNK_REFLECTION_SYSTEM_PROMPT: Dict[str, Any] = {
    "SYSTEM": (
        "You are an expert Python Code Reviewer and Refinement Specialist for Kaggle competition solutions. "
        "The solution is organized into multiple segments (e.g. Setup & Imports, Data Loading, etc.). "
        "Now you will be given a *chunk* of these segments together, along with:\n"
        "  - The overall 'Master Plan'.\n"
        # "  - The individual 'Task Summaries' for each segment in the chunk.\n" # Task Summary is overall
        "  - The full code generated *before* this chunk.\n"
        "  - The concatenated 'Initial Chunk Code' for all segments in the chunk.\n\n"
        "Your job is to reflect on this entire chunk at once:  \n"
        "1. **Correctness & Bugs:** Identify any syntax, runtime or logical errors across *any* of the segments in this chunk.  \n"
        "2. **Plan Adherence:** Does each segment accurately implement its part of the Master Plan?  \n"
        "3. **Inter-Segment Consistency:** Do the segments in this chunk integrate seamlessly with one another and with the prior code?  \n"
        "4. **Best Practices & Clarity:** Is the combined code clean, readable, and well-commented?  \n"
        "5. **Boundary Conditions:** Are there missing imports or variable definitions needed by later segments?\n\n"
        "Then produce:\n"
        "  - A single **Reflection Summary** covering the chunk as a whole.\n"
        "  - A single **Revised Code Snippet** (a unified code block replacing the entire chunk)."
    ),
    "user_instructions": {
        "Review Focus Areas for the Initial Chunk Code": [
            "1. Correctness & Bugs across all segments in the chunk.",
            "2. Each segment\u2019s adherence to the Master Plan.",
            "3. Integration among the chunk\u2019s segments and with preceding code.",
            "4. Cleanliness, readability, and appropriate '# Thought:' commentary.",
            "5. Completeness (no missing imports, variables, or function definitions needed later)."
        ],
        "Output Requirements (Strict Adherence Mandatory)": (
            "Your response **MUST** be exactly in two parts, clearly demarcated:\n\n"
            "Reflection Summary:\n"
            "[A concise paragraph or bullet-list summarizing issues or confirmations across the chunk.]\n\n"
            "---\n\n"
            "Revised Code Snippet:\n"
            "```python\n"
            "# Thought: [Your updated, chunk-level reasoning]\n"
            "[The full, corrected Python code covering all segments in this chunk]\n"
            "```"
        )
    }
}
def get_chunked_reflection_system_prompt() -> Dict[str, Any]:
    return deepcopy(CHUNK_REFLECTION_SYSTEM_PROMPT)

def get_chunked_reflection_user_prompt(
    task_summary: str, master_plan: str, segment_names: List[str],
    code_before_chunk: str, initial_chunk_code: str
) -> Dict[str, Any]:
    # For 8B, ensure master_plan and code_before_chunk passed here are also managed for length if possible
    # However, the primary control is within _get_base_coder_chain_user_prompt_args_8b
    truncated_code_before_chunk = _truncate_code_so_far(code_before_chunk, max_lines=15) # Very aggressive

    # For chunk reflection, we might not need to extract specific plan steps for *each* segment in the chunk,
    # but rather provide the relevant portion of the master plan that covers the *entire chunk*.
    # This is tricky. For now, let's pass the full master_plan text, assuming the LLM can pick out relevance.
    # Or, one could concatenate the outputs of _extract_relevant_plan_steps for all segments in the chunk.
    # For simplicity and token saving, perhaps a general instruction is better here.

    relevant_plan_for_chunk = f"Master Plan excerpt relevant to segments: {', '.join(segment_names)}. Review against full plan if needed."
    if len(master_plan.splitlines()) < 20: # If master plan is short, show more
        relevant_plan_for_chunk = master_plan
    else: # Otherwise, try to get combined relevant steps
        all_plan_lines_for_chunk = _get_master_plan_lines(master_plan)
        combined_relevant_text = ""
        for seg_name in segment_names:
            # This re-extraction per segment might be token-heavy if done naively inside the prompt.
            # It's better to pre-process this if possible, or make _extract_relevant_plan_steps very efficient.
            # For now, we pass the full master plan and let the LLM figure it out for the chunk.
            pass
        if not combined_relevant_text: # Fallback
             relevant_plan_for_chunk = "Review the 'Initial Chunk Code' against the overall 'Full Master Plan' for consistency and correctness across these segments: " + ", ".join(segment_names)


    return {
        "Context for Chunk Review": {
            "Overall Task Summary": task_summary,
            "Full Master Plan": master_plan, # Pass the full plan, reflector LLM needs to be robust
            "Segments in This Chunk": segment_names,
            "Python Code Generated Before This Chunk (Truncated)": wrap_code(truncated_code_before_chunk),
        },
        "Initial Chunk Code to Review": wrap_code(initial_chunk_code),
        "Your Chunk Reflection Task": (
            f"Review 'Initial Chunk Code' for segments: {', '.join(segment_names)} as ONE unit. "
            "Address focus areas from system prompt. Output 'Reflection Summary' & 'Revised Code Snippet' (for the whole chunk) per format."
        )
    }









# # aide/utils/prompt_utils.py
# import random
# from typing import Any, Dict, List,Optional
# import copy # For deepcopying system prompts
# from ..backend import FunctionSpec

# # --- Helper for wrapping code (already in your codebase) ---
# def wrap_code(code_str: str, lang: str = "python") -> str:
#     if not code_str: # Handle None or empty string gracefully
#         return f"```{lang}\n# No code provided.\n```" if lang else "```\n# No content provided.\n```"
#     if lang:
#         return f"```{lang}\n{code_str}\n```"
#     return f"```\n{code_str}\n```"

# # --- Data for competitions (from your codebase) ---
# COMPETITION_METADATA = {
#     "aerial-cactus-identification": {"Task Type": "Image Classification", "Size (GB)": 0.0254},
#     "denoising-dirty-documents": {"Task Type": "Image To Image", "Size (GB)": 0.06},
#     "detecting-insults-in-social-commentary": {"Task Type": "Text Classification", "Size (GB)": 0.002},
#     "dog-breed-identification": {"Task Type": "Image Classification", "Size (GB)": 0.75},
#     "dogs-vs-cats-redux-kernels-edition": {"Task Type": "Image Classification", "Size (GB)": 0.85},
#     "jigsaw-toxic-comment-classification-challenge": {"Task Type": "Text Classification", "Size (GB)": 0.06},
#     "leaf-classification": {"Task Type": "Image Classification", "Size (GB)": 0.036},
#     "mlsp-2013-birds": {"Task Type": "Audio Classification", "Size (GB)": 0.5851},
#     "nomad2018-predict-transparent-conductors": {"Task Type": "Tabular", "Size (GB)": 0.00624},
#     "plant-pathology-2020-fgvc7": {"Task Type": "Image Classification", "Size (GB)": 0.8},
#     "random-acts-of-pizza": {"Task Type": "Text Classification", "Size (GB)": 0.003},
#     "spooky-author-identification": {"Task Type": "Text Classification", "Size (GB)": 0.0019},
#     "tabular-playground-series-dec-2021": {"Task Type": "Tabular", "Size (GB)": 0.7},
#     "tabular-playground-series-may-2022": {"Task Type": "Tabular", "Size (GB)": 0.57},
#     "text-normalization-challenge-english-language": {"Task Type": "Seq->Seq", "Size (GB)": 0.01},
#     "text-normalization-challenge-russian-language": {"Task Type": "Seq->Seq", "Size (GB)": 0.01},
# }

# PACKAGE_CATEGORIES = {
#     "common": ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"],
#     "tabular": ["xgboost", "lightgbm", "catboost", "statsmodels"],
#     "image": ["torch", "torchvision", "timm","albumentations", "opencv-python", "Pillow"],
#     "text": ["torch", "transformers", "nltk", "spacy"],
#     "audio": ["torch", "torchaudio", "librosa"],
#     "graph": ["torch-geometric", "networkx"],
#     "optimization": ["bayesian-optimization", "optuna"]
# }

# def get_competition_environment_text(competition_name: str) -> str:
#     """Generates a text string describing the environment and suggested libraries."""
#     # This function remains largely the same as provided in the original agent.py's _prompt_environment

#     if competition_name in COMPETITION_METADATA:
#         current_comp_data = COMPETITION_METADATA[competition_name]
#         task_type = current_comp_data["Task Type"]
#         task_type_lower = task_type.lower()
#         suggested_pkgs = set(PACKAGE_CATEGORIES["common"]) | set(PACKAGE_CATEGORIES["optimization"])
#         task_specific_guidance = ""

#         if "image" in task_type_lower:
#             suggested_pkgs.update(PACKAGE_CATEGORIES["image"])
#             task_specific_guidance = "For this image-based task, libraries like `OpenCV/Pillow`, `torchvision`, and `timm` are highly relevant."
#         elif "tabular" in task_type_lower:
#             suggested_pkgs.update(PACKAGE_CATEGORIES["tabular"])
#             task_specific_guidance = "For tabular data, consider `XGBoost`, `LightGBM`, `CatBoost`, and `Statsmodels`."
#         elif "text" in task_type_lower or "seq->seq" in task_type_lower:
#             suggested_pkgs.update(PACKAGE_CATEGORIES["text"])
#             task_specific_guidance = "For text/NLP tasks, `transformers`, `NLTK`, and `spaCy` are powerful choices."
#         elif "audio" in task_type_lower:
#             suggested_pkgs.update(PACKAGE_CATEGORIES["audio"])
#             task_specific_guidance = "For audio tasks, `torchaudio` and `librosa` are key."
#         elif "graph" in task_type_lower:
#             suggested_pkgs.update(PACKAGE_CATEGORIES["graph"])
#             task_specific_guidance = "For graph tasks, `torch-geometric` and `networkx` are useful."
#         else:
#              task_specific_guidance = "Consider a general set of machine learning libraries."


#         pkgs_list = list(suggested_pkgs)
#         random.shuffle(pkgs_list)
#         pkg_str = ", ".join([f"`{p}`" for p in pkgs_list])

#         return (
#             f"Competition Name: '{competition_name}'\n"
#             f"Task Type: {task_type}\n"
#             f"Data Size (GB): {current_comp_data.get('Size (GB)', 'N/A')}\n\n"
#             f"Installed Packages: Your solution can use any relevant machine learning packages such as: {pkg_str}. {task_specific_guidance} "
#             "Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
#         )
#     else: # Fallback for unknown competition (original Agent's _prompt_environment behavior)
#         pkgs = [
#             "numpy", "pandas", "scikit-learn", "statsmodels", "xgboost",
#             "lightGBM", "torch", "torchvision", "torch-geometric",
#             "bayesian-optimization", "timm",
#         ]
#         random.shuffle(pkgs)
#         pkg_str = ", ".join([f"`{p}`" for p in pkgs])
#         return (
#             f"Competition Name: '{competition_name}' (Details not found in metadata).\n\n"
#             f"Installed Packages: Your solution can use any relevant machine learning packages such as: {pkg_str}. "
#             "Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
#         )

# # --- Static Prompt Components (from Agent) ---
# AGENT_IMPLEMENTATION_GUIDELINE_LIST: List[str] = [
#     "1. Deliver a complete solution, composed of a plan and a code implementations that successfully solves the kaggle competition and saves the submission.csv file ",
#     "2. The plan should not be generic, it should be specific to the task and the data, and should be a single sentence for each step, specefically tailored to the task and the data",
#     "3. the code should be complete, single-file Python script that successfully solves the kaggle competition and saves the submission.csv file ",
#     "4. Implement the solution proposed in your plan.",
#     "5. Calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
#     "6. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",
#     "7. The script must run without errors. Focus on correctness first.",
#     "8. The code should be clean and easy to understand. It should be well-documented and well-structured."
# ]

# AGENT_RESPONSE_FORMAT_TEXT: str = (
#     "Format the response as follows: "
#     "1) PLAN (plain text, no fences):\n as numbered list of steps, each step should be a bullet point, each step should be a single action that can be taken to solve the task"
#     "followed by a single markdown code block (wrapped in ```python ... ```) which implements this solution and prints out the evaluation metric. "
#     "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
#     "Your entire response MUST strictly follow this format:\n\n"
#     "PLAN:\n" # No "1)"
#     "<your step-by-step reasoning here, as detailed bullet points>\n\n" # Removed "plain text, no fences" as it's implied by not having backticks
#     "---\n" # Separator
#     "CODE:\n" # No "2)"
#     "```python\n"
#     "<your python code here, with '# Thought:' comments before logical blocks>\n"
#     "```\n"
#     "There should be NO text before 'PLAN:' and NO text after the final '```'."
# )

# AGENT_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": "You are a Kaggle Grandmaster. You can plan, implement, debug, and improve machine learning engineering code.",
#     "user_instructions": {
#         "Possible Questions you will face": "You will be asked to either come up with a plan and code to solve a Kaggle competition, debug existing code, or improve working code to get better results.",
#         "How to answer the user": (
#             'Whenever you answer, always: '
#             '1. Write a "PLAN:" section in plain text with 7-10*highly detailed, step-by-step bullet points*. Each step should be actionable and explicit, explaining *how* it will be achieved. '
#             'Example plan step: "1. Load \'train.csv\' and \'test.csv\' using pandas, then use train_test_split to split the data to 80%-20% training and validation sets."\n'
#             '2. Then write a "CODE:" section containing exactly one fenced Python block: ```python. Within this code block, *before each major logical section of code*, include a comment explaining your immediate thought process, the specific purpose of that section, and how it relates to your PLAN step. '
#             'Example CODE format: ```python\n'
#             '# Thought: First, I need to load the data using pandas as per step 1 of the plan.\n'
#             'import pandas as pd\n'
#             'train_df = pd.read_csv("./input/train.csv")\n'
#             'test_df = pd.read_csv("./input/test.csv")\n'
#             '# Thought: Now, preprocess the features. Based on preliminary analysis, fill missing numerical values with the mean, as mentioned in the plan.\n'
#             'train_df["Feature"] = train_df["Feature"].fillna(train_df["Feature"].mean())\n'
#         ),
#         "Critical Instruction": "Ensure your plan is explicit and your code is well-commented with your thought process as instructed."
#     },
# }

# AGENT_draft_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": (
#         "You are a Kaggle Grandmaster. Your task is to devise a clear, step-by-step PLAN and then write the corresponding Python CODE to solve machine learning competitions. "
#         "Adhere strictly to the specified output format. The primary goal for this draft is a working, bug-free solution, so prioritize simplicity and correctness in your design."
#         "You might receive a 'Memory' section summarizing previous attempts. Consider this information AND AVOID REPEATING past mistakes or unsuccessful approaches. And it is recommended that you design a different solution from the previous attempts in order to explore more."
#     ),
#     "user_instructions": {
#         "Task Context / Possible Questions": "You will be provided with a description of a Kaggle competition and asked to generate a complete solution, which includes both a PLAN and the corresponding CODE.",
        
#         "How to Answer (Output Structure, Plan Details, Code Details, Examples)": (
#             "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator between the PLAN and CODE sections. There should be no text before 'PLAN:' and no text after the final ``` of the CODE block.\n\n"
#             "**1. PLAN Section Requirements:**\n"
#             "   Construct a \"PLAN:\" section. This plan must consist of 7-10 highly detailed, sequential bullet points. "
#             "   Each point must describe a specific, actionable step required to solve the problem, including *what* to do and *how* it will be achieved (e.g., specific libraries, functions, or techniques to use). "
#             "   This plan will directly guide your code implementation. Avoid overly generic steps and do NOT include steps for Exploratory Data Analysis (EDA). "
#             "   Each bullet point in the PLAN must be self-contained, explaining not just *what* to do but also *how* it will be achieved, mentioning key libraries or specific functions if applicable, as if you are explaining it to someone who will implement it based solely on that plan step.\n\n"
#             "   *Example plan steps demonstrating required detail for a complete simple solution (for a hypothetical customer churn prediction task):*\n"
#             '   "1. **Data Loading**: Load `train.csv` and `test.csv` datasets into pandas DataFrames using `pd.read_csv()`. Store the `customerID` from the test set for later use in the submission file."\n'
#             '   "2. **Target Variable Preparation**: Separate the target variable (e.g., `Churn`) from the features in the training DataFrame. If the target is categorical (e.g., Yes/No), encode it into numerical format (0/1) using `sklearn.preprocessing.LabelEncoder` or a simple map function."\n'
#             '   "3. **Basic Feature Selection & Preprocessing - Numerical**: Identify numerical features. For simplicity in this draft, select a subset of obviously relevant numerical features (e.g., `tenure`, `MonthlyCharges`). Impute any missing values in these selected features using the median strategy with `sklearn.impute.SimpleImputer(strategy=\'median\')`. Fit the imputer on the training data and transform both train and test sets for these features."\n'
#             '   "4. **Basic Feature Preprocessing - Categorical**: Identify categorical features. For simplicity, select a few key categorical features (e.g., `Contract`, `PaymentMethod`). Apply one-hot encoding using `pd.get_dummies()` to these features for both train and test sets. Ensure consistent columns by aligning them post-encoding, possibly by reindexing based on training set columns."\n'
#             '   "5. **Combine Preprocessed Features**: Concatenate the preprocessed numerical and categorical features into final training (X_train_processed) and test (X_test_processed) feature sets using `pd.concat()`."\n'
#             '   "6. **Data Splitting for Validation**: Split `X_train_processed` and the encoded target variable into training and validation subsets (e.g., 80% train, 20% validation) using `sklearn.model_selection.train_test_split`, setting a `random_state` for reproducibility and `stratify` by the target if it\'s a classification task."\n'
#             '   "7. **Model Training**: Instantiate a simple classification model, for example, `sklearn.linear_model.LogisticRegression(random_state=42, solver=\'liblinear\')`. Train this model on the (scaled, if done) training subset (`X_train_fold`, `y_train_fold`)."\n'
#             '   "8. **Validation and Metric Display**: Predict probabilities on the validation subset using `model.predict_proba()[:, 1]` (for the positive class). Calculate and print a relevant validation metric (e.g., ROC AUC using `sklearn.metrics.roc_auc_score`) using the format: `print(f\'Validation ROC AUC: {auc_score}\')`."\n'
#             '   "9. **Test Set Prediction**: Predict probabilities on the fully preprocessed test set (`X_test_processed`) using `model.predict_proba()[:, 1]` to get the likelihood of churn for each test customer."\n'
#             '   "10. **Submission File Generation**: Create a pandas DataFrame for the submission. It should contain the `customerID` column from the original test data and a `Churn` (or the target name specified by the competition) column with the predicted probabilities. Save this DataFrame to `./submission/submission.csv` using `submission_df.to_csv(path, index=False)`."\n\n'
#             "**2. CODE Section Requirements:**\n"
#             "   Follow the PLAN with a \"CODE:\" section, containing a single, complete Python script enclosed in ```python ... ```. "
#             "   Crucially, *before every distinct logical block of code that corresponds to a step in your PLAN*, you MUST include a comment starting with \"# Thought:\". This comment should briefly state: "
#             "   a) Your immediate thought process or strategy for implementing that part. "
#             "   b) The specific purpose of the upcoming code block. "
#             "   c) Which PLAN step number(s) it directly addresses. \n"
#             "   *Example CODE format snippet:*\n"
#             "   ```python\n"
#             "   # Thought: Implementing PLAN step 1. Need to load the training data CSV. Pandas is the standard tool.\n"
#             "   import pandas as pd\n"
#             "   train_df = pd.read_csv(\"./input/train.csv\")\n\n"
#             "   # Thought: Continuing PLAN step 1. Construct full image file paths.\n"
#             "   import os\n"
#             "   IMAGE_DIR = \"./input/train/\"\n"
#             "   train_df[\"filepath\"] = train_df[\"id\"].apply(lambda x: os.path.join(IMAGE_DIR, x))\n"
#             "   ```"
#         ),
        
#         "Critical Adherence / Final Instructions": (
#             "Strict adherence to the detailed PLAN structure (as per the examples provided) and the '# Thought:' commenting convention in the CODE is mandatory. "
#             "The primary objective for this draft is a working, bug-free solution. Therefore, the proposed solution should be simple in its overall design and ideas, focusing on correctness and the avoidance of BUGS. Do NOT include EDA."
#             "You might receive a 'Memory' section summarizing previous attempts. Consider this information AND AVOID REPEATING past mistakes or unsuccessful approaches. Also, it is recommended that you design a different solution from the previous attempts. "

#         )
#     }
# }

# ## Last version abd best so far
AGENT_debug_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": (
        "You are an expert Kaggle Grandmaster, specializing in meticulous, step-by-step debugging of Python machine learning code. "
        "Your task is to analyze provided buggy Python code, its execution traceback, and an initial bug summary. "
        "Based *primarily* on the traceback and the initial summary, you must formulate a precise PLAN to fix the *exact error reported in the traceback*. "
        "Then, you must implement this fix by providing the *entire corrected, runnable Python script*. Prioritize fixing only the immediate bug causing the script to fail, unless other changes are directly necessitated by that fix."
    ),
    "user_instructions": {
        "Input Breakdown": "You will receive: \n1. 'Task Description': The overall goal.\n2. 'Previous (Buggy) Implementation': The full Python script that failed.\n3. 'Execution Output (Traceback)': The error message and stack trace from the last run. This is your primary guide.\n4. 'Initial Bug Summary (from analysis tool)': A brief analysis of the bug. Use this to confirm or refine your own diagnosis based *directly* on the traceback.",
        
        "Output Format (Strict Adherence Required)": (
            "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator between the PLAN and CODE sections. No text before 'PLAN:' and no text after the final ``` of the CODE block.\n\n"
            
            "**1. PLAN Section Requirements:**\n"
            "   a. **Bug Analysis Subsection (Mandatory First Part of PLAN):** Start with 'Bug Analysis:'.\n"
            "      - **Traceback First:** State the specific error type (e.g., `NameError`, `IndexError`) and the exact line number from the 'Execution Output (Traceback)'. Quote the problematic line of code if possible.\n"
            "      - **Root Cause Diagnosis:** Explain *why* this error occurred in the context of the 'Previous (Buggy) Implementation'. Be precise. For instance, if it's a `NameError`, state which name is not defined and why. If it's an `IndexError`, explain which index is out of bounds and for what data structure.\n"
            "      - **Corroborate with Initial Summary:** Refer to the 'Initial Bug Summary'. State if your direct traceback analysis confirms it. If your analysis differs, explain why, always prioritizing the direct evidence from the traceback for the *immediate error*.\n"
            "      - **Focus:** Concentrate only on the error that directly caused the script to terminate as shown in the traceback. Do not speculate on other potential bugs unless they are directly related to the primary error.\n"
            "      *Example Bug Analysis:*\n"
            "      'Bug Analysis:\n      - The traceback indicates a `NameError: name 'np' is not defined` on line 59 of `runfile.py`, within the `CactusDataset.__getitem__` method. The problematic line is `image = np.array(image)`.\n"
            "      - This error occurred because the `numpy` library, aliased as `np`, was used without being imported at the beginning of the script.\n"
            "      - The 'Initial Bug Summary' correctly identified a missing numpy import. My analysis confirms this is the direct cause of the script's failure.'\n\n"
            
            "   b. **Fix Plan Subsection (Following Bug Analysis):** Start with 'Fix Plan:'.\n"
            "      - Provide a concise, bulleted list of the *minimal and targeted changes* required to resolve *only the root cause(s)* identified in your Bug Analysis.\n"
            "      - Each step must clearly state *what* code will be added or modified, *where* (e.g., which function, beginning of script), and *how* this directly fixes the identified error.\n"
            "      - Do NOT propose new features, unrelated refactoring, or performance optimizations in this debugging step. The goal is a correct, runnable script that fixes the immediate error.\n"
            "      *Example Fix Plan:*\n"
            "      'Fix Plan:\n      1. Add the import statement `import numpy as np` at the top of the script, among other imports. This makes the `np` alias available globally, resolving the `NameError`.\n      2. Ensure no other part of the script was relying on `np` being undefined (unlikely, but a mental check).'\n\n"

            "**2. CODE Section Requirements:**\n"
            "   Follow the PLAN with a \"CODE:\" section. This section must contain a *single, complete, and runnable* Python script enclosed in ```python ... ```. "
            "   This script should be the *entire* previous buggy script, with *only the necessary modifications* as outlined in your 'Fix Plan' to address the identified bug.\n"
            "   *Before each modified or newly added logical block of code that implements a step from your 'Fix Plan'*, you MUST include a comment starting with \"# Bugfix Thought:\". This comment should briefly state:\n"
            "   a) The specific bug being addressed from your Bug Analysis (e.g., 'Addressing NameError for np').\n"
            "   b) How the code change implements the corresponding 'Fix Plan' step.\n"
            "   c) A concise thought on the change (e.g., 'Standard import for numpy.').\n"
            "   *Example CODE snippet for a bugfix:*\n"
            "   ```python\n"
            "   # Bugfix Thought: Addressing NameError for 'np' from Bug Analysis. Implementing Fix Plan step 1: Add numpy import.\n"
            "   import numpy as np # FIX: Added this line for numerical operations.\n"
            "   import os\n"
            "   import pandas as pd\n"
            "   # ... (rest of the original imports and code) ...\n\n"
            "   # ... (original code until the part that was buggy) ...\n"
            "       # Bugfix Thought: The line `image = np.array(image)` previously caused a NameError. With `np` now imported, this line is correct.\n"
            "       image = np.array(image) # This was the failing line, now fixed by the import.\n"
            "   # ... (rest of the function and script) ...\n"
            "   ```\n"
        ),
        
        "Critical Adherence / Final Instructions": (
            "Strict adherence to the 'Bug Analysis' and 'Fix Plan' structure is mandatory. The CODE section must contain the *entire runnable script* with minimal targeted fixes. "
            "Focus *exclusively* on fixing the bug(s) directly identified from the traceback and confirmed with the Initial Bug Summary. "
            "Do NOT introduce new features, unrelated refactoring, or performance optimizations during this debug step. Ensure all original, necessary imports are preserved and any new ones required for the fix are added."
        )
    }
}


# AGENT_improve_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": (
#         "You are an analytical Kaggle Grandmaster, focused on iterative performance enhancement. "
#         "Given a working Python solution for a machine learning competition, your task is to: "
#         "1. Propose *one single, atomic, and well-justified* improvement to enhance its predictive performance (e.g., reduce log loss, increase AUC). "
#         "2. Create a detailed PLAN explaining this specific improvement: what it is, *how* it will be implemented, and *why* it's expected to boost performance. "
#         "3. Provide the *entire modified, runnable Python CODE* implementing only this single improvement. "
#         "Adhere strictly to the specified output format."
#     ),
#     "user_instructions": {
#         "Input Provided": "You will receive: the 'Task Description', the 'Previous (working) solution's CODE', and a 'Memory' of past attempts (if any).",
        
#         "Output Format (Strict Adherence Required)": (
#             "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator. No text before 'PLAN:' or after the final ``` of the CODE block.\n\n"
            
#             "**1. PLAN Section Requirements:**\n"
#             "   a. **Improvement Rationale (Brief Introduction - 2-3 sentences):** Briefly state the single improvement you are proposing and the core reason you believe it will enhance performance based on the previous solution and general ML principles.\n"
#             "      *Example Rationale:* 'The previous solution used a simple CNN. To potentially capture more complex image features and improve AUC, I propose replacing it with a pre-trained ResNet18 model, fine-tuning its final layers.'\n\n"
#             "   b. **Detailed Improvement Plan (Bulleted List - 3-7 detailed steps):** \n"
#             "      - Outline the precise, actionable steps to implement *only this single improvement*. \n"
#             "      - Each step must detail *what* changes will be made to the existing code, *where* these changes will occur (e.g., which functions/classes), and *how* (e.g., specific libraries, functions, parameter changes). \n"
#             "      - Crucially, for each step, explain *why* this specific modification contributes to the overall proposed improvement and is expected to lead to better performance. \n"
#             "      - If introducing new libraries (e.g., Albumentations for advanced augmentation), explicitly mention the import and the key components to be used.\n"
#             "      *Example Detailed Improvement Plan Steps (for switching to ResNet18):*\n"
#             "      '1. **Modify Imports**: Add `from torchvision import models` to import pre-trained models. Ensure `torch.nn as nn` is present.\n"
#             "      2. **Replace Model Architecture**: In the `CactusClassifier` class (or equivalent), remove the existing `self.conv_layers` and `self.fc_layers`. Instantiate `models.resnet18(pretrained=True)` as the backbone. Explain that pre-trained weights capture general image features.\n"
#             "      3. **Adapt Final Layer**: The ResNet18 `fc` layer outputs 1000 classes. Replace `model.fc` with a new `nn.Linear(model.fc.in_features, 1)` followed by `nn.Sigmoid()` for binary classification. This adapts the ResNet to the specific task.\n"
#             "      4. **Adjust Image Preprocessing**: ResNet models are typically trained on images normalized with ImageNet statistics and often larger input sizes (e.g., 224x224, though 32x32 can still work but resizing might be an option). Update the `transforms.Compose` to include `transforms.Resize((desired_size, desired_size))` (e.g., 32 or 64 for this task) and `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`. This ensures input data matches ResNet's expectations.\n"
#             "      5. **Adjust Learning Rate (Potentially Lower):** Pre-trained models often benefit from a smaller learning rate for fine-tuning. Consider reducing the LR in `optim.Adam` (e.g., to 1e-4 or 5e-4) to prevent distorting pre-trained weights too quickly.'\n\n"

#             "**2. CODE Section Requirements:**\n"
#             "   Follow the PLAN with a \"CODE:\" section, containing a single, *complete, and runnable* Python script enclosed in ```python ... ```. "
#             "   This script should be the *entire* previous working script, with *only the modifications outlined in your Detailed Improvement Plan* to implement the single proposed enhancement.\n"
#             "   *Before each modified or newly added logical block of code*, you MUST include a comment starting with \"# Improvement Thought:\". This comment should briefly state:\n"
#             "   a) The specific part of the improvement being implemented.\n"
#             "   b) How the code change relates to the corresponding 'Detailed Improvement Plan' step.\n"
#             "   c) A concise thought on the change (e.g., 'Replacing custom CNN with ResNet18 for better feature extraction.').\n"
#             "   Ensure all original necessary imports are preserved and any new ones required for the improvement are added. Do not remove unrelated working code.\n"
#         ),
        
#         "Critical Adherence / Final Instructions": (
#             "Strict adherence to proposing a *single, atomic improvement* and detailing it in the PLAN is mandatory. The CODE section must contain the *entire runnable script* with minimal targeted changes for that one improvement. "
#             "Clearly justify the improvement. Do NOT include EDA. Ensure all necessary imports are present."
#         )
#     }
# }


# AGENT_DRAFT_SOLUTION_GUIDELINE_LIST: List[str] = [
#     "This is an initial solution draft. Prioritize a simple, correct, and bug-free design. Avoid complex techniques like ensembling or extensive hyper-parameter optimization for this iteration.",
#     "If a 'Memory' section is provided (summarizing previous attempts), consider its contents to avoid repeating past mistakes or unsuccessful approaches.",
#     "Do not include steps for Exploratory Data Analysis (EDA) in your plan or code.",
#     "The PLAN should consist of 7-10 bullet points. Each point must be explicit, detailed, and actionable, clearly stating *how* the step will be accomplished.", # Aligns with 7-10
#     "The evaluation metric to be used is typically found in the 'Overall Task Description'. Ensure your code calculates and prints a validation metric, preferably this one.",
#     "Assume all necessary data files are already prepared and located in the `./input/` directory. No unzipping or complex data discovery is needed.",
#     "If you receive a 'Memory of Previous Attempts', learn from it and avoid repeating unsuccessful strategies or explicitly address past issues if relevant to the new plan. It is also recommended that you design a different solution from the previous attempts in order to explore more.",
# ]

# AGENT_IMPROVE_SOLUTION_GUIDELINE_LIST: List[str] = [
#     "Propose *one single, atomic, and well-justified* improvement to the provided working solution. Do not suggest multiple changes at once.",
#     "Your PLAN must start with a brief 'Improvement Rationale' (2-3 sentences) explaining your chosen improvement and why it should boost performance.",
#     "Following the rationale, provide a 'Detailed Improvement Plan' as a bulleted list of 3-7 steps. Each step must be highly specific about *what* code to change, *where*, *how* (mentioning specific libraries/functions/parameters), and *why* this contributes to the improvement.",
#     "The CODE section must implement *only* this single improvement, modifying the provided previous solution. It must be a complete, runnable script.",
#     "Use '# Improvement Thought:' comments before each changed/new code block, linking it to your plan and explaining your reasoning for that specific code modification.",
#     "Consider the 'Memory' section (if provided) to avoid repeating unsuccessful strategies or to build on previously identified good ideas.",
#     "Do not suggest Exploratory Data Analysis (EDA). Focus on a direct code/model/feature enhancement."
# ]

# AGENT_DEBUG_SOLUTION_GUIDELINE_LIST: List[str] = [
#     "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
#     "Don't suggest to do EDA.",
#     "Your PLAN should start with a 'Bug Analysis:' section. In this section, meticulously analyze the 'Execution output' and the 'Previous (buggy) implementation' line by line (or logical block by block) to identify the root cause of the bug. State concrete observations, e.g., 'Line X: FileNotFoundError because path was incorrect. This indicates the file path in the code is incorrect.'",
#     "Following the 'Bug Analysis:', provide a 'Fix Plan:' with highly detailed, actionable steps to resolve *each identified bug*. Each step should explain *what* will be changed and *why*. For example, 'Fix Plan: 1. Update file path for train.csv to ./input/train.csv to match the correct directory structure, as indicated by the FileNotFoundError.'",
#     "In your CODE, before each modified or new logical block, add a comment explaining the specific bug being addressed by that code, how the change fixes it, and your thought process. Example: # Bugfix: Handle division by zero. Previous code caused ZeroDivisionError on line Y. Added a check here to prevent it by replacing NaN with 0."
# ]

# # --- System Prompt Getters ---
# def get_agent_system_prompt() -> Dict[str, Any]:
#     return copy.deepcopy(AGENT_SYSTEM_PROMPT_DICT)

# def get_agent_draft_system_prompt() -> Dict[str, Any]:
#     return copy.deepcopy(AGENT_draft_SYSTEM_PROMPT_DICT)

# # --- User Message Assemblers for Agent ---
# def get_agent_draft_user_prompt(
#     task_desc: str,
#     journal_summary: str,
#     competition_name: str,
#     obfuscate: bool,
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     introduction = "You are a Kaggle grandmaster. Plan and develop a complete Python script to solve the described machine learning competition."
#     if obfuscate:
#         introduction = "You are an expert machine learning engineer. Your task is to develop a complete Python script to solve the described machine learning problem."
#     data_prev = " "
#     if acfg_data_preview and data_preview_content:
#         data_prev= data_preview_content
#     # This structure matches the original _draft method's prompt_user_message
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": introduction,
#         "Overall Task Description": task_desc,
#         "Data Overview" :data_preview_content,
#         "Memory (Summary of Previous Attempts on this Task)": journal_summary,
#         "Instructions": {
#             "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
#             "Environment and Packages": get_competition_environment_text(competition_name),
#             "Response format": AGENT_RESPONSE_FORMAT_TEXT,
#             "Solution sketch guideline": AGENT_DRAFT_SOLUTION_GUIDELINE_LIST,        },
#     }

#     return prompt_user_message

# def get_agent_improve_user_prompt(
#     task_desc: str,
#     journal_summary: str,
#     competition_name: str,
#     parent_node_code: str,
# ) -> Dict[str, Any]:
#     introduction = (
#         "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
#         "solution below and should improve it in order to further increase the (test time) performance. "
#         "For this you should first outline a brief plan in natural language for how the solution can be improved and "
#         "then implement this improvement in Python based on the provided previous solution. "
#     )
#     # This structure matches the original _improve method's prompt
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": introduction,
#         "Task description": task_desc,
#         "Memory": journal_summary,
#         "Previous solution": { # Original had this as a top-level key, kept for consistency
#             "Code": wrap_code(parent_node_code),
#         },
#         "Instructions": {
#             "Response format": AGENT_RESPONSE_FORMAT_TEXT,
#             "Solution improvement sketch guideline": AGENT_IMPROVE_SOLUTION_GUIDELINE_LIST,
#             "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
#             "Environment and Packages": get_competition_environment_text(competition_name) # Added for consistency
#         },
#     }
#     return prompt_user_message

# def get_agent_debug_user_prompt(
#     task_desc: str,
#     competition_name: str, # For Environment and Packages
#     parent_node_code: str,
#     parent_node_term_out: str,
#     parent_node_feedback: str, # <-- NEW: Summary from o3-mini or your feedback model
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     introduction = (
#         "You are a Kaggle grandmaster tasked with debugging a Python script. "
#         "Your previous solution attempt resulted in an error or failed to produce the required `submission.csv`. "
#         "Analyze the provided buggy code, its execution traceback, and an initial bug summary. "
#         "Then, formulate a precise PLAN (Bug Analysis, then Fix Plan) and provide the fully corrected Python CODE."
#     )
    
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": introduction,
#         "Task Description": task_desc,
#         "Previous (Buggy) Implementation": wrap_code(parent_node_code),
#         "Execution Output (Traceback)": wrap_code(parent_node_term_out, lang=""), # Explicitly label as traceback
#         "Initial Bug Summary (from analysis tool)": parent_node_feedback if parent_node_feedback else "No initial summary provided. Perform analysis based on traceback and code.",
#         "Instructions": {
#             # "Response format": AGENT_RESPONSE_FORMAT_TEXT, # This is now handled in the SYSTEM prompt
#             "Debug Guidelines (Refer to System Prompt for full details)": AGENT_DEBUG_SOLUTION_GUIDELINE_LIST, # Keep this for detailed points
#             "General Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST, # These are still good general rules
#             "Environment and Packages": get_competition_environment_text(competition_name)
#         },
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message


# review_func_spec = FunctionSpec(
#     name="submit_review",
#     json_schema={
#         "type": "object",
#         "properties": {
#             "is_bug": {
#                 "type": "boolean",
#                 "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
#             },
#             "has_csv_submission": {
#                 "type": "boolean",
#                 "description": "true if the code saves the predictions on the test data"
#                 " in a `submission.csv` file in the `./submission/` directory, otherwise false."
#                 " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
#                 " Otherwise, it should be evaluated as false."
#                 " You can assume the ./submission/ directory exists and is writable.",
#             },
#             "summary": {
#                 "type": "string",
#                 "description": "write a short summary (2-3 sentences) describing "
#                 " the empirical findings. Alternatively mention if there is a bug or"
#                 " the submission.csv was not properly produced."
#                 " DO NOT suggest fixes or improvements.",
#             },
#             "metric": {
#                 "type": "number",
#                 "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
#             },
#             "lower_is_better": {
#                 "type": "boolean",
#                 "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
#             },
#             "code_quality": {
#                 "type": "number",
#                 "description": "give a score between 0-10 on the quality of the code, where 0 is a terrible code/ non-code at all, and 9-10 is a clean code with a great value for the evaluation metric.",
#             },
#         },
#         "required": [
#             "is_bug",
#             "has_csv_submission",
#             "summary",
#             "metric",
#             "lower_is_better",
#             "code_quality",
#         ],
#     },
#     description="Submit a review evaluating the output of the training script.",
# )

# # Experimental
# # AGENT_debug_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
# #     "SYSTEM": (
# #         "You are an expert Kaggle Grandmaster, specializing in meticulous, step-by-step debugging of Python machine learning code. "
# #         "Your primary task is to analyze the provided buggy Python script, its execution traceback, and an initial bug summary. "
# #         "Based *primarily* on the traceback and the initial summary, you must formulate a precise PLAN to fix the *exact error reported in the traceback*. "
# #         "Then, you must implement this fix by providing the *entire corrected, runnable Python script*, making only the absolute minimal changes necessary to resolve the identified error."
# #     ),
# #     "user_instructions": {
# #         "Input Breakdown": "You will receive: \n1. 'Task Description': The overall goal.\n2. 'Previous (Buggy) Implementation': The full Python script that failed.\n3. 'Execution Output (Traceback)': The error message and stack trace from the last run. This is your primary guide.\n4. 'Initial Bug Summary (from analysis tool)': A brief analysis of the bug. Use this to confirm or refine your own diagnosis based *directly* on the traceback.",
        
# #         "Output Format (Strict Adherence Required)": (
# #             "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator. No text before 'PLAN:' or after the final ``` of the CODE block.\n\n"
            
# #             "**1. PLAN Section Requirements:**\n"
# #             "   a. **Bug Analysis Subsection (Mandatory First Part of PLAN):** Start with 'Bug Analysis:'. \n"
# #             "      - **Traceback First:** State the specific error type (e.g., `NameError`, `IndexError`, `KeyError`, `ValueError`) and the exact line number from the 'Execution Output (Traceback)'. Quote the problematic line of code from 'Previous (Buggy) Implementation'.\n"
# #             "      - **Root Cause Diagnosis:** Explain *precisely why* this error occurred. For instance, if `KeyError: 'author'`, state: 'The DataFrame `test_df` does not contain a column named 'author' when `label_encoder.transform(test_df['author'])` is called.' If `ValueError: y contains previously unseen labels: 0` for `log_loss(label_encoder.transform(y_val_split), val_preds)`, state: 'The `y_val_split` variable already contains numerically encoded labels (e.g., 0, 1, 2). Applying `label_encoder.transform()` to these numeric values is incorrect as the encoder expects original string labels (EAP, HPL, MWS).'\n"
# #             "      - **Corroborate with Initial Summary:** Refer to the 'Initial Bug Summary'. State if your direct traceback analysis confirms it. If your analysis differs, explain why, always prioritizing the direct evidence from the traceback for the *immediate error*.\n"
# #             "      - **Focus:** Concentrate *only* on the error that directly caused the script to terminate. Do not speculate on other potential bugs or suggest unrelated improvements at this stage.\n\n"
            
# #             "   b. **Fix Plan Subsection (Following Bug Analysis):** Start with 'Fix Plan:'.\n"
# #             "      - Provide a concise, bulleted list of the *minimal and targeted code changes* required to resolve *only the root cause(s)* identified in your Bug Analysis. \n"
# #             "      - Each step must clearly state *what* code will be added, removed, or modified, *where* (e.g., which function, specific line if possible), and *how* this directly fixes the identified error. \n"
# #             "      - If a variable's state is the issue (e.g., already encoded), the fix is often to *not* re-apply a transformation. \n"
# #             "      *Example Fix Plan for ValueError with `label_encoder.transform(y_val_split)`:*\n"
# #             "      'Fix Plan:\n      1. Modify the line calculating `validation_loss`. Instead of `log_loss(label_encoder.transform(y_val_split), val_preds)`, use `log_loss(y_val_split, val_preds)` directly, because `y_val_split` already contains the numerically encoded labels suitable for `log_loss`.\n      2. No other changes are required to fix this specific `ValueError`.'\n\n"

# #             "**2. CODE Section Requirements:**\n"
# #             "   Follow the PLAN with a \"CODE:\" section, containing a single, *complete, and runnable* Python script enclosed in ```python ... ```. "
# #             "   This script should be the *entire* previous buggy script, with *only the minimal modifications* as per your 'Fix Plan'.\n"
# #             "   *Before each modified or newly added logical block of code related to the fix*, you MUST include a comment starting with \"# Bugfix Thought:\". This comment should briefly state:\n"
# #             "   a) The specific bug being addressed (e.g., 'Addressing ValueError from re-transforming y_val_split').\n"
# #             "   b) How the code change implements the corresponding 'Fix Plan' step.\n"
# #             "   c) A concise thought on the change (e.g., 'Using y_val_split directly as it is already encoded.').\n"
# #             "   Ensure all original, necessary imports are preserved and any new ones required for the fix are added. Do not remove unrelated working code.\n"
# #         ),
        
# #         "Critical Adherence / Final Instructions": (
# #             "Strict adherence to the 'Bug Analysis' and 'Fix Plan' structure is mandatory. The CODE section must contain the *entire runnable script* with minimal targeted fixes. "
# #             "Focus *exclusively* on fixing the bug(s) directly identified from the traceback and confirmed with the Initial Bug Summary. "
# #             "Do NOT introduce new features, unrelated refactoring, or performance optimizations during this debug step. Verify variable states before applying transformations."
# #         )
# #     }
# # }


# # def get_agent_debug_user_prompt(
# #     task_desc: str,
# #     competition_name: str,
# #     parent_node_code: str,
# #     parent_node_term_out: str,
# #     parent_node_feedback: str,
# #     acfg_data_preview: bool,
# #     data_preview_content: str = None
# # ) -> Dict[str, Any]:
# #     introduction = (
# #         "You are a Kaggle grandmaster attending a competition. "
# #         "Your previous solution had a bug and/or did not produce a submission.csv, "
# #         "so based on the information below, you should revise it in order to fix this. "
# #         "Your response should be an implementation plan in natural language,"
# #         " followed by a single markdown code block which implements the bugfix/solution."
# #     )
# #     # This structure matches the original _debug method's prompt
# #     prompt_user_message: Dict[str, Any] = {
# #         "Introduction": introduction,
# #         "Task description": task_desc,
# #         "Previous (buggy) implementation": wrap_code(parent_node_code),
# #         "Execution output": wrap_code(parent_node_term_out, lang=""),
# #         "Instructions": {
# #             "Response format": AGENT_RESPONSE_FORMAT_TEXT,
# #             "Bugfix improvement sketch guideline": AGENT_DEBUG_SOLUTION_GUIDELINE_LIST,
# #             "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
# #             "Environment and Packages": get_competition_environment_text(competition_name) # Added for consistency
# #         },
# #     }
# #     if acfg_data_preview and data_preview_content:
# #         prompt_user_message["Data Overview"] = data_preview_content
# #     return prompt_user_message
