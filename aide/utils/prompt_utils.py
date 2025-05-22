# aide/utils/prompt_utils.py
import random
from typing import Any, Dict, List
import copy # For deepcopying system prompts

# --- Helper for wrapping code (already in your codebase) ---
def wrap_code(code_str: str, lang: str = "python") -> str:
    if not code_str: # Handle None or empty string gracefully
        return f"```{lang}\n# No code provided.\n```" if lang else "```\n# No content provided.\n```"
    if lang:
        return f"```{lang}\n{code_str}\n```"
    return f"```\n{code_str}\n```"

# --- Data for competitions (from your codebase) ---
COMPETITION_METADATA = {
    "aerial-cactus-identification": {"Task Type": "Image Classification", "Size (GB)": 0.0254},
    "aptos2019-blindness-detection": {"Task Type": "Image Classification", "Size (GB)": 10.22},
    "denoising-dirty-documents": {"Task Type": "Image To Image", "Size (GB)": 0.06},
    "detecting-insults-in-social-commentary": {"Task Type": "Text Classification", "Size (GB)": 0.002},
    "dog-breed-identification": {"Task Type": "Image Classification", "Size (GB)": 0.75},
    "dogs-vs-cats-redux-kernels-edition": {"Task Type": "Image Classification", "Size (GB)": 0.85},
    "histopathologic-cancer-detection": {"Task Type": "Image Regression", "Size (GB)": 7.76},
    "jigsaw-toxic-comment-classification-challenge": {"Task Type": "Text Classification", "Size (GB)": 0.06},
    "leaf-classification": {"Task Type": "Image Classification", "Size (GB)": 0.036},
    "mlsp-2013-birds": {"Task Type": "Audio Classification", "Size (GB)": 0.5851},
    "new-york-city-taxi-fare-prediction": {"Task Type": "Tabular", "Size (GB)": 5.7},
    "nomad2018-predict-transparent-conductors": {"Task Type": "Tabular", "Size (GB)": 0.00624},
    "plant-pathology-2020-fgvc7": {"Task Type": "Image Classification", "Size (GB)": 0.8},
    "random-acts-of-pizza": {"Task Type": "Text Classification", "Size (GB)": 0.003},
    "ranzcr-clip-catheter-line-classification": {"Task Type": "Image Classification", "Size (GB)": 13.13},
    "siim-isic-melanoma-classification": {"Task Type": "Image Classification", "Size (GB)": 116.16},
    "spooky-author-identification": {"Task Type": "Text Classification", "Size (GB)": 0.0019},
    "tabular-playground-series-dec-2021": {"Task Type": "Tabular", "Size (GB)": 0.7},
    "tabular-playground-series-may-2022": {"Task Type": "Tabular", "Size (GB)": 0.57},
    "text-normalization-challenge-english-language": {"Task Type": "Seq->Seq", "Size (GB)": 0.01},
    "text-normalization-challenge-russian-language": {"Task Type": "Seq->Seq", "Size (GB)": 0.01},
    "the-icml-2013-whale-challenge-right-whale-redux": {"Task Type": "Audio Classification", "Size (GB)": 0.29314},
}

PACKAGE_CATEGORIES = {
    "common": ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"],
    "tabular": ["xgboost", "lightgbm", "catboost", "statsmodels"],
    "image": ["torch", "torchvision", "timm", "opencv-python", "Pillow", "albumentations"],
    "text": ["torch", "transformers", "nltk", "spacy"],
    "audio": ["torch", "torchaudio", "librosa"],
    "graph": ["torch-geometric", "networkx"],
    "optimization": ["bayesian-optimization", "optuna"]
}

def get_competition_environment_text(competition_name: str) -> str:
    """Generates a text string describing the environment and suggested libraries."""
    # This function remains largely the same as provided in the original agent.py's _prompt_environment
    # for PlannerAgent, but made more robust for unknown competitions and task types.
    if competition_name in COMPETITION_METADATA:
        current_comp_data = COMPETITION_METADATA[competition_name]
        task_type = current_comp_data["Task Type"]
        task_type_lower = task_type.lower()
        suggested_pkgs = set(PACKAGE_CATEGORIES["common"]) | set(PACKAGE_CATEGORIES["optimization"])
        task_specific_guidance = ""

        if "image" in task_type_lower:
            suggested_pkgs.update(PACKAGE_CATEGORIES["image"])
            task_specific_guidance = "For this image-based task, libraries like `torchvision`, `timm`, `albumentations`, and `OpenCV/Pillow` are highly relevant."
        elif "tabular" in task_type_lower:
            suggested_pkgs.update(PACKAGE_CATEGORIES["tabular"])
            task_specific_guidance = "For tabular data, consider `XGBoost`, `LightGBM`, `CatBoost`, and `Statsmodels`."
        elif "text" in task_type_lower or "seq->seq" in task_type_lower:
            suggested_pkgs.update(PACKAGE_CATEGORIES["text"])
            task_specific_guidance = "For text/NLP tasks, `transformers`, `NLTK`, and `spaCy` are powerful choices."
        elif "audio" in task_type_lower:
            suggested_pkgs.update(PACKAGE_CATEGORIES["audio"])
            task_specific_guidance = "For audio tasks, `torchaudio` and `librosa` are key."
        elif "graph" in task_type_lower:
            suggested_pkgs.update(PACKAGE_CATEGORIES["graph"])
            task_specific_guidance = "For graph tasks, `torch-geometric` and `networkx` are useful."
        else:
             task_specific_guidance = "Consider a general set of machine learning libraries."


        pkgs_list = list(suggested_pkgs)
        random.shuffle(pkgs_list)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs_list])

        return (
            f"Competition Name: '{competition_name}'\n"
            f"Task Type: {task_type}\n"
            f"Data Size (GB): {current_comp_data.get('Size (GB)', 'N/A')}\n\n"
            f"Installed Packages: Your solution can use any relevant machine learning packages such as: {pkg_str}. {task_specific_guidance} "
            "Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        )
    else: # Fallback for unknown competition (original Agent's _prompt_environment behavior)
        pkgs = [
            "numpy", "pandas", "scikit-learn", "statsmodels", "xgboost",
            "lightGBM", "torch", "torchvision", "torch-geometric",
            "bayesian-optimization", "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])
        return (
            f"Competition Name: '{competition_name}' (Details not found in metadata).\n\n"
            f"Installed Packages: Your solution can use any relevant machine learning packages such as: {pkg_str}. "
            "Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        )

# --- Static Prompt Components (from Agent) ---
AGENT_IMPLEMENTATION_GUIDELINE_LIST: List[str] = [
    "1. Write a complete, single-file Python script. ",
    "2. starting with imports, and load necessary data from the './input/' directory.",
    "3. Implement the solution proposed in your plan.",
    "4. Calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
    "5. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",
    "6. The script must run without errors. Focus on correctness first.",
    "7. The code should be clean and easy to understand. It should be well-documented and well-structured."
]

AGENT_RESPONSE_FORMAT_TEXT: str = (
    "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
    "followed by a single markdown code block (wrapped in ```python ... ```) which implements this solution and prints out the evaluation metric. "
    "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    "explicitly,structure your answer exactly like this: "
    "\n\n---\n"
    "1) PLAN (plain text, no fences):\n"
    "<your step‑by‑step reasoning here>\n\n"
    "2) CODE (one fenced Python block):\n"
    "```python\n"
    "<your python code here>\n"
    "```"
)

AGENT_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": "You are a Kaggle Grandmaster. You can plan, implement, debug, and improve machine learning engineering code.",
    "user_instructions": {
        "Possible Questions you will face": "You will be asked to either come up with a plan and code to solve a Kaggle competition, debug existing code, or improve working code to get better results.",
        "How to answer the user": (
            'Whenever you answer, always: '
            '1. Write a "PLAN:" section in plain text with 3-5 concise, *highly detailed, step-by-step bullet points*. Each step should be actionable and explicit, explaining *how* it will be achieved. '
            'Example plan step: "1. Load \'train.csv\' and \'test.csv\' using pandas, then use train_test_split to split the data to 80%-20% training and validation sets."\n'
            '2. Then write a "CODE:" section containing exactly one fenced Python block: ```python. Within this code block, *before each major logical section of code*, include a comment explaining your immediate thought process, the specific purpose of that section, and how it relates to your PLAN step. '
            'Example CODE format: ```python\n'
            '# Thought: First, I need to load the data using pandas as per step 1 of the plan.\n'
            'import pandas as pd\n'
            'train_df = pd.read_csv("./input/train.csv")\n'
            'test_df = pd.read_csv("./input/test.csv")\n'
            '# Thought: Now, preprocess the features. Based on preliminary analysis, fill missing numerical values with the mean, as mentioned in the plan.\n'
            'train_df["Feature"] = train_df["Feature"].fillna(train_df["Feature"].mean())\n'
            # The empty string at the end `''` was removed, it's not necessary.
        ),
        "Critical Instruction": "Ensure your plan is explicit and your code is well-commented with your thought process as instructed."
    },
}

AGENT_DRAFT_SOLUTION_GUIDELINE_LIST: List[str] = [
    "This is a first draft solution and we will refine it iterativly, so the idea and design for the solution should be relatively simple, without ensembling or hyper-parameter optimization. or any complex approach, we simply want a working first draft",
    "Take the Memory section into consideration when proposing the design.",
    "The solution sketch should be 5-7 steps.",
    "Propose an evaluation metric that is reasonable for this task., you will find the desired metric in the task description",
    "Don't suggest to do EDA.",
    "The data is already prepared and available in the `./input` directory. There is no need to suggest any unzip step to any file.",
]

AGENT_IMPROVE_SOLUTION_GUIDELINE_LIST: List[str] = [
    "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
    "You should be very specific and should only propose a single actionable improvement.",
    "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
    "Take the Memory section into consideration when proposing the improvement.",
    "Each bullet point in your PLAN should specify the exact improvement, *how* it will be implemented, and *why* it's expected to improve performance. For example, instead of 'Add more features', write 'Derive new features by calculating interaction terms between FeatureA and FeatureB, as this might capture non-linear relationships.'",
    "In your CODE, before each modified or new logical block, add a comment explaining the purpose of the change, how it relates to the improvement plan, and your thought process.",
    "Don't suggest to do EDA.",
]

AGENT_DEBUG_SOLUTION_GUIDELINE_LIST: List[str] = [
    "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
    "Don't suggest to do EDA.",
    "Your PLAN should start with a 'Bug Analysis:' section. In this section, meticulously analyze the 'Execution output' and the 'Previous (buggy) implementation' line by line (or logical block by block) to identify the root cause of the bug. State concrete observations, e.g., 'Line X: FileNotFoundError because path was incorrect. This indicates the file path in the code is incorrect.'",
    "Following the 'Bug Analysis:', provide a 'Fix Plan:' with highly detailed, actionable steps to resolve *each identified bug*. Each step should explain *what* will be changed and *why*. For example, 'Fix Plan: 1. Update file path for train.csv to ./input/train.csv to match the correct directory structure, as indicated by the FileNotFoundError.'",
    "In your CODE, before each modified or new logical block, add a comment explaining the specific bug being addressed by that code, how the change fixes it, and your thought process. Example: # Bugfix: Handle division by zero. Previous code caused ZeroDivisionError on line Y. Added a check here to prevent it by replacing NaN with 0."
]

# --- Static Prompt Components (from PlannerAgent) ---
PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT: str = (
    "Your response for the summary should be a detailed and high quality bullet points of what the task is about, summarizing all the information in the task description (5-7 sentences), "
    "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
    "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Task Summary: and natural language text (plan) under ## Plan: "
    "explicitly,structure your answer exactly like this: "
    "\n\n---\n"
    "## Task Summary: (plain text, no fences):\n"
    "<your step‑by‑step reasoning abd summary of the task here>\n\n"
    "## Plan: (plain text, no fences):\n"
    "<your step‑by‑step reasoning and plan steps here>\n\n"
)

PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT: str = (
    "Your response should be a single markdown code block (wrapped in ```python ... ```) which implements this solution and prints out the evaluation metric. "
    "There should be no additional headings or text in your response. Just the markdown code block. "
    "explicitly,structure your answer exactly like this: "
    "\n\n---\n"
    "1) CODE (one fenced Python block):\n"
    "```python\n<your python code here>\n```"
)

PLANNER_AGENT_DEBUG_RESPONSE_FORMAT_TEXT: str = (
    "Your response for the summary should be a detailed and high quality bullet points of the bugs in the previous solution, summarizing all the information and problems(5-7 sentences), "
    "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
    "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Bugs Summary/Analysis: and natural language text (plan) under ## Plan: "
    "explicitly,structure your answer exactly like this: "
    "\n\n---\n"
    "## Bugs Summary/Analysis: (plain text, no fences):\n"
    "<your step‑by‑step reasoning abd summary of the bugs in the previous solution here>\n\n"
    "## Plan: (plain text, no fences):\n"
    "<your step‑by‑step reasoning and plan steps for fixing the bugs here>\n\n"
)

PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": "You are a Kaggle Grandmaster and a team leader. you can plan high detailed and quality machine learning engineering solutions,",
    "user_instructions": {
        "Possible Questions you will face": "You will be asked to come up with a step by step plan to solve the kaggle competetion",
        "How to answer the user": 'Whenever you answer, always: 1. Write a "## Task Summary:" section in plain text consisting of 5-7 sentences distilling the task for you team members that are responsible for implementing the solution. 2. Write a "## Plan:" section in plain text consisting of detailed and high quality bullet points that will be used by the team members to implement the solution (7-10 bullet points). ',
        "Critical Instructions": "Do not give/write code solutions, coding is not your job, just consice summary and detailed plan",
    },
}

PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": "You are a Kaggle Grandmaster and great at implementing machine learning engineering code. Precisely follow the plan to implement the code that solves the kaggle competetion.",
    "user_instructions": {
        "What you will face": "You will be given a plan to implement the code that solves the kaggle competetion. Precisely follow the plan to implement the code.",
        "How to answer the user": 'Whenever you answer, always: answer in one section called "CODE:" containing exactly one fenced Python block: ```python implementing the plan',
    },
}

# --- System Prompt Getters ---
def get_agent_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_SYSTEM_PROMPT_DICT)

def get_planner_agent_plan_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT)

def get_planner_agent_code_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT)


# --- User Message Assemblers for Agent ---
def get_agent_draft_user_prompt(
    task_desc: str,
    journal_summary: str,
    competition_name: str,
    obfuscate: bool,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = "You are a Kaggle grandmaster. Your task is to develop a complete Python script to solve the described machine learning competition."
    if obfuscate:
        introduction = "You are an expert machine learning engineer. Your task is to develop a complete Python script to solve the described machine learning problem."

    # This structure matches the original _draft method's prompt_user_message
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Overall Task Description": task_desc,
        "Memory (Summary of Previous Attempts on this Task)": journal_summary,
        "Instructions": {
            "Response format": AGENT_RESPONSE_FORMAT_TEXT,
            "Solution sketch guideline": AGENT_DRAFT_SOLUTION_GUIDELINE_LIST,
            "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
            "Environment and Packages": get_competition_environment_text(competition_name)
        },
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

def get_agent_improve_user_prompt(
    task_desc: str,
    journal_summary: str,
    competition_name: str,
    parent_node_code: str,
) -> Dict[str, Any]:
    introduction = (
        "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
        "solution below and should improve it in order to further increase the (test time) performance. "
        "For this you should first outline a brief plan in natural language for how the solution can be improved and "
        "then implement this improvement in Python based on the provided previous solution. "
    )
    # This structure matches the original _improve method's prompt
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Task description": task_desc,
        "Memory": journal_summary,
        "Previous solution": { # Original had this as a top-level key, kept for consistency
            "Code": wrap_code(parent_node_code),
        },
        "Instructions": {
            "Response format": AGENT_RESPONSE_FORMAT_TEXT,
            "Solution improvement sketch guideline": AGENT_IMPROVE_SOLUTION_GUIDELINE_LIST,
            "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
            "Environment and Packages": get_competition_environment_text(competition_name) # Added for consistency
        },
    }
    return prompt_user_message

def get_agent_debug_user_prompt(
    task_desc: str,
    competition_name: str,
    parent_node_code: str,
    parent_node_term_out: str,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = (
        "You are a Kaggle grandmaster attending a competition. "
        "Your previous solution had a bug and/or did not produce a submission.csv, "
        "so based on the information below, you should revise it in order to fix this. "
        "Your response should be an implementation plan in natural language,"
        " followed by a single markdown code block which implements the bugfix/solution."
    )
    # This structure matches the original _debug method's prompt
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Task description": task_desc,
        "Previous (buggy) implementation": wrap_code(parent_node_code),
        "Execution output": wrap_code(parent_node_term_out, lang=""),
        "Instructions": {
            "Response format": AGENT_RESPONSE_FORMAT_TEXT,
            "Bugfix improvement sketch guideline": AGENT_DEBUG_SOLUTION_GUIDELINE_LIST,
            "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
            "Environment and Packages": get_competition_environment_text(competition_name) # Added for consistency
        },
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message


# --- User Message Assemblers for PlannerAgent ---

# For PlannerAgent's _draft stage (plan_query part)
def get_planner_agent_draft_plan_user_prompt(
    task_desc: str,
    journal_summary: str,
    competition_name: str,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    plan_introduction = f"Given the following task description for a machine learning competition named {competition_name}, develop a complete and detailed plan to solve it."
    prompt_user_message: Dict[str, Any] = {
        "Introduction": plan_introduction,
        "Overall Task Description": task_desc,
        "Memory (Summary of Previous Attempts on this Task)": journal_summary,
        "Instructions": {
            "Guidance on Summary": "The summary should be 5-7 sentences that describe the task in a nutshell, so that the team members can understand the task and the plan.",
            "Response format": PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT,
            "Instructions for generating the plan": [
                "Every step of the plan should be very detailed and explicit, point exactly how is the steps going to be solved. e.g. 'Use XGBoost to train a model with the following parameters: ...'",
                "The plan should be detailed in a step by step manner that is easy to follow.",
                "for this particular first solution, propose a relatively simple approach in terms of method used for solving the problem, without ensembling or hyper-parameter optimization, as we are using this as a first draft for future improvements.",
                "Take the Memory section into consideration when proposing the design.",
                "The solution plan should be detailed and high quality bullet points that are easy to follow.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to suggest any unzip for any files.",
            ],
            "Environment and Packages": get_competition_environment_text(competition_name)
        }
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

# For PlannerAgent's _draft stage (code_query part)
def get_planner_agent_draft_code_user_prompt(
    task_summary_from_planner: str, # Summary generated by the planner model
    plan_from_planner: str,         # Plan generated by the planner model
    journal_summary: str,
    competition_name: str,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    code_introduction = f"Given the following task description about a machine learning competition named {competition_name}, and the plan to solve it, develop a complete code to solve it."
    prompt_user_message: Dict[str, Any] = {
        "Introduction": code_introduction,
        "Overall Task Description": task_summary_from_planner,
        "Plan to implement": plan_from_planner,
        "Memory (Summary of Previous Attempts on this Task)": journal_summary,
        "Instructions": {
            "Environment and Packages": get_competition_environment_text(competition_name),
            "Solution code guideline": [
                "Strictly implement the code that implements the plan.",
                "Provide a single, complete Python script wrapped in a ```python code block.",
                "Include all necessary imports and load data from './input/' correctly.",
                "Write clear, concise comments explaining each part of the code.",
                "Ensure the code adheres to PEP8 style and is easy to read.",
                "Optimize performance without sacrificing clarity.",
                "Calculate and print the validation metric in the format: `Validation Metric: {metric_value}`.",
                "Save test predictions to './submission/submission.csv' exactly as required.",
                "The code should be between ```python fences",
                "only write code, do not write any other text",
            ],
            "Response format": PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT
        }
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

# For PlannerAgent's _improve stage (plan_query part)
def get_planner_agent_improve_plan_user_prompt(
    task_desc: str,
    parent_node_code: str,
    competition_name: str, # Added for environment if needed by planner
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    planner_introduction = (
        "You are a Kaggle grandmaster and a team leader. You are provided with a previously developed solution and "
        "should summarize the task, and outline your proposed improvement to further increase the (test time) performance. "
        "Then, outline a high quality and detailed step by step plan that your team members will use to implement this improvement."
    )
    prompt_user_message: Dict[str, Any] = {
        "Introduction": planner_introduction,
        "Overall Task Description": task_desc,
        "Previous solution": {"Code": wrap_code(parent_node_code)},
        "Instructions": {
            "Response format": PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT,
            "Solution improvement sketch guideline": [
                "You should provide a summary of the task description and the previous solution and then outline a high quality and detailed step by step plan in natural language for how the solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
            ],
            # "Environment and Packages": get_competition_environment_text(competition_name) # If planner model benefits
        }
    }
    if acfg_data_preview and data_preview_content: # If planner needs data context for improvement ideas
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

# For PlannerAgent's _improve stage (code_query part)
def get_planner_agent_improve_code_user_prompt(
    task_summary_from_planner: str,
    improvement_plan_from_planner: str,
    parent_node_code: str,
    journal_summary: str,
    competition_name: str,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    code_introduction = (
        "You are an expert machine learning engineer and a team member. You are provided with a previous solution, "
        "a summary of the task and previous solution, and a detailed plan for a single, atomic improvement. "
        "Your task is to implement this improvement."
    )
    prompt_user_message: Dict[str, Any] = {
        "Introduction": code_introduction,
        "Task description summary and previous solution": task_summary_from_planner,
        "Improvement plan": {"Plan": improvement_plan_from_planner},
        "Previous solution code": {"Code": wrap_code(parent_node_code)},
        "Memory": journal_summary,
        "Instructions": {
            "Environment and Packages": get_competition_environment_text(competition_name),
            "Response format": PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT,
            "code improvement guideline": [
                "You should precisely follow the plan for improvement and implement the code that implements the improvement.",
                "The final code should be a single code block, complete, and self-contained.",
                "The code should be well documented and easy to understand.",
                "Strictly follow the plan for improvement.",
                "Take the Memory section into consideration during implementation to avoid bugs.",
                "Code should be between ```python fences.",
                "Only write code; do not write any other text.",
            ],
            "additional guidelines": AGENT_IMPLEMENTATION_GUIDELINE_LIST # Reusing agent's guidelines
        }
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

# For PlannerAgent's _debug stage (plan_query part)
def get_planner_agent_debug_plan_user_prompt(
    task_desc: str,
    parent_node_code: str,
    parent_node_term_out: str,
    # competition_name: str, # If planner needs environment for debug plan
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    plan_introduction = (
        "You are a Kaggle grandmaster AND A TEAM LEADER. "
        "Your team's previous solution had a bug and/or did not produce a submission.csv. "
        "Based on the information below, you should first provide a detailed summary of the bugs "
        "and then outline a detailed step-by-step plan to fix them. This plan will be given to a team member to implement."
    )
    prompt_user_message: Dict[str, Any] = {
        "Introduction": plan_introduction,
        "Task description": task_desc,
        "Previous (buggy) implementation": wrap_code(parent_node_code),
        "Execution output": wrap_code(parent_node_term_out, lang=""),
        "Instructions": {
            "Response format": PLANNER_AGENT_DEBUG_RESPONSE_FORMAT_TEXT,
            # Guidelines are embedded in the response format and intro for planner debug plan
            # "Environment and Packages": get_competition_environment_text(competition_name) # If planner benefits
        }
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

# For PlannerAgent's _debug stage (code_query part)
def get_planner_agent_debug_code_user_prompt(
    bug_summary_from_planner: str,
    fix_plan_from_planner: str,
    parent_node_code: str,
    parent_node_term_out: str, # For coder's context
    competition_name: str,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    code_introduction = (
        "You are a Kaggle grandmaster AND A TEAM MEMBER. Your previous solution had bugs. "
        "You are provided with an analysis of the bugs, a detailed plan to fix them, the original buggy code, and its execution output. "
        "Your task is to implement the bugfix."
    )
    prompt_user_message: Dict[str, Any] = {
        "Introduction": code_introduction,
        "Problem Description and Analysis": bug_summary_from_planner,
        "Plan for fixing the bug": fix_plan_from_planner,
        "Previous (buggy) implementation": wrap_code(parent_node_code),
        "Execution output of buggy code": wrap_code(parent_node_term_out, lang=""),
        "Instructions": {
            "Environment and Packages": get_competition_environment_text(competition_name),
            "Response format": PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT,
            "Bugfix implementation guideline": [
                "Precisely follow the plan for fixing the bugs and implement the code that implements the fix.",
                "The final code should be a single code block, complete, and self-contained.",
            ],
            "additional guidelines": AGENT_IMPLEMENTATION_GUIDELINE_LIST # Reusing agent's guidelines
        }
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

