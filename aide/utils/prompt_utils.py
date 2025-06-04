# aide/utils/prompt_utils.py
import random
from typing import Any, Dict, List,Optional
from typing import Any, Dict, List,Optional
import copy # For deepcopying system prompts
from ..backend import FunctionSpec
from copy import deepcopy

# --- Helper for wrapping code (already in your codebase) ---
def wrap_code(code_str: str, lang: str = "python") -> str:
    if not code_str: # Handle None or empty string gracefully
        return f"```{lang}\n# No code provided.\n```" if lang else "```\n# No content provided.\n```"
    if lang:
        return f"```{lang}\n{code_str}\n```"
    return f"```\n{code_str}\n```"

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
                "type": "number",
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
            "metric",
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

    if competition_name in COMPETITION_METADATA:
        current_comp_data = COMPETITION_METADATA[competition_name]
        task_type = current_comp_data["Task Type"]
        task_type_lower = task_type.lower()
        suggested_pkgs = set(PACKAGE_CATEGORIES["common"]) | set(PACKAGE_CATEGORIES["optimization"])
        task_specific_guidance = ""

        if "image" in task_type_lower:
            suggested_pkgs.update(PACKAGE_CATEGORIES["image"])
            task_specific_guidance = "For this image-based task, libraries like `OpenCV/Pillow`, `torchvision`, and `timm` are highly relevant."
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
    "1. Deliver a complete solution, composed of a plan and a code implementations that successfully solves the kaggle competition and saves the submission.csv file ",
    "2. The plan should not be generic, it should be specific to the task and the data, and should be a single sentence for each step, specefically tailored to the task and the data",
    "3. the code should be complete, single-file Python script that successfully solves the kaggle competition and saves the submission.csv file ",
    "4. Implement the solution proposed in your plan.",
    "5. Calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
    "6. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",
    "7. The script must run without errors. Focus on correctness first.",
    "8. The code should be clean and easy to understand. It should be well-documented and well-structured."
]

AGENT_RESPONSE_FORMAT_TEXT: str = (
    "Format the response as follows: "
    "1) PLAN (plain text, no fences):\n as numbered list of steps, each step should be a bullet point, each step should be a single action that can be taken to solve the task"
    "followed by a single markdown code block (wrapped in ```python ... ```) which implements this solution and prints out the evaluation metric. "
    "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    "Your entire response MUST strictly follow this format:\n\n"
    "PLAN:\n" # No "1)"
    "<your step-by-step reasoning here, as detailed bullet points>\n\n" # Removed "plain text, no fences" as it's implied by not having backticks
    "---\n" # Separator
    "CODE:\n" # No "2)"
    "```python\n"
    "<your python code here, with '# Thought:' comments before logical blocks>\n"
    "```\n"
    "There should be NO text before 'PLAN:' and NO text after the final '```'."
)

AGENT_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": "You are a Kaggle Grandmaster. You can plan, implement, debug, and improve machine learning engineering code.",
    "user_instructions": {
        "Possible Questions you will face": "You will be asked to either come up with a plan and code to solve a Kaggle competition, debug existing code, or improve working code to get better results.",
        "How to answer the user": (
            'Whenever you answer, always: '
            '1. You strat by writing a "PLAN:" section in plain text with 7-10*highly detailed, step-by-step bullet points*. Each step should be actionable and explicit, explaining *how* it will be achieved. '
            'Example plan step: "1. Load \'train.csv\' and \'test.csv\' using pandas, then use train_test_split to split the data to 80%-20% training and validation sets."\n'
            '2. Then write a "CODE:" section containing exactly one fenced Python block: ```python. Within this code block, *before each major logical section of code*, include a comment explaining your immediate thought process, the specific purpose of that section, and how it relates to your PLAN step. '
            "Your entire response MUST strictly follow this format:\n\n"
            "PLAN:\n" # No "1)"
            "<your step-by-step reasoning here, as detailed bullet points>\n\n" # Removed "plain text, no fences" as it's implied by not having backticks
            "---\n" # Separator
            "CODE:\n" # No "2)"
            "```python\n"
            "<your python code here, with '# Thought:' comments before logical blocks>\n"
            "```\n"
            "There should be NO text before 'PLAN:' and NO text after the final '```'."

        ),
        "Critical Instruction": "Ensure your plan is explicit and your code is well-commented with your thought process as instructed."
    },
}

AGENT_draft_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": (
        "You are a Kaggle Grandmaster. Your task is to devise a clear, step-by-step PLAN and then write the corresponding Python CODE to solve machine learning competitions. "
        "Adhere strictly to the specified output format. The primary goal for this draft is a working, bug-free solution, so prioritize simplicity and correctness in your design."
    ),
    "user_instructions": {
        "Task Context / Possible Questions": "You will be provided with a description of a Kaggle competition and asked to generate a complete solution, which includes both a PLAN and the corresponding CODE.",
        
        "How to Answer (Output Structure, Plan Details, Code Details, Examples)": (
            "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator between the PLAN and CODE sections. There should be no text before 'PLAN:' and no text after the final ``` of the CODE block.\n\n"
            "**1. PLAN Section Requirements:**\n"
            "   Construct a \"PLAN:\" section. This plan must consist of 7-10 highly detailed, sequential bullet points. "
            "   Each point must describe a specific, actionable step required to solve the problem, including *what* to do and *how* it will be achieved (e.g., specific libraries, functions, or techniques to use). "
            "   This plan will directly guide your code implementation. Avoid overly generic steps and do NOT include steps for Exploratory Data Analysis (EDA). "
            "   Each bullet point in the PLAN must be self-contained, explaining not just *what* to do but also *how* it will be achieved, mentioning key libraries or specific functions if applicable, as if you are explaining it to someone who will implement it based solely on that plan step.\n\n"
            "   *Example plan steps demonstrating required detail for a complete simple solution (for a hypothetical customer churn prediction task):*\n"
            '   "1. **Data Loading**: Load `train.csv` and `test.csv` datasets into pandas DataFrames using `pd.read_csv()`. Store the `customerID` from the test set for later use in the submission file."\n'
            '   "2. **Target Variable Preparation**: Separate the target variable (e.g., `Churn`) from the features in the training DataFrame. If the target is categorical (e.g., Yes/No), encode it into numerical format (0/1) using `sklearn.preprocessing.LabelEncoder` or a simple map function."\n'
            '   "3. **Basic Feature Selection & Preprocessing - Numerical**: Identify numerical features. For simplicity in this draft, select a subset of obviously relevant numerical features (e.g., `tenure`, `MonthlyCharges`). Impute any missing values in these selected features using the median strategy with `sklearn.impute.SimpleImputer(strategy=\'median\')`. Fit the imputer on the training data and transform both train and test sets for these features."\n'
            '   "4. **Basic Feature Preprocessing - Categorical**: Identify categorical features. For simplicity, select a few key categorical features (e.g., `Contract`, `PaymentMethod`). Apply one-hot encoding using `pd.get_dummies()` to these features for both train and test sets. Ensure consistent columns by aligning them post-encoding, possibly by reindexing based on training set columns."\n'
            '   "5. **Combine Preprocessed Features**: Concatenate the preprocessed numerical and categorical features into final training (X_train_processed) and test (X_test_processed) feature sets using `pd.concat()`."\n'
            '   "6. **Data Splitting for Validation**: Split `X_train_processed` and the encoded target variable into training and validation subsets (e.g., 80% train, 20% validation) using `sklearn.model_selection.train_test_split`, setting a `random_state` for reproducibility and `stratify` by the target if it\'s a classification task."\n'
            '   "7. **Model Training**: Instantiate a simple classification model, for example, `sklearn.linear_model.LogisticRegression(random_state=42, solver=\'liblinear\')`. Train this model on the (scaled, if done) training subset (`X_train_fold`, `y_train_fold`)."\n'
            '   "8. **Validation and Metric Display**: Predict probabilities on the validation subset using `model.predict_proba()[:, 1]` (for the positive class). Calculate and print a relevant validation metric (e.g., ROC AUC using `sklearn.metrics.roc_auc_score`) using the format: `print(f\'Validation ROC AUC: {auc_score}\')`."\n'
            '   "9. **Test Set Prediction**: Predict probabilities on the fully preprocessed test set (`X_test_processed`) using `model.predict_proba()[:, 1]` to get the likelihood of churn for each test customer."\n'
            '   "10. **Submission File Generation**: Create a pandas DataFrame for the submission. It should contain the `customerID` column from the original test data and a `Churn` (or the target name specified by the competition) column with the predicted probabilities. Save this DataFrame to `./submission/submission.csv` using `submission_df.to_csv(path, index=False)`."\n\n'
            "**2. CODE Section Requirements:**\n"
            "   Follow the PLAN with a \"CODE:\" section, containing a single, complete Python script enclosed in ```python ... ```. "
            "   Crucially, *before every distinct logical block of code that corresponds to a step in your PLAN*, you MUST include a comment starting with \"# Thought:\". This comment should briefly state: "
            "   a) Your immediate thought process or strategy for implementing that part. "
            "   b) The specific purpose of the upcoming code block. "
            "   c) Which PLAN step number(s) it directly addresses. \n"
            "   *Example CODE format snippet:*\n"
            "   ```python\n"
            "   # Thought: Implementing PLAN step 1. Need to load the training data CSV. Pandas is the standard tool.\n"
            "   import pandas as pd\n"
            "   train_df = pd.read_csv(\"./input/train.csv\")\n\n"
            "   # Thought: Continuing PLAN step 1. Construct full image file paths.\n"
            "   import os\n"
            "   IMAGE_DIR = \"./input/train/\"\n"
            "   train_df[\"filepath\"] = train_df[\"id\"].apply(lambda x: os.path.join(IMAGE_DIR, x))\n"
            "   ```"
        ),
        
        "Critical Adherence / Final Instructions": (
            "Strict adherence to the detailed PLAN structure (as per the examples provided) and the '# Thought:' commenting convention in the CODE is mandatory. "
            "The primary objective for this draft is a working, bug-free solution. Therefore, the proposed solution should be simple in its overall design and ideas, focusing on correctness and the avoidance of BUGS. Do NOT include EDA."
            "You might receive a 'Memory' section summarizing previous attempts. Consider this information AND AVOID REPEATING past mistakes or unsuccessful approaches. Also, it is recommended that you design a different solution from the previous attempts. "

        )
    }
}


AGENT_improve_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": (
        "You are an analytical Kaggle Grandmaster, focused on iterative performance enhancement. "
        "Given a working Python solution for a machine learning competition, your task is to: "
        "1. Propose *one single, atomic, and well-justified* improvement to enhance its predictive performance (e.g., reduce log loss, increase AUC). "
        "2. Create a detailed PLAN explaining this specific improvement: what it is, *how* it will be implemented, and *why* it's expected to boost performance. "
        "3. Provide the *entire modified, runnable Python CODE* implementing only this single improvement. "
        "Adhere strictly to the specified output format."
    ),
    "user_instructions": {
        "Input Provided": "You will receive: the 'Task Description', the 'Previous (working) solution's CODE', and a 'Memory' of past attempts (if any).",
        
        "Output Format (Strict Adherence Required)": (
            "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator. No text before 'PLAN:' or after the final ``` of the CODE block.\n\n"
            
            "**1. PLAN Section Requirements:**\n"
            "   a. **Improvement Rationale (Brief Introduction - 2-3 sentences):** Briefly state the single improvement you are proposing and the core reason you believe it will enhance performance based on the previous solution and general ML principles.\n"
            "      *Example Rationale:* 'The previous solution used a simple CNN. To potentially capture more complex image features and improve AUC, I propose replacing it with a pre-trained ResNet18 model, fine-tuning its final layers.'\n\n"
            "   b. **Detailed Improvement Plan (Bulleted List - 3-7 detailed steps):** \n"
            "      - Outline the precise, actionable steps to implement *only this single improvement*. \n"
            "      - Each step must detail *what* changes will be made to the existing code, *where* these changes will occur (e.g., which functions/classes), and *how* (e.g., specific libraries, functions, parameter changes). \n"
            "      - Crucially, for each step, explain *why* this specific modification contributes to the overall proposed improvement and is expected to lead to better performance. \n"
            "      - If introducing new libraries (e.g., Albumentations for advanced augmentation), explicitly mention the import and the key components to be used.\n"
            "      *Example Detailed Improvement Plan Steps (for switching to ResNet18):*\n"
            "      '1. **Modify Imports**: Add `from torchvision import models` to import pre-trained models. Ensure `torch.nn as nn` is present.\n"
            "      2. **Replace Model Architecture**: In the `CactusClassifier` class (or equivalent), remove the existing `self.conv_layers` and `self.fc_layers`. Instantiate `models.resnet18(pretrained=True)` as the backbone. Explain that pre-trained weights capture general image features.\n"
            "      3. **Adapt Final Layer**: The ResNet18 `fc` layer outputs 1000 classes. Replace `model.fc` with a new `nn.Linear(model.fc.in_features, 1)` followed by `nn.Sigmoid()` for binary classification. This adapts the ResNet to the specific task.\n"
            "      4. **Adjust Image Preprocessing**: ResNet models are typically trained on images normalized with ImageNet statistics and often larger input sizes (e.g., 224x224, though 32x32 can still work but resizing might be an option). Update the `transforms.Compose` to include `transforms.Resize((desired_size, desired_size))` (e.g., 32 or 64 for this task) and `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`. This ensures input data matches ResNet's expectations.\n"
            "      5. **Adjust Learning Rate (Potentially Lower):** Pre-trained models often benefit from a smaller learning rate for fine-tuning. Consider reducing the LR in `optim.Adam` (e.g., to 1e-4 or 5e-4) to prevent distorting pre-trained weights too quickly.'\n\n"

            "**2. CODE Section Requirements:**\n"
            "   Follow the PLAN with a \"CODE:\" section, containing a single, *complete, and runnable* Python script enclosed in ```python ... ```. "
            "   This script should be the *entire* previous working script, with *only the modifications outlined in your Detailed Improvement Plan* to implement the single proposed enhancement.\n"
            "   *Before each modified or newly added logical block of code*, you MUST include a comment starting with \"# Improvement Thought:\". This comment should briefly state:\n"
            "   a) The specific part of the improvement being implemented.\n"
            "   b) How the code change relates to the corresponding 'Detailed Improvement Plan' step.\n"
            "   c) A concise thought on the change (e.g., 'Replacing custom CNN with ResNet18 for better feature extraction.').\n"
            "   Ensure all original necessary imports are preserved and any new ones required for the improvement are added. Do not remove unrelated working code.\n"
        ),
        
        "Critical Adherence / Final Instructions": (
            "Strict adherence to proposing a *single, atomic improvement* and detailing it in the PLAN is mandatory. The CODE section must contain the *entire runnable script* with minimal targeted changes for that one improvement. "
            "Clearly justify the improvement. Do NOT include EDA. Ensure all necessary imports are present."
        )
    }
}


## Last version abd best so far
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


AGENT_draft_SOLUTION_GUIDELINE_LIST: List[str] = [
    "This is an initial solution draft. Prioritize a simple, correct, and bug-free design. Avoid complex techniques like ensembling or extensive hyper-parameter optimization for this iteration.",
    "If a 'Memory' section is provided (summarizing previous attempts), consider its contents to avoid repeating past mistakes or unsuccessful approaches.",
    "The PLAN should consist of 7-10 bullet points. Each point must be explicit, detailed, and actionable, clearly stating *how* the step will be accomplished.", # Aligns with 7-10
    "The evaluation metric to be used is typically found in the 'Overall Task Description'. Ensure your code calculates and prints a validation metric, preferably this one.",
    "Do not include steps for Exploratory Data Analysis (EDA) in your plan or code.",
    "Assume all necessary data files are already prepared and located in the `./input/` directory. No unzipping or complex data discovery is needed.",
]


AGENT_improve_SOLUTION_GUIDELINE_LIST: List[str] = [
    "Propose *one single, atomic, and well-justified* improvement to the provided working solution. Do not suggest multiple changes at once.",
    "Your PLAN must start with a brief 'Improvement Rationale' (2-3 sentences) explaining your chosen improvement and why it should boost performance.",
    "Following the rationale, provide a 'Detailed Improvement Plan' as a bulleted list of 3-7 steps. Each step must be highly specific about *what* code to change, *where*, *how* (mentioning specific libraries/functions/parameters), and *why* this contributes to the improvement.",
    "The CODE section must implement *only* this single improvement, modifying the provided previous solution. It must be a complete, runnable script.",
    "Use '# Improvement Thought:' comments before each changed/new code block, linking it to your plan and explaining your reasoning for that specific code modification.",
    "Consider the 'Memory' section (if provided) to avoid repeating unsuccessful strategies or to build on previously identified good ideas.",
    "Do not suggest Exploratory Data Analysis (EDA). Focus on a direct code/model/feature enhancement."
]

AGENT_debug_SOLUTION_GUIDELINE_LIST: List[str] = [
    "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
    "Don't suggest to do EDA.",
    "Your PLAN should start with a 'Bug Analysis:' section. In this section, meticulously analyze the 'Execution output' and the 'Previous (buggy) implementation' line by line (or logical block by block) to identify the root cause of the bug. State concrete observations, e.g., 'Line X: FileNotFoundError because path was incorrect. This indicates the file path in the code is incorrect.'",
    "Following the 'Bug Analysis:', provide a 'Fix Plan:' with highly detailed, actionable steps to resolve *each identified bug*. Each step should explain *what* will be changed and *why*. For example, 'Fix Plan: 1. Update file path for train.csv to ./input/train.csv to match the correct directory structure, as indicated by the FileNotFoundError.'",
    "In your CODE, before each modified or new logical block, add a comment explaining the specific bug being addressed by that code, how the change fixes it, and your thought process. Example: # Bugfix: Handle division by zero. Previous code caused ZeroDivisionError on line Y. Added a check here to prevent it by replacing NaN with 0."
]



# --- System Prompt Getters ---
def get_agent_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_SYSTEM_PROMPT_DICT)

def get_agent_draft_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_draft_SYSTEM_PROMPT_DICT)

def get_agent_improve_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_improve_SYSTEM_PROMPT_DICT)

def get_agent_debug_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_debug_SYSTEM_PROMPT_DICT)

# --- User Message Assemblers for Agent ---
def get_agent_draft_user_prompt(
    task_desc: str,
    journal_summary: str,
    competition_name: str,
    obfuscate: bool,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = "You are a Kaggle grandmaster. Plan and develop a complete Python script to solve the described machine learning competition."
    if obfuscate:
        introduction = "You are an expert machine learning engineer. Your task is to develop a complete Python script to solve the described machine learning problem."
    # This structure matches the original _draft method's prompt_user_message
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Overall Task Description": task_desc,
        "Data Overview" :data_preview_content if acfg_data_preview and data_preview_content else "No detailed data overview provided; rely on file names in plan and task description.",
        "Memory": journal_summary,
        "Instructions": {
            "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
            "Environment and Packages": get_competition_environment_text(competition_name),
            "Response format": AGENT_RESPONSE_FORMAT_TEXT,
            "Solution sketch guideline": AGENT_draft_SOLUTION_GUIDELINE_LIST
            },
    }

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
            "Solution improvement sketch guideline": AGENT_improve_SOLUTION_GUIDELINE_LIST,
            "Solution improvement sketch guideline": AGENT_improve_SOLUTION_GUIDELINE_LIST,
            "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
            "Environment and Packages": get_competition_environment_text(competition_name) # Added for consistency
        },
    }
    return prompt_user_message

def get_agent_debug_user_prompt(
    task_desc: str,
    competition_name: str, # For Environment and Packages
    parent_node_code: str,
    parent_node_term_out: str,
    parent_node_feedback: str, # <-- NEW: Summary from o3-mini or your feedback model
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:
    introduction = (
        "You are a Kaggle grandmaster tasked with debugging a Python script. "
        "Your previous solution attempt resulted in an error or failed to produce the required `submission.csv`. "
        "Analyze the provided buggy code, its execution traceback, and an initial bug summary. "
        "Then, formulate a precise PLAN (Bug Analysis, then Fix Plan) and provide the fully corrected Python CODE."
    )
    
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Task Description": task_desc,
        "Previous (Buggy) Implementation": wrap_code(parent_node_code),
        "Execution Output (Traceback)": wrap_code(parent_node_term_out, lang=""), # Explicitly label as traceback
        "Initial Bug Summary (from analysis tool)": parent_node_feedback if parent_node_feedback else "No initial summary provided. Perform analysis based on traceback and code.",
        "Instructions": {
            "General Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST, # These are still good general rules
            "Environment and Packages": get_competition_environment_text(competition_name)
        },
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message

#################################################################################################################################################################
############################################################## --- Planner Agent --- #############################################################################
#################################################################################################################################################################

PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
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
            "   *Example Plan Steps (for a hypothetical house price prediction task - Tabular Regression, Evaluation: RMSLE):*\n"
            '   " - **1. Initial Setup & Data Loading**: \n'
            "     *WHAT:* Import core libraries and load the training (`train.csv`) and test (`test.csv`) datasets.\n"
            "     *HOW:* Use `import pandas as pd`, `import numpy as np`, `from sklearn.model_selection import train_test_split`. Load data using `train_df = pd.read_csv('./input/train.csv')` and `test_df = pd.read_csv('./input/test.csv')`. Store test IDs: `test_ids = test_df['Id']`.\n"
            "     *WHY:* To set up the environment and get the primary datasets into pandas DataFrames for manipulation, and preserve test IDs for submission.\n"
            '   " - **2. Target Variable Transformation & Feature Separation**: \n'
            "     *WHAT:* Apply a log transformation to the target variable `SalePrice` to handle skewness (common for price data and RMSLE metric) and separate features from the target.\n"
            "     *HOW:* Create `train_labels = np.log1p(train_df['SalePrice'])`. Drop `SalePrice` from `train_df` to create `train_features = train_df.drop(['SalePrice', 'Id'], axis=1)`. Prepare `test_features = test_df.drop('Id', axis=1)`.\n"
            "     *WHY:* Log transforming the target helps normalize its distribution for regression models and aligns with RMSLE. Separating features and target is standard practice.\n"
            '   " - **3. Align Train/Test Columns & Identify Feature Types**: \n'
            "     *WHAT:* Ensure training and test feature sets have the same columns in the same order, and identify numerical and categorical features.\n"
            "     *HOW:* Get `all_features = pd.concat((train_features, test_features)).reset_index(drop=True)`. Then, re-align: `train_features = all_features.iloc[:len(train_labels)].copy()`, `test_features = all_features.iloc[len(train_labels):].copy()`. Identify `numeric_cols = train_features.select_dtypes(include=np.number).columns` and `categorical_cols = train_features.select_dtypes(include='object').columns`.\n"
            "     *WHY:* Consistent feature sets are crucial. Identifying feature types guides subsequent preprocessing.\n"
            '   " - **4. Preprocessing - Numerical Features**: \n'
            "     *WHAT:* Impute missing values in numerical features and then scale them.\n"
            "     *HOW:* Use `from sklearn.impute import SimpleImputer`. Create `imputer_num = SimpleImputer(strategy='median')`. Fit on `train_features[numeric_cols]`: `imputer_num.fit(train_features[numeric_cols])`. Transform both: `train_features[numeric_cols] = imputer_num.transform(train_features[numeric_cols])`, `test_features[numeric_cols] = imputer_num.transform(test_features[numeric_cols])`. Then, use `from sklearn.preprocessing import StandardScaler`. Create `scaler_num = StandardScaler()`. Fit on `train_features[numeric_cols]`, then transform both.\n"
            "     *WHY:* Imputation handles missing data. Scaling helps models that are sensitive to feature magnitudes (e.g., linear models, SVMs, NNs).\n"
            '   " - **5. Preprocessing - Categorical Features**: \n'
            "     *WHAT:* Impute missing values in categorical features and then apply one-hot encoding.\n"
            "     *HOW:* Use `SimpleImputer(strategy='most_frequent')` for categorical columns, fitting on train and transforming both. Then, use `pd.get_dummies(data, columns=categorical_cols, dummy_na=False)` on concatenated train/test categorical features, then split back. Ensure alignment by `train_features_encoded.align(test_features_encoded, join='inner', axis=1)` before concatenating with numerical.\n"
            "     *WHY:* Imputation handles missing categories. One-hot encoding converts categories to a machine-understandable numerical format.\n"
            '   " - **6. Combine Features & Split for Validation**: \n'
            "     *WHAT:* Combine processed numerical and categorical features. Split combined training features for local validation.\n"
            "     *HOW:* Use `pd.concat([train_features_numeric_processed, train_features_categorical_processed], axis=1)` to create `X_train_final`. Do similarly for `X_test_final`. Then, `X_train, X_val, y_train, y_val = train_test_split(X_train_final, train_labels, test_size=0.2, random_state=42)`.\n"
            "     *WHY:* To create the final feature matrices for model training and evaluation.\n"
            '   " - **7. Model Training**: \n'
            "     *WHAT:* Instantiate and train a simple regression model.\n"
            "     *HOW:* Use `from sklearn.ensemble import RandomForestRegressor`. Create `model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`. Train using `model.fit(X_train, y_train)`.\n"
            "     *WHY:* RandomForest is a robust baseline for tabular regression.\n"
            '   " - **8. Validation Metric Calculation & Display**: \n'
            "     *WHAT:* Predict on the validation set and calculate RMSLE.\n"
            "     *HOW:* Make predictions: `val_preds_log = model.predict(X_val)`. Since target was log-transformed, inverse transform: `val_preds = np.expm1(val_preds_log)`. Calculate RMSLE: `rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_val), val_preds))`. Print: `print(f'Validation RMSLE: {rmsle_score:.5f}')`.\n"
            "     *WHY:* To assess model performance locally using the competition metric.\n"
            '   " - **9. Test Set Prediction**: \n'
            "     *WHAT:* Generate predictions on the processed test set.\n"
            "     *HOW:* `test_preds_log = model.predict(X_test_final)`. Inverse transform: `test_preds = np.expm1(test_preds_log)`.\n"
            "     *WHY:* To get predictions for the submission.\n"
            '   " - **10. Submission File Generation**: \n'
            "     *WHAT:* Create and save the submission CSV file.\n"
            "     *HOW:* Create DataFrame: `submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_preds})`. Save: `submission.to_csv('./submission/submission.csv', index=False)`.\n"
            "     *WHY:* To produce the final output in the format required by the competition.\n"
        ),
        "Critical Reminder": "Your role is exclusively planning and summarizing. The Coder agent relies ENTIRELY on the clarity, explicitness (especially the 'HOW' for each step, including function/library/variable specifics), and logical correctness of your 'Task Summary' and 'Plan'. DO NOT generate any Python code."
    }
}

# --- Static Prompt Components (from PlannerAgent) ---
PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT: str = (
    "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
    "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Task Summary: and natural language text (plan) under ## Plan: "
    "explicitly,structure your answer exactly like this: "
    "\n\n---\n"
    "## Task Summary: (plain text, no fences):\n"
    "<your step‑by‑step reasoning abd summary of the task here>\n\n"
    "## Plan: (plain text, no fences):\n"
    "<your step‑by‑step reasoning and plan steps here>\n\n"
)

PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": (
        "You are an expert Kaggle Grandmaster Python Coder. You will be given a 'Task Summary' and a detailed 'Plan' (created by a Technical Lead). "
        "Your sole responsibility is to write a single, complete, runnable Python script that precisely implements this Plan to solve the described Kaggle competition. "
        "Pay meticulous attention to detail and the specified commenting conventions."
    ),
    "user_instructions": {
        "Input Context": "You will receive: \n1. 'Task Summary': A brief overview of the competition.\n2. 'Plan to Implement': A detailed, step-by-step plan. This is your primary guide for coding.\n3. 'Memory (optional)': Summary of previous attempts on this task.\n4. 'Environment and Packages': Information about available tools.",
        
        "Core Coding Task": "Translate the *entire* 'Plan to Implement' into a single, coherent Python script.",
        
        "Mandatory Commenting Convention ('# Thought:' Comments)": (
            "Crucially, *before every distinct logical block of Python code that directly implements one or more steps from the 'Plan to Implement'*, you MUST include a comment starting with '# Thought:'. This comment must briefly state:\n"
            "   a) Your immediate thought process or strategy for implementing that specific part of the plan.\n"
            "   b) The specific purpose of the upcoming code block.\n"
            "   c) Which PLAN step number(s) from the 'Plan to Implement' it directly addresses.\n"
            "*Example # Thought: comment (if Plan step 3 was 'Scale numerical features using StandardScaler'):*\n"
            "```python\n"
            "# Thought: Implementing PLAN step 3. The plan specifies using StandardScaler for numerical features. I need to fit it on the training data and then transform both train and validation/test sets.\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train_numerical)\n"
            "X_val_scaled = scaler.transform(X_val_numerical) # Assuming X_val_numerical exists\n"
            "```"
        ),

        "Output Requirement": (
            "Your entire response MUST consist of ONLY a single Python code block, enclosed in ```python ... ```. "
            "Do NOT include any text, explanations, or pleasantries before or after this code block. "
            "The script must be complete, including all necessary imports."
        ),

        "General Coding Guidelines for this Draft": [
            "Implement the plan as simply and directly as possible to ensure a correct, bug-free first draft.",
            "Ensure all necessary libraries mentioned in the plan or commonly used for the task (e.g., pandas, numpy, sklearn, torch, cv2, PIL) are imported at the beginning of the script.",
            "Load data from the `./input/` directory as specified or implied by the plan.",
            "Calculate and print a validation metric (e.g., as specified in the plan or task description) using a clear format like `print(f'Validation Metric: {metric_value}')`.",
            "CRITICAL: Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv` in the format required by the competition.",
            "Focus on correctness. The script must run without errors."
        ]
    }
}

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

PLANNER_AGENT_PLAN_generic_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": "You are a Kaggle Grandmaster and a team leader. you can plan high detailed and quality machine learning engineering solutions,",
    "user_instructions": {
        "Possible Questions you will face": "You will be asked to come up with a step by step plan to solve the kaggle competetion",
        "How to answer the user": 'Whenever you answer, always: 1. Write a "## Task Summary:" section in plain text consisting of 5-7 sentences distilling the task for you team members that are responsible for implementing the solution. 2. Write a "## Plan:" section in plain text consisting of detailed and high quality bullet points that will be used by the team members to implement the solution (7-10 bullet points). ',
        "Critical Instructions": "Do not give/write code solutions, coding is not your job, just consice summary and detailed plan",
    },
}


def get_planner_agent_plan_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT)

def get_planner_agent_code_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT)

def get_planner_agent_plan_generic_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(PLANNER_AGENT_PLAN_generic_SYSTEM_PROMPT_DICT)
# For PlannerAgent's _draft stage (plan_query part)
def get_planner_agent_draft_plan_user_prompt(
    task_desc: str,
    journal_summary: str,
    competition_name: str,
    acfg_data_preview: bool,
    data_preview_content: str = None
) -> Dict[str, Any]:

    data_overview = " "
    if acfg_data_preview and data_preview_content:
        data_overview = data_preview_content

    
    plan_introduction = f"Given the following task description for a machine learning competition named {competition_name}, develop a complete and detailed plan to solve it."
    prompt_user_message: Dict[str, Any] = {
        "Introduction": plan_introduction,
        "Overall Task Description": task_desc,
        "Data Overview": data_preview_content if acfg_data_preview and data_preview_content else "No detailed data overview provided; rely on file names in plan and task description.",
        "Environment and Packages": get_competition_environment_text(competition_name),
        "Memory": journal_summary,
        "Instructions": {
            "Guidance on Summary": "The summary should be 5-7 sentences that describe the task in a nutshell, so that the team members can understand the task and the plan.",

            "Instructions for generating the plan": [
                "Every step of the plan should be very detailed and explicit, point exactly how is the steps going to be solved. e.g. 'Use XGBoost to train a model with the following parameters: ...'",
                "your aim in this plan is to create a first draft solution that is correct and bug free",
                "for this particular first solution, propose a relatively simple approach in terms of method used for solving the problem, without ensembling or hyper-parameter optimization, as we are using this as a first draft for future improvements.",
                "Take the Memory section into consideration when proposing the design.",
                "The data is already prepared and available in the `./input` directory. There is no need to suggest any unzip for any files.",
            ],
        }
    }


    return prompt_user_message

def get_planner_agent_draft_code_user_prompt(
    task_summary_from_planner: str, # Summary generated by the planner model
    plan_from_planner: str,         # Plan generated by the planner model
    journal_summary: str,           # This is the "Memory"
    competition_name: str,
    acfg_data_preview: bool,        # From agent config
    data_preview_content: str = None # Actual preview content
) -> Dict[str, Any]:
    
    introduction = (
        f"You are the Coder implementing a solution for the '{competition_name}' Kaggle competition. "
        "You have been provided with a Task Summary and a detailed Plan by your Technical Lead. "
        "Your task is to write the complete Python script based *strictly* on this plan."
    )
    
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Context Provided by Technical Lead": { # Grouping planner's output
            "Task Summary": task_summary_from_planner if task_summary_from_planner else "No task summary was provided by the planner.",
            "Plan to Implement": plan_from_planner if plan_from_planner else "CRITICAL ERROR: No plan was provided by the planner. Cannot generate code."
        },
        "Data Overview": data_preview_content if acfg_data_preview and data_preview_content else "No detailed data overview provided; rely on file names in plan and task description.",
        "Environment and Packages": get_competition_environment_text(competition_name),
        "Memory": journal_summary if journal_summary else "No previous attempts on record for this specific task.",
        "Key Instructions for Your Code": {
            "Primary Goal": "Generate a single, complete, runnable Python script that meticulously follows the 'Plan to Implement'.",
            "Commenting": "MANDATORY: Use '# Thought:' comments before each code block implementing a plan step, as detailed in your system instructions.",
            "Output Format": "Your response must be *only* the Python code block. No extra text.",
            "Core Requirements": [ # These are from your AGENT_IMPLEMENTATION_GUIDELINE_LIST, slightly adapted
                "Include all necessary imports at the beginning.",
                "Load data from './input/' as specified in the plan.",
                "Implement the solution proposed in the 'Plan to Implement'.",
                "Calculate and print the evaluation metric on a validation set (e.g., `print(f'Validation Metric: {metric_value}')`).",
                "CRITICAL: Generate test predictions and save them EXACTLY to `./submission/submission.csv` in the required format.",
                "Prioritize correctness and ensure the script runs without errors for this first draft."
            ],
        }
    }
    # The PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT (```python ... ```) is now heavily reinforced by the system prompt.
    return prompt_user_message

def get_planner_agent_draft_code_user_prompt2(
    task_summary_from_planner: str, # Summary generated by the planner model
    plan_from_planner: str,         # Plan generated by the planner model
    journal_summary: str,           # This is the "Memory"
    competition_name: str,
    acfg_data_preview: bool,        # From agent config
    data_preview_content: str = None # Actual preview content
) -> Dict[str, Any]:
    
    introduction = (
        f"You are the Coder implementing a solution for the '{competition_name}' Kaggle competition. "
        "You have been provided with a Task Summary and a detailed Plan by your Technical Lead. "
        "Your task is to write the complete Python script based *strictly* on this plan."
    )
    
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Context Provided by Technical Lead": { # Grouping planner's output
            "Task Summary": task_summary_from_planner if task_summary_from_planner else "No task summary was provided by the planner.",
            "Plan to Implement": plan_from_planner if plan_from_planner else "CRITICAL ERROR: No plan was provided by the planner. Cannot generate code."
        },
        "Data Overview": data_preview_content if acfg_data_preview and data_preview_content else "No detailed data overview provided; rely on file names in plan and task description.",
        "Environment and Packages": get_competition_environment_text(competition_name),
        "Memory": journal_summary if journal_summary else "No previous attempts on record for this specific task.",
        "Key Instructions for Your Code": {
            "Primary Goal": "Generate a single, complete, runnable Python script that meticulously follows the 'Plan to Implement'.",
            "Commenting": "MANDATORY: Use '# Thought:' comments before each code block implementing a plan step, as detailed in your system instructions.",
            "Output Format": "Your response must be *only* the Python code block. No extra text.",
            "Core Requirements": [ # These are from your AGENT_IMPLEMENTATION_GUIDELINE_LIST, slightly adapted
                "Include all necessary imports at the beginning.",
                "Load data from './input/' as specified in the plan.",
                "Implement the solution proposed in the 'Plan to Implement'.",
                "Calculate and print the evaluation metric on a validation set (e.g., `print(f'Validation Metric: {metric_value}')`).",
                "CRITICAL: Generate test predictions and save them EXACTLY to `./submission/submission.csv` in the required format.",
                "Prioritize correctness and ensure the script runs without errors for this first draft."
            ],
        }
    }
    # The PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT (```python ... ```) is now heavily reinforced by the system prompt.
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
    parent_node_feedback: str, # For coder's context
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

#################################################################################################################################################################
############################################################## --- prompt chaining --- #############################################################################
#################################################################################################################################################################

_BASE_CODER_CHAIN_SYSTEM_MESSAGE1 = (
    "You will be given a 'Task Summary', the 'Full Master Plan' (generated by a Technical Lead), the 'Python Code Generated So Far' in previous segments, and specific instructions for the current coding segment. "
    " **Your Output MUST be ONLY a Python code block for the current segment.**\n"
    "Ensure it integrates seamlessly with the existing code and precisely follows the Master Plan.\n"
    "You MUST include '# Thought:' comments before logical code blocks, explaining your reasoning and linking to the Master Plan steps you are implementing for this segment.\n"
    "because this is only a segment of the full solution, you should be aware of the libraries, variables, and functions that are already defined in the previous segments, and not use something that is not imported or defined in the previous segments."
    "if you need to use a library that is not imported in the previous segments, you should import it in the current segment."\
    " - Do NOT include any conversational text, explanations, or self-corrections outside the '# Thought:' comments and the ```python ... ``` code block. Your entire response for this segment is the code block itself."
)


_BASE_CODER_CHAIN_SYSTEM_MESSAGE = (
    "You are an expert Python Coder specializing in implementing specific segments of a larger machine learning solution for Kaggle competitions. "
    "Your **sole task** is to generate a Python code snippet for a specific segment of a larger program. your snippet must be correct, runnable, and integrate seamlessly with the existing code. without inducing errors\n"
    "You will receive context: 'Task Summary', 'Full Master Plan', 'Python Code Generated So Far', and 'Your Current Coding Segment' instructions.\n"
     "because this is only a segment of the full solution, you should be aware of the libraries, variables, and functions that are already defined in the previous segments."
     "focus on CORRECTNESS,"
    "**CRITICAL OUTPUT REQUIREMENTS:**\n"
    "1. Your response MUST start *immediately* with ```python and end *immediately* with ```. NO TEXT BEFORE OR AFTER THE CODE BLOCK.\n"
    "2. Inside the code block, *before each distinct logical code unit* that implements a part of the plan for THIS SEGMENT, include a *concise* comment starting with '# Thought:'. This comment should explain: a) Your immediate coding strategy. b) The purpose of the upcoming code. c) The Master Plan step(s) it addresses for this segment.\n"
    "3. The code you generate for this segment must be self-contained for its purpose but integrate with 'Python Code Generated So Far'. If new imports are needed *for this segment's logic only* and were not in prior code, include them at the start of *your* code block for this segment.\n"
    "4. Implement *only* the functionalities specified for the current segment. Do NOT include code for other segments. just make it simple, implement the required functionality without any extra complexity. and without inducing bugs\n"
    "Violation of this format will result in failure. Focus on directly generating the required code snippet."
)
CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SETUP: Dict[str, Any] = {
    "SYSTEM": _BASE_CODER_CHAIN_SYSTEM_MESSAGE,
    "user_instructions": { # Meta-instructions for this system prompt's design
        "Current Segment Objective": "Initial script setup: all major anticipated library imports, global configurations (e.g., random seeds using a `set_seed` function), and defining the primary PyTorch `DEVICE`.",
       "Context for This Segment": "Refer to the 'Full Master Plan' and 'Task Summary' to anticipate necessary libraries (e.g., `pandas`, `numpy`, `sklearn`, `torch`, `PIL`, `cv2`, `timm`, `transformers` based on task type).",
        "Output Code Block Content": "A Python code block containing only imports, a `set_seed()` function definition and its call, and `DEVICE` assignment. No data loading or other functional code yet."
    }
}

CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_DATA_LOADING: Dict[str, Any] = {
    "SYSTEM": _BASE_CODER_CHAIN_SYSTEM_MESSAGE,
    "user_instructions": {
        "Current Segment Focus": "Loading all primary data files (e.g., training data, test data, sample submission for IDs) into appropriate data structures (typically pandas DataFrames) and defining essential file/directory path constants.",
        "Context Awareness": "Use file names and paths *as specified in the 'Full Master Plan'*. The plan should detail which files to load (e.g., `train_features.csv`, `test_images/`, `submission_format.csv`). "
        "The data is in `./input/` . Utilize imports from 'Python Code Generated So Far'.",
        "Output Requirement": "A Python code block for defining path constants (e.g., `INPUT_DIR`, `TRAIN_DATA_PATH`) using `os.path.join` or `pathlib.Path`, and loading data into variables like `train_df`, `test_df`. Define these variables clearly for subsequent stages."
    }
}

CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_PREPROCESSING: Dict[str, Any] = {
    "SYSTEM": _BASE_CODER_CHAIN_SYSTEM_MESSAGE,
    "user_instructions": {
        "Current Segment Focus": "All data preprocessing, feature engineering, data splitting, and preparation of data loaders or generators. This includes defining custom `Dataset` classes (e.g., for PyTorch) and data augmentation/transformation pipelines.",
        "Context Awareness": "Work with the data structures (e.g., `train_df`, `test_df`) created in the 'Python Code Generated So Far'. Follow the 'Full Master Plan' for specific preprocessing steps (missing value imputation, encoding, scaling), feature creation, how to split data (e.g., `train_test_split` from `sklearn.model_selection`), and how to set up `Dataset`s and `DataLoader`s if using PyTorch/TensorFlow.",
        "Output Requirement": "A Python code block containing all preprocessing logic. This may include function definitions (like a `Dataset` class or transformation functions) and their application. Ensure final processed data variables (e.g., `train_loader`, `val_loader`, `X_test_processed`) are clearly defined."
    }
}

CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_MODELING: Dict[str, Any] = {
    "SYSTEM": _BASE_CODER_CHAIN_SYSTEM_MESSAGE,
    "user_instructions": {
        "Current Segment Focus": "Defining and instantiating the machine learning model architecture.",
        "Context Awareness": "Follow the 'Full Master Plan' for the choice of model (e.g., CNN, ResNet , RandomForest, custom `nn.Module`). Utilize imports from 'Python Code Generated So Far'. The model should be compatible with the data prepared in previous segments.",
        "Output Requirement": "A Python code block that defines the model class (if custom) and/or instantiates the model (e.g., `model = timm.create_model(...)` or `model = MyCustomCNN()`). Ensure the model is assigned to a variable (e.g., `model`) and moved to the `DEVICE` defined in the setup segment."
    }
}

CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_TRAINING: Dict[str, Any] = {
    "SYSTEM": _BASE_CODER_CHAIN_SYSTEM_MESSAGE,
    "user_instructions": {
        "Current Segment Focus": "Setting up and executing the model training loop, including loss function, optimizer, and validation within/after epochs.",
        "Context Awareness": "Use the `model`, data loaders (`train_loader`, `val_loader`), and `DEVICE` from 'Python Code Generated So Far'. Follow the 'Full Master Plan' for loss function (e.g., `nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss`), optimizer (e.g., `torch.optim.Adam`), learning rate, number of epochs, and how validation should be performed.",
        "Output Requirement": "A Python code block implementing the training loop. This includes defining the criterion and optimizer, iterating through epochs and batches, performing forward/backward passes, and optimizer steps. Calculate and print the specified validation metric (e.g., `print(f'Validation Metric (AUC): {val_auc:.4f}')`) during or after training. The trained `model` object should be updated in place. If the plan specifies saving the best model, implement that logic."
    }
}

CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SUBMISSION: Dict[str, Any] = {
    "SYSTEM": _BASE_CODER_CHAIN_SYSTEM_MESSAGE,
    "user_instructions": {
        "Current Segment Focus": "Generating predictions on the test set using the trained model and creating the `submission.csv` file in the precise format required.",
        "Context Awareness": "Use the trained `model`, test data loader (`test_loader` or `X_test_processed`), and test IDs (e.g., from `test_df`) from 'Python Code Generated So Far'. Follow the 'Full Master Plan' and the competition's submission file format instructions for creating the output.",
        "Output Requirement": "A Python code block that loads the best model (if saved separately), sets it to evaluation mode, iterates through the test data, generates predictions, formats these predictions into a pandas DataFrame (typically with 'id' and target columns), "
        " **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",

    }
}



def get_coder_chain_system_prompt(segment_name: str) -> Dict[str, Any]:
    if segment_name == "Setup & Imports":
        return copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SETUP)
    elif segment_name == "Data Loading":
        return copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_DATA_LOADING)
    elif segment_name == "Data Preprocessing":
        return copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_PREPROCESSING)
    elif segment_name == "Modeling":
        return copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_MODELING)
    elif segment_name == "Training & Validation":
        return copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_TRAINING)
    elif segment_name == "Prediction & Submission":
        return copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SUBMISSION)
    else:
        raise ValueError(f"Unknown coder chain segment name: {segment_name}")

def _get_base_coder_chain_user_prompt_args(
    task_summary: str,
    master_plan_text: str,
    current_code_so_far: str,
    competition_name: str,
    data_preview_content: Optional[str]
) -> Dict[str, Any]:
    return {
        "Guidance from Technical Lead": {
            "Task Summary": task_summary or "No task summary was provided by the planner.",
            "Full Master Plan": master_plan_text or "CRITICAL ERROR: No master plan provided by planner."
        },
        "Python Code Generated So Far": wrap_code(current_code_so_far if current_code_so_far.strip() else "# This is the first code segment to be generated."),
        "Environment and Packages": get_competition_environment_text(competition_name),
        "Data Overview": data_preview_content if data_preview_content else "Refer to Master Plan and Task Summary for data details."
    }

def get_coder_chain_user_prompt_segment_setup(
    task_summary: str, master_plan_text: str, current_code_so_far: str, competition_name: str, data_preview_content: Optional[str]
) -> Dict[str, Any]:
    args = _get_base_coder_chain_user_prompt_args(task_summary, master_plan_text, "", competition_name, data_preview_content)
    args["Your Current Coding Segment: Initial Setup & Imports"] = (
        "Based on the 'Task Summary' and 'Full Master Plan' (anticipating needs for the entire script), "
        "write the Python code block *only* for the initial setup. This MUST include:\n"
        "1. All standard and task-specific library imports (e.g., `os`, `pandas`, `numpy`, `sklearn.model_selection`, `torch`, `torch.nn`, `PIL`, `cv2`, `timm`, `albumentations`, `transformers` as appropriate based on the overall plan and task type).\n"
        "2. A function `def set_seed(seed_value: int): ...` that sets seeds for `random`, `numpy`, and `torch` for reproducibility, and a call to this function (e.g., `set_seed(42)`).\n"
        "3. Definition of the PyTorch `DEVICE` variable (e.g., `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).\n"
        "Do NOT include any data loading or functional code beyond this setup. "
        "Remember your output must ONLY be the Python code block" # Added reminder
        "Remember to include '# Thought:' comments explaining your choices for imports or setup, linking to general good practice or broad implications from the Master Plan."
    )
    return args

def get_coder_chain_user_prompt_segment_data_loading(
    task_summary: str, master_plan_text: str, current_code_so_far: str, competition_name: str, data_preview_content: Optional[str]
) -> Dict[str, Any]:
    args = _get_base_coder_chain_user_prompt_args(task_summary, master_plan_text, current_code_so_far, competition_name, data_preview_content)
    args["Your Current Coding Segment: Data Loading & Path Definitions"] = (
        "Based on the 'Full Master Plan' (focus on steps related to data input and file locations) and using variables/imports from 'Python Code Generated So Far', "
        "write the Python code block *only* for:\n"
        "1. Defining global string constants for all necessary base directories (e.g., `INPUT_DIR = './input'`, `TRAIN_IMG_DIR = os.path.join(INPUT_DIR, 'train/')` if it's an image task and the plan implies it) and full paths to primary data files (e.g., `TRAIN_DATA_PATH = os.path.join(INPUT_DIR, 'train_data.csv')`). Use `os.path.join` or `pathlib.Path` for robust path construction. The exact file names (`train_data.csv`, etc.) MUST come from the Master Plan.\n"
        "2. Loading the primary data files (e.g., training data, test data or sample submission for test IDs) specified in the Master Plan into pandas DataFrames. Name these DataFrames clearly (e.g., `train_df`, `test_df`, `submission_df_ids`).\n"
        "Remember your output must ONLY be the Python code block with " # Added reminder
        "Remember to include '# Thought:' comments explaining your logic and linking to the relevant Master Plan steps you are addressing for this segment."
    )
    return args

def get_coder_chain_user_prompt_segment_preprocessing(
    task_summary: str, master_plan_text: str, current_code_so_far: str, competition_name: str, data_preview_content: Optional[str]
) -> Dict[str, Any]:
    args = _get_base_coder_chain_user_prompt_args(task_summary, master_plan_text, current_code_so_far, competition_name, data_preview_content)
    args["Your Current Coding Segment: Data Preprocessing, Feature Engineering, Splitting, Datasets/Loaders"] = (
        "Based on the 'Full Master Plan' and using variables/DataFrames (e.g., `train_df`, `test_df`) defined in 'Python Code Generated So Far', "
        "write the Python code block *only* for all data preprocessing and preparation steps. This includes (as specified in the plan):\n"
        "1. Handling missing values (e.g., using `SimpleImputer` or `fillna`).\n"
        "2. Encoding categorical features (e.g., `LabelEncoder`, `OneHotEncoder`, or `pd.get_dummies`).\n"
        "3. Scaling/normalizing numerical features (e.g., `StandardScaler`, `MinMaxScaler`).\n"
        "4. Any feature engineering steps outlined in the plan.\n"
        "5. Defining custom `Dataset` classes (e.g., for PyTorch, ensuring `__init__`, `__len__`, `__getitem__` are correctly implemented to handle data loading like `PIL.Image.open` or `cv2.imread` for images, and applying transformations) or data generators.\n"
        "6. Defining data augmentation/transformation pipelines (e.g., using `albumentations.Compose` or `torchvision.transforms.Compose`).\n"
        "7. Splitting the training data into training and validation sets (e.g., using `sklearn.model_selection.train_test_split`, ensuring to use `random_state` and `stratify` if appropriate as per the plan).\n"
        "8. Instantiating `DataLoader` objects for train, validation, and test sets if using PyTorch/TensorFlow, specifying `batch_size`, `shuffle`, `num_workers` as per the plan or sensible defaults.\n"
        "Ensure all newly created key variables (e.g., `X_train_processed`, `y_train_encoded`, `train_loader`, `val_loader`, `test_loader`, fitted scalers/encoders) are clearly defined for subsequent stages. "
        "Include '# Thought:' comments linking to Master Plan steps."
    )
    return args

def get_coder_chain_user_prompt_segment_modeling(
    task_summary: str, master_plan_text: str, current_code_so_far: str, competition_name: str, data_preview_content: Optional[str]
) -> Dict[str, Any]:
    args = _get_base_coder_chain_user_prompt_args(task_summary, master_plan_text, current_code_so_far, competition_name, data_preview_content)
    args["Your Current Coding Segment: Model Architecture Definition & Instantiation"] = (
        "Based on the 'Full Master Plan' (focus on model choice and architecture details) and using imports from 'Python Code Generated So Far', "
        "write the Python code block *only* for defining and instantiating the machine learning model. This includes:\n"
        "1. Defining the model class if it's a custom architecture (e.g., a PyTorch `nn.Module`).\n"
        "2. Instantiating the model (e.g., `model = MyCustomCNN()` or `model = timm.create_model('resnet18', pretrained=True, num_classes=... )` or `model = RandomForestClassifier(...)`). "
        "Ensure parameters like `pretrained`, `num_classes`, or classifier hyperparameters are set according to the plan.\n"
        "3. Moving the instantiated model to the `DEVICE` variable (which should have been defined in the setup segment, e.g., `model.to(DEVICE)` if using PyTorch).\n"
        "Assign the instantiated model to a variable named `model`. Include '# Thought:' comments."
    )
    return args

def get_coder_chain_user_prompt_segment_training(
    task_summary: str, master_plan_text: str, current_code_so_far: str, competition_name: str, data_preview_content: Optional[str]
) -> Dict[str, Any]:
    args = _get_base_coder_chain_user_prompt_args(task_summary, master_plan_text, current_code_so_far, competition_name, data_preview_content)
    args["Your Current Coding Segment: Training Setup & Loop"] = (
        "Based on the 'Full Master Plan' and using the `model`, data loaders (e.g., `train_loader`, `val_loader`), and `DEVICE` from 'Python Code Generated So Far', "
        "write the Python code block *only* for setting up and running the training process. This includes:\n"
        "1. Defining and instantiating the loss function (criterion, e.g., `nn.BCEWithLogitsLoss()`, `nn.CrossEntropyLoss()`).\n"
        "2. Defining and instantiating the optimizer (e.g., `torch.optim.Adam(model.parameters(), lr=...)`), using the learning rate specified in the plan.\n"
        "3. Implementing the main training loop (e.g., `for epoch in range(NUM_EPOCHS):`).\n"
        "4. Inside the loop: set model to train mode, iterate through `train_loader`, move data to `DEVICE`, zero gradients, perform forward pass, calculate loss, perform `loss.backward()`, and `optimizer.step()`.\n"
        "5. Implementing validation logic: after each training epoch (or as specified), set model to eval mode, iterate through `val_loader`, calculate validation loss and the primary competition metric (e.g., AUC, accuracy, log_loss). Print these validation metrics clearly using the format like `print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Val Metric (AUC): {val_auc_score:.4f}')`.\n"
        "6. (Optional, if in plan) Implement logic for saving the best performing model based on the validation metric (e.g., `torch.save(model.state_dict(), 'best_model.pth')`).\n"
        "The `model` variable should be updated (trained) in place. Include '# Thought:' comments."
    )
    return args

def get_coder_chain_user_prompt_segment_submission(
    task_summary: str, master_plan_text: str, current_code_so_far: str, competition_name: str, data_preview_content: Optional[str]
) -> Dict[str, Any]:
    args = _get_base_coder_chain_user_prompt_args(task_summary, master_plan_text, current_code_so_far, competition_name, data_preview_content)
    args["Your Current Coding Segment: Test Prediction & Submission File Generation"] = (
        "Based on the 'Full Master Plan' and using the trained `model`, test data and test IDs ,'Python Code Generated So Far', "
        "write the Python code block *only* for generating predictions on the test set and creating the `submission.csv` file. This includes:\n"
        "1. Loading the best model weights if they were saved to a file during training `).\n"
        "2. Setting the model to evaluation mode (`model.eval()`).\n"
        "3. Iterating through the test data, moving it to `DEVICE`, and generating predictions (e.g., probabilities using `torch.sigmoid(model(images))` or `model.predict_proba()`).\n"
        "4. Collecting all test predictions and their corresponding IDs.\n"
        "5. Creating a pandas DataFrame matching the submission file format specified in the 'Task Description' or 'Full Master Plan' (usually columns like 'id' and the target prediction, e.g.).\n"
        "6. Saving this DataFrame to `./submission/submission.csv` using `df.to_csv(path, index=False)`.\n"
        "Remember to include '# Thought:' comments."
    )
    return args

CHAINED_CODER_USER_PROMPT_CONSTRUCTORS = {
    "Setup & Imports": get_coder_chain_user_prompt_segment_setup,
    "Data Loading": get_coder_chain_user_prompt_segment_data_loading,
    "Data Preprocessing": get_coder_chain_user_prompt_segment_preprocessing,
    "Modeling": get_coder_chain_user_prompt_segment_modeling,
    "Training & Validation": get_coder_chain_user_prompt_segment_training,
    "Prediction & Submission": get_coder_chain_user_prompt_segment_submission,
}

CHAINED_CODER_SYSTEM_PROMPT_GETTERS = {
    "Setup & Imports": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SETUP), # Assuming segment1 is setup
    "Data Loading": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_DATA_LOADING),
    "Data Preprocessing": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_PREPROCESSING),
    "Modeling": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_MODELING),
    "Training & Validation": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_TRAINING),
    "Prediction & Submission": lambda: copy.deepcopy(CODER_CHAIN_SYSTEM_PROMPT_SEGMENT_SUBMISSION),
}




segments_order = [
            "Setup & Imports",
            "Data Loading",
            "Data Preprocessing",
            "Modeling",
            "Training & Validation", 
            "Prediction & Submission"
        ]




SEGMENT_REFLECTION_SYSTEM_PROMPT: Dict[str, Any] = {
    "SYSTEM": (
        "You are an expert Python Code Reviewer and Refinement Specialist for Kaggle competition solutions. "
        "you and the other coders are solving this competition: and the coding is segmented to 6 segments, these are :  "    
           "Setup & Imports"
           "Data Loading"
           "Data Preprocessing"
           "Modeling"
           "Training & Validation"
           "Prediction & Submission"  

        "You will be given an overall 'Master Plan', a 'Task Summary', the 'Python Code Generated So Far' (by previous segments), and an 'Initial Code Snippet' for the current segment. "
        "Your task is to meticulously reflect and review the 'Initial Code Snippet' for the current segment to ensure it is correct, adheres to the 'Master Plan', integrates seamlessly with 'Python Code Generated So Far', and follows best practices. "
        "Then, provide a 'Reflection Summary' of your findings and the 'Revised Code Snippet' for the current segment. "
        "Your goal is to produce a robust, correct, and plan-adherent code snippet for *this segment only*."
    ),
    "user_instructions": { # Meta-instructions for system prompt design
        "Review Focus Areas for the 'Initial Code Snippet'": [
            "1. **Correctness & Bugs:** Identify any syntax errors, runtime errors (e.g., NameError, TypeError, IndexError), or logical flaws within the snippet.",
            "2. **Plan Adherence:** Does the snippet accurately and completely implement the functionalities described for the current segment in the 'Master Plan'? Are there deviations or omissions?",
            "3. **Integration with Prior Code:** Does the snippet correctly use variables, functions, or classes defined in 'Python Code Generated So Far'? Does it correctly define new variables/functions needed by *subsequent* segments as implied by the Master Plan? Does it avoid unnecessary re-declarations or conflicting definitions? Does it include necessary new imports if not covered by prior code?",
            "4. **Best Practices & Clarity:** Is the code clean, readable? Are '# Thought:' comments present, clear, and correctly referencing Master Plan steps relevant to *this segment*?",
            "5. **Self-Contained for Segment:** Does the snippet focus *only* on the current segment's responsibilities as per the Master Plan, without prematurely implementing parts of future segments?"
        ],
        "Output Requirements (Strict Adherence Mandatory)": (
            "Your response MUST be structured in two main sections, clearly demarcated:\n\n"
            "Reflection Summary:\n"
            "[Your detailed analysis (3-5 bullet points or a short paragraph) covering findings from the review focus areas. Specifically mention any bugs found, deviations from the plan, or integration issues.]\n\n"
            "---\n"
            "Revised Code Snippet:\n"
            "```python\n"
            "# Thought: [Updated thought process for this revised segment, linking to Master Plan step(s)]\n"
            "[Your corrected and improved Python code for THIS SEGMENT ONLY. It should replace the initial snippet entirely.]\n"
            "```\n"
            "If the 'Initial Code Snippet' is already perfect and requires no changes, state this in the 'Reflection Summary' and reproduce the original snippet under 'Revised Code Snippet'."
        )
    }
}

def get_segment_reflection_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(SEGMENT_REFLECTION_SYSTEM_PROMPT)


def get_segment_reflection_user_prompt(
    task_summary: str,
    master_plan_text: str,          # Full plan
    current_segment_name: str, 
    code_generated_before_this_segment: str, # The code_accumulator *before* current snippet
    initial_code_snippet_for_this_segment: str # The snippet just generated by Coder
) -> Dict[str, Any]:
 
    relevant_plan_steps = f"# Relevant Master Plan Excerpt for Segment: {current_segment_name}\n"

    prompt = {
        "Context for Review": {
            "Overall Task Summary": task_summary,
            "Full Master Plan": master_plan_text,
            "Current Segment Being Reviewed": current_segment_name,
            "Current Segment Original Objective": get_coder_chain_system_prompt(current_segment_name)['user_instructions'],
            "Python Code Generated in PREVIOUS Segments": wrap_code(code_generated_before_this_segment if code_generated_before_this_segment.strip() else "# No code generated in prior segments."),
        },
        "Code Snippet to Review for THIS Segment": wrap_code(initial_code_snippet_for_this_segment),
        "Your Reflection Task for THIS Segment": (
            f"Carefully review the provided 'Code Snippet to Review for THIS Segment' ({current_segment_name}).\n"
            "1. Does it correctly implement the relevant part of the 'Full Master Plan' for this segment?\n"
            "2. Does it correctly implement the 'Current Segment Original Objective' for this segment?\n"
            "3. Is it free of bugs and errors?\n"
            "4. Does it integrate correctly with the 'Python Code Generated in PREVIOUS Segments' (using existing variables, not causing conflicts)?\n"
            "5. Are the '# Thought:' comments clear and accurate for this segment's actions?\n"
            "Provide your 'Reflection Summary' and then the 'Revised Code Snippet' for *this segment only*. "
            "If the initial snippet is perfect, say so and provide it again as the revised snippet."
        )
    }
    return prompt



# -----------------------------------------------------------------------------
# Chunk-level reflection prompts
# -----------------------------------------------------------------------------
CHUNK_REFLECTION_SYSTEM_PROMPT: Dict[str, Any] = {
    "SYSTEM": (
        "You are an expert Python Code Reviewer and Refinement Specialist for Kaggle competition solutions. "
        "The solution is organized into multiple segments (e.g. Setup & Imports, Data Loading, etc.). "
        "Now you will be given a *chunk* of these segments together, along with:\n"
        "  - The overall 'Master Plan'.\n"
        "  - The individual 'Task Summaries' for each segment in the chunk.\n"
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
            "2. Each segment’s adherence to the Master Plan.",
            "3. Integration among the chunk’s segments and with preceding code.",
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
    task_summary: str,
    master_plan: str,
    segment_names: List[str],
    code_before_chunk: str,
    initial_chunk_code: str
) -> Dict[str, Any]:
    """
    Builds the user prompt for chunk-level reflection.
    - task_summaries: one per segment in the chunk, in same order as segment_names.
    - master_plan: the full plan text.
    - code_before_chunk: everything generated so far, before this chunk.
    - initial_chunk_code: the concatenated code snippets for this chunk.
    """


    return {
        "Context for Chunk Review": {
            "Overall Task Summary": task_summary,
            "Full Master Plan": master_plan,
            "Segments in This Chunk": segment_names,
            "Python Code Generated Before This Chunk": wrap_code(
                code_before_chunk or "# No prior code generated."
            ),
        },
        "Initial Chunk Code to Review": wrap_code(initial_chunk_code),
        "Your Chunk Reflection Task": (
            f"Please review the above 'Initial Chunk Code to Review' for segments "
            f"{', '.join(segment_names)} as a single unit.  \n"
            "Address the focus areas listed in the system prompt, then produce your "
            "**Reflection Summary** and **Revised Code Snippet** exactly in the format specified."
        )
    }
