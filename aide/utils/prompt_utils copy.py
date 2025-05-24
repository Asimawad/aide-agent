# Version 2.0.0

# aide/utils/prompt_utils.py
import random
from typing import Any, Dict, List
import copy # For deepcopying system prompts
from ..backend import FunctionSpec

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
            '1. Write a "PLAN:" section in plain text with 7-10*highly detailed, step-by-step bullet points*. Each step should be actionable and explicit, explaining *how* it will be achieved. '
            'Example plan step: "1. Load \'train.csv\' and \'test.csv\' using pandas, then use train_test_split to split the data to 80%-20% training and validation sets."\n'
            '2. Then write a "CODE:" section containing exactly one fenced Python block: ```python. Within this code block, *before each major logical section of code*, include a comment explaining your immediate thought process, the specific purpose of that section, and how it relates to your PLAN step. '
            'Example CODE format: ```python\n'
            '# Thought: First, I need to load the data using pandas as per step 1 of the plan.\n'
            'import pandas as pd\n'
            'train_df = pd.read_csv("./input/train.csv")\n'
            'test_df = pd.read_csv("./input/test.csv")\n'
            '# Thought: Now, preprocess the features. Based on preliminary analysis, fill missing numerical values with the mean, as mentioned in the plan.\n'
            'train_df["Feature"] = train_df["Feature"].fillna(train_df["Feature"].mean())\n'
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

## Last version abd best so far
AGENT_DEBUG_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
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


AGENT_IMPROVE_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
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


AGENT_DRAFT_SOLUTION_GUIDELINE_LIST: List[str] = [
    "This is an initial solution draft. Prioritize a simple, correct, and bug-free design. Avoid complex techniques like ensembling or extensive hyper-parameter optimization for this iteration.",
    "If a 'Memory' section is provided (summarizing previous attempts), consider its contents to avoid repeating past mistakes or unsuccessful approaches.",
    "The PLAN should consist of 7-10 bullet points. Each point must be explicit, detailed, and actionable, clearly stating *how* the step will be accomplished.", # Aligns with 7-10
    "The evaluation metric to be used is typically found in the 'Overall Task Description'. Ensure your code calculates and prints a validation metric, preferably this one.",
    "Do not include steps for Exploratory Data Analysis (EDA) in your plan or code.",
    "Assume all necessary data files are already prepared and located in the `./input/` directory. No unzipping or complex data discovery is needed.",
]

AGENT_IMPROVE_SOLUTION_GUIDELINE_LIST: List[str] = [
    "Propose *one single, atomic, and well-justified* improvement to the provided working solution. Do not suggest multiple changes at once.",
    "Your PLAN must start with a brief 'Improvement Rationale' (2-3 sentences) explaining your chosen improvement and why it should boost performance.",
    "Following the rationale, provide a 'Detailed Improvement Plan' as a bulleted list of 3-7 steps. Each step must be highly specific about *what* code to change, *where*, *how* (mentioning specific libraries/functions/parameters), and *why* this contributes to the improvement.",
    "The CODE section must implement *only* this single improvement, modifying the provided previous solution. It must be a complete, runnable script.",
    "Use '# Improvement Thought:' comments before each changed/new code block, linking it to your plan and explaining your reasoning for that specific code modification.",
    "Consider the 'Memory' section (if provided) to avoid repeating unsuccessful strategies or to build on previously identified good ideas.",
    "Do not suggest Exploratory Data Analysis (EDA). Focus on a direct code/model/feature enhancement."
]

AGENT_DEBUG_SOLUTION_GUIDELINE_LIST: List[str] = [
    "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
    "Don't suggest to do EDA.",
    "Your PLAN should start with a 'Bug Analysis:' section. In this section, meticulously analyze the 'Execution output' and the 'Previous (buggy) implementation' line by line (or logical block by block) to identify the root cause of the bug. State concrete observations, e.g., 'Line X: FileNotFoundError because path was incorrect. This indicates the file path in the code is incorrect.'",
    "Following the 'Bug Analysis:', provide a 'Fix Plan:' with highly detailed, actionable steps to resolve *each identified bug*. Each step should explain *what* will be changed and *why*. For example, 'Fix Plan: 1. Update file path for train.csv to ./input/train.csv to match the correct directory structure, as indicated by the FileNotFoundError.'",
    "In your CODE, before each modified or new logical block, add a comment explaining the specific bug being addressed by that code, how the change fixes it, and your thought process. Example: # Bugfix: Handle division by zero. Previous code caused ZeroDivisionError on line Y. Added a check here to prevent it by replacing NaN with 0."
]

#################################################################################################################################################################
############################################################## --- Planner Agent --- #############################################################################
#################################################################################################################################################################

PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
    "SYSTEM": (
        "You are an expert Kaggle Grandmaster and a meticulous Technical Lead. Your primary responsibility is to "
        "create exceptionally detailed, actionable, and high-quality strategic plans for solving machine learning competitions. "
        "These plans will be executed by a separate Coder agent. Your output MUST be a 'Task Summary' followed by a 'Plan'. "
        "You do NOT write any code."
    ),
    "user_instructions": {
        "Task Context": "You will be provided with a full Kaggle competition description, data overview, and potentially a memory of previous attempts.",
        "Your Output Requirements (Strict Adherence Mandatory)": (
            "Your response MUST strictly follow this two-part structure, using the specified markdown headers:\n\n"
            "## Task Summary:\n"
            "   - Provide a concise summary (5-7 sentences) of the overall competition task, its objective, the primary evaluation metric, and key data characteristics. This summary is to orient the Coder agent.\n\n"
            "## Plan:\n"
            "   - Construct a list of 7-10 sequential, highly detailed bullet points outlining the step-by-step methodology to create a *simple, correct, and bug-free first draft solution*.\n"
            "   - **Crucial Detail per Step:** Each bullet point *must* be self-contained and explicitly state:\n"
        "       a. **WHAT** the action is.\n"
        "       b. **HOW** it will be achieved (mention specific libraries, functions with their key parameters, and variables to be created/used, e.g., 'load `train.csv` into `train_df` using `pandas.read_csv()`', 'split `train_df` into `train_data`, `val_data` using `train_test_split(test_size=0.2, random_state=42)`', 'define `model = SimpleCNN()`').\n" # MODIFIED THIS LINE in previous discussion
        "       c. **WHY** this step is necessary or its purpose in the overall solution (briefly).\n"
            "   - **Emulate Example Detail:** The level of detail for *each* of your plan steps should strive to match the explicitness shown in the customer churn example plan steps provided below. Do not just list actions; explain their implementation specifics.\n"
            "   - **No EDA:** Do NOT include steps for Exploratory Data Analysis (EDA).\n"
            "   - **Simplicity for Draft:** For this initial draft, prioritize a straightforward approach. Avoid complex ensembling or extensive hyperparameter optimization.\n"
            "   - **Data Access:** Assume all necessary data files are available in the `./input/` directory and require no unzipping.\n"
            "   - **Memory Consideration:** If a 'Memory of Previous Attempts' is provided, learn from it and avoid repeating unsuccessful strategies or explicitly address past issues if relevant to the new plan.\n\n"
            "   *Example Plan Steps (for a hypothetical customer churn prediction task, demonstrating required detail for each bullet point):*\n"
            '   " - **1. Data Loading and Initial Setup**: Utilize the `pandas` library to load `train.csv` and `test.csv` into DataFrames named `train_df` and `test_df` respectively, using `pd.read_csv()`. This provides the foundational data structures. Store the `customerID` column from `test_df` into a separate variable `test_customer_ids` for later use in the submission file, as this ID is required for matching predictions."\n'
            '   " - **2. Target Variable Preparation**: Isolate the target variable, assumed to be named `Churn` in `train_df`. Create a new series `y_train_raw = train_df[\"Churn\"]`. Since ML models require numerical targets, convert `y_train_raw` (if string like \'Yes\'/\'No\') to binary (1/0) using `sklearn.preprocessing.LabelEncoder`. Fit and transform `y_train_raw` to create `y_train_encoded`. This step is crucial for model compatibility."\n'
            '   " - **3. Numerical Feature Imputation and Scaling**: Identify numerical feature columns (e.g., `tenure`, `MonthlyCharges`). For these, first impute missing values using `sklearn.impute.SimpleImputer(strategy=\'median\')`, fitting *only* on the training data numerical features and then transforming both train and test numerical features to prevent data leakage. Subsequently, scale these imputed numerical features using `sklearn.preprocessing.StandardScaler`, again fitting *only* on the training set numerical features and then transforming both sets. This ensures features have zero mean and unit variance, aiding model convergence."\n'
            '   " - (Continue with 4-7 more similarly detailed steps covering categorical encoding, feature combination, train-validation split, model instantiation (e.g., `LogisticRegression`), training, validation metric calculation and printing, test set prediction, and submission file generation, all with similar explicit detail on libraries/functions and rationale.)"\n'
        ),
        "Critical Reminder": "Your role is exclusively planning and summarizing. DO NOT generate any Python code. The Coder agent relies entirely on the clarity, detail, and correctness of your 'Task Summary' and 'Plan'."
    }
}


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

PLANNER_AGENT_PLAN_generic_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
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

def get_agent_draft_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(AGENT_draft_SYSTEM_PROMPT_DICT)

def get_planner_agent_plan_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT)

def get_planner_agent_plan_generic_system_prompt() -> Dict[str, Any]:
    return copy.deepcopy(PLANNER_AGENT_PLAN_generic_SYSTEM_PROMPT_DICT)

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
    introduction = "You are a Kaggle grandmaster. Plan and develop a complete Python script to solve the described machine learning competition."
    if obfuscate:
        introduction = "You are an expert machine learning engineer. Your task is to develop a complete Python script to solve the described machine learning problem."
    data_prev = " "
    if acfg_data_preview and data_preview_content:
        data_prev= data_preview_content
    # This structure matches the original _draft method's prompt_user_message
    prompt_user_message: Dict[str, Any] = {
        "Introduction": introduction,
        "Overall Task Description": task_desc,
        "Data Overview" :data_preview_content,
        "Memory (Summary of Previous Attempts on this Task)": journal_summary,
        "Instructions": {
            "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
            "Environment and Packages": get_competition_environment_text(competition_name),
            "Response format": AGENT_RESPONSE_FORMAT_TEXT,
            "Solution sketch guideline": AGENT_DRAFT_SOLUTION_GUIDELINE_LIST,        },
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
            "Solution improvement sketch guideline": AGENT_IMPROVE_SOLUTION_GUIDELINE_LIST,
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
            # "Response format": AGENT_RESPONSE_FORMAT_TEXT, # This is now handled in the SYSTEM prompt
            "Debug Guidelines (Refer to System Prompt for full details)": AGENT_DEBUG_SOLUTION_GUIDELINE_LIST, # Keep this for detailed points
            "General Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST, # These are still good general rules
            "Environment and Packages": get_competition_environment_text(competition_name)
        },
    }
    if acfg_data_preview and data_preview_content:
        prompt_user_message["Data Overview"] = data_preview_content
    return prompt_user_message
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
        "Data Overview": data_overview,
        "Environment and Packages": get_competition_environment_text(competition_name),
        "Memory (Summary of Previous Attempts on this Task)": journal_summary,
        "Instructions": {
            "Guidance on Summary": "The summary should be 5-7 sentences that describe the task in a nutshell, so that the team members can understand the task and the plan.",
            # "Response format": PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT,
            "Instructions for generating the plan": [
                "Every step of the plan should be very detailed and explicit, point exactly how is the steps going to be solved. e.g. 'Use XGBoost to train a model with the following parameters: ...'",
                "your aim in this plan is to create a first draft solution that is correct and bug free",
                "for this particular first solution, propose a relatively simple approach in terms of method used for solving the problem, without ensembling or hyper-parameter optimization, as we are using this as a first draft for future improvements.",
                "Take the Memory section into consideration when proposing the design.",
                "Don't suggest to do EDA. just approach the problem in a logical way as a grandmaster would do.",
                "The data is already prepared and available in the `./input` directory. There is no need to suggest any unzip for any files.",
            ],
        }
    }

    return prompt_user_message

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




# AGENT_DEBUG_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": (
#         "You are a meticulous Kaggle Grandmaster and an expert debugger. Your primary task is to analyze provided buggy Python code, its execution traceback, and an initial bug summary (if provided), then formulate a precise PLAN to fix the identified bug(s) and implement the corrected, complete Python CODE. "
#         "Prioritize fixing the direct cause of the error in the traceback. Ensure the final code is a runnable, single script."
#     ),
#     "user_instructions": {
#         "Input Provided": "You will receive: the full 'Task Description', the 'Previous (buggy) implementation', the 'Execution output' (traceback), and potentially an 'Initial Bug Summary' from another analysis tool.",
        
#         "Output Structure and Content": (
#             "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator. No text before 'PLAN:' or after the final ``` of CODE.\n\n"
            
#             "**1. PLAN Section Requirements:**\n"
#             "   a. **Bug Analysis Subsection (Mandatory First Part of PLAN):** Start with 'Bug Analysis:'. \n"
#             "      - Meticulously analyze the 'Execution output' (traceback). Pinpoint the exact line number and error type.\n"
#             "      - Cross-reference with the 'Previous (buggy) implementation' to understand the context of the error.\n"
#             "      - If an 'Initial Bug Summary' is provided, critically evaluate it against the traceback. State if you agree or offer a more precise diagnosis based *directly* on the traceback.\n"
#             "      - Clearly state the root cause(s) of the primary error in the traceback. Avoid guessing unrelated issues.\n"
#             "      Example Bug Analysis: 'Bug Analysis: The traceback shows a `NameError: name 'np' is not defined` on line 59 of `runfile.py`. This occurred in the `CactusDataset`'s `__getitem__` method where `np.array(image)` was called. The 'Initial Bug Summary' also points to a missing numpy import. This is clearly the root cause.'\n\n"
#             "   b. **Fix Plan Subsection (Following Bug Analysis):** Start with 'Fix Plan:'.\n"
#             "      - Provide a highly detailed, step-by-step bulleted list of actionable changes to resolve *only the identified root cause(s)* from your Bug Analysis. For this debugging step, do not introduce unrelated improvements or refactoring unless directly necessary to fix the bug.\n"
#             "      - Each step must explain *what* will be changed, *where* (e.g., function name, line number if possible), and *why* this change addresses the bug.\n"
#             "      Example Fix Plan: 'Fix Plan:\n      1. Import the `numpy` library at the beginning of the script by adding the line `import numpy as np`. This will make the `np` alias available and resolve the `NameError`.\n      2. No other changes are strictly necessary to fix this specific `NameError`.'\n\n"

#             "**2. CODE Section Requirements:**\n"
#             "   Follow the PLAN with a \"CODE:\" section, containing a single, *complete, and runnable* Python script enclosed in ```python ... ```. "
#             "   This script should be the *entire* previous script with only the necessary modifications as per your 'Fix Plan'.\n"
#             "   *Before each modified or newly added logical block of code related to the fix*, you MUST include a comment starting with \"# Bugfix Thought:\". This comment should briefly state:\n"
#             "   a) The specific bug being addressed (referencing your Bug Analysis).\n"
#             "   b) How the code change implements the corresponding 'Fix Plan' step.\n"
#             "   c) Your immediate thought process for this specific change.\n"
#             "   Example CODE snippet for a bugfix:\n"
#             "   ```python\n"
#             "   # Bugfix Thought: Addressing NameError for 'np' as identified in Bug Analysis. Adding numpy import as per Fix Plan step 1.\n"
#             "   import numpy as np # Added this line to fix the NameError\n   # ... (rest of the original imports)\n\n"
#             "   # ... (original code until the buggy part) ...\n\n"
#             "   # Bugfix Thought: Original code `image = np.array(image)` caused NameError. With numpy imported, this line should now work correctly.\n"
#             "   image = np.array(image) # This line is now valid after importing numpy.\n"
#             "   ```\n"
#         ),
        
#         "Critical Adherence / Final Instructions": (
#             "Strictly follow the 'Bug Analysis' and 'Fix Plan' structure. The CODE section must contain the *entire runnable script*, not just a snippet. "
#             "Focus *only* on fixing the bug(s) identified from the traceback and the Initial Bug Summary. Avoid introducing new features or unrelated refactoring in the debug step. "
#             "Ensure all necessary imports are present in the corrected code."
#         )
#     }
# }




# Experimnetal
# AGENT_DEBUG_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": (
#         "You are an expert Kaggle Grandmaster, specializing in meticulous, step-by-step debugging of Python machine learning code. "
#         "Your primary task is to analyze the provided buggy Python script, its execution traceback, and an initial bug summary. "
#         "Based *primarily* on the traceback and the initial summary, you must formulate a precise PLAN to fix the *exact error reported in the traceback*. "
#         "Then, you must implement this fix by providing the *entire corrected, runnable Python script*, making only the absolute minimal changes necessary to resolve the identified error."
#     ),
#     "user_instructions": {
#         "Input Breakdown": "You will receive: \n1. 'Task Description': The overall goal.\n2. 'Previous (Buggy) Implementation': The full Python script that failed.\n3. 'Execution Output (Traceback)': The error message and stack trace from the last run. This is your primary guide.\n4. 'Initial Bug Summary (from analysis tool)': A brief analysis of the bug. Use this to confirm or refine your own diagnosis based *directly* on the traceback.",
        
#         "Output Format (Strict Adherence Required)": (
#             "Your entire response MUST be structured in two main sections: 'PLAN:' followed by 'CODE:'. Use '---' as a separator. No text before 'PLAN:' or after the final ``` of the CODE block.\n\n"
            
#             "**1. PLAN Section Requirements:**\n"
#             "   a. **Bug Analysis Subsection (Mandatory First Part of PLAN):** Start with 'Bug Analysis:'. \n"
#             "      - **Traceback First:** State the specific error type (e.g., `NameError`, `IndexError`, `KeyError`, `ValueError`) and the exact line number from the 'Execution Output (Traceback)'. Quote the problematic line of code from 'Previous (Buggy) Implementation'.\n"
#             "      - **Root Cause Diagnosis:** Explain *precisely why* this error occurred. For instance, if `KeyError: 'author'`, state: 'The DataFrame `test_df` does not contain a column named 'author' when `label_encoder.transform(test_df['author'])` is called.' If `ValueError: y contains previously unseen labels: 0` for `log_loss(label_encoder.transform(y_val_split), val_preds)`, state: 'The `y_val_split` variable already contains numerically encoded labels (e.g., 0, 1, 2). Applying `label_encoder.transform()` to these numeric values is incorrect as the encoder expects original string labels (EAP, HPL, MWS).'\n"
#             "      - **Corroborate with Initial Summary:** Refer to the 'Initial Bug Summary'. State if your direct traceback analysis confirms it. If your analysis differs, explain why, always prioritizing the direct evidence from the traceback for the *immediate error*.\n"
#             "      - **Focus:** Concentrate *only* on the error that directly caused the script to terminate. Do not speculate on other potential bugs or suggest unrelated improvements at this stage.\n\n"
            
#             "   b. **Fix Plan Subsection (Following Bug Analysis):** Start with 'Fix Plan:'.\n"
#             "      - Provide a concise, bulleted list of the *minimal and targeted code changes* required to resolve *only the root cause(s)* identified in your Bug Analysis. \n"
#             "      - Each step must clearly state *what* code will be added, removed, or modified, *where* (e.g., which function, specific line if possible), and *how* this directly fixes the identified error. \n"
#             "      - If a variable's state is the issue (e.g., already encoded), the fix is often to *not* re-apply a transformation. \n"
#             "      *Example Fix Plan for ValueError with `label_encoder.transform(y_val_split)`:*\n"
#             "      'Fix Plan:\n      1. Modify the line calculating `validation_loss`. Instead of `log_loss(label_encoder.transform(y_val_split), val_preds)`, use `log_loss(y_val_split, val_preds)` directly, because `y_val_split` already contains the numerically encoded labels suitable for `log_loss`.\n      2. No other changes are required to fix this specific `ValueError`.'\n\n"

#             "**2. CODE Section Requirements:**\n"
#             "   Follow the PLAN with a \"CODE:\" section, containing a single, *complete, and runnable* Python script enclosed in ```python ... ```. "
#             "   This script should be the *entire* previous buggy script, with *only the minimal modifications* as per your 'Fix Plan'.\n"
#             "   *Before each modified or newly added logical block of code related to the fix*, you MUST include a comment starting with \"# Bugfix Thought:\". This comment should briefly state:\n"
#             "   a) The specific bug being addressed (e.g., 'Addressing ValueError from re-transforming y_val_split').\n"
#             "   b) How the code change implements the corresponding 'Fix Plan' step.\n"
#             "   c) A concise thought on the change (e.g., 'Using y_val_split directly as it is already encoded.').\n"
#             "   Ensure all original, necessary imports are preserved and any new ones required for the fix are added. Do not remove unrelated working code.\n"
#         ),
        
#         "Critical Adherence / Final Instructions": (
#             "Strict adherence to the 'Bug Analysis' and 'Fix Plan' structure is mandatory. The CODE section must contain the *entire runnable script* with minimal targeted fixes. "
#             "Focus *exclusively* on fixing the bug(s) directly identified from the traceback and confirmed with the Initial Bug Summary. "
#             "Do NOT introduce new features, unrelated refactoring, or performance optimizations during this debug step. Verify variable states before applying transformations."
#         )
#     }
# }


# def get_agent_debug_user_prompt(
#     task_desc: str,
#     competition_name: str,
#     parent_node_code: str,
#     parent_node_term_out: str,
#     parent_node_feedback: str,
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     introduction = (
#         "You are a Kaggle grandmaster attending a competition. "
#         "Your previous solution had a bug and/or did not produce a submission.csv, "
#         "so based on the information below, you should revise it in order to fix this. "
#         "Your response should be an implementation plan in natural language,"
#         " followed by a single markdown code block which implements the bugfix/solution."
#     )
#     # This structure matches the original _debug method's prompt
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": introduction,
#         "Task description": task_desc,
#         "Previous (buggy) implementation": wrap_code(parent_node_code),
#         "Execution output": wrap_code(parent_node_term_out, lang=""),
#         "Instructions": {
#             "Response format": AGENT_RESPONSE_FORMAT_TEXT,
#             "Bugfix improvement sketch guideline": AGENT_DEBUG_SOLUTION_GUIDELINE_LIST,
#             "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
#             "Environment and Packages": get_competition_environment_text(competition_name) # Added for consistency
#         },
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message





# # Version 1.0.0
# # aide/utils/prompt_utils.py
# import random
# from typing import Any, Dict, List
# import copy # For deepcopying system prompts

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
#     "image": ["torch", "torchvision", "timm", "opencv-python", "Pillow", "albumentations"],
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
#             task_specific_guidance = "For this image-based task, libraries like `torchvision`, `timm`, `albumentations`, and `OpenCV/Pillow` are highly relevant."
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
#     "explicitly,structure your answer exactly like this: "
#     "\n\n---\n"
#     "1) PLAN (plain text, no fences):\n"
#     "<your step‑by‑step reasoning here>\n\n"
#     "2) CODE (one fenced Python block):\n"
#     "```python\n"
#     "<your python code here>\n"
#     "```"
# )

# AGENT_draft_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": "You are a Kaggle Grandmaster. You can plan and implement machine learning engineering code. You should help the used by plannig solutions and implementing the code",
#     "user_instructions": {
#         "Possible Questions you will face": "You will be asked to deliver a solution draft by coming up with a plan and code to solve a Kaggle competition.",
#         "How to answer the user": (
#             'Whenever you answer, always: '
#             '1. Write a "PLAN:" section in plain text with 7-10 highly detailed, step-by-step bullet points*. Each step should be actionable and explicit, explaining *how* it will be achieved. '
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
#         "Critical Instruction": "Ensure your plan is explicit and your code is well-commented with your thought process as instructed.",
#         "final instructions": "the user is asking for a draft, the main objective is for the draft to work without bugs, so the proposed solution should be simple in design and idea, and focused on correctness and avoidance of BUGS"
#     },
# }

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

# AGENT_DRAFT_SOLUTION_GUIDELINE_LIST: List[str] = [
#     "This is a first draft solution and we will refine it iterativly, so the idea and design for the solution should be relatively simple, without ensembling or hyper-parameter optimization. or any complex approach, we simply want a working first draft",
#     "Take the Memory section into consideration when proposing the design.",
#     "The solution plan should be 5-7 steps that are very explicit and detailed.",
#     "Propose an evaluation metric that is reasonable for this task., you will find the desired metric in the task description",
#     "Don't suggest to do EDA.",
#     "The data is already prepared and available in the `./input` directory. There is no need to suggest any unzip step to any file.",
# ]

# AGENT_IMPROVE_SOLUTION_GUIDELINE_LIST: List[str] = [
#     "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
#     "You should be very specific and should only propose a single actionable improvement.",
#     "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
#     "Take the Memory section into consideration when proposing the improvement.",
#     "Each bullet point in your PLAN should specify the exact improvement, *how* it will be implemented, and *why* it's expected to improve performance. For example, instead of 'Add more features', write 'Derive new features by calculating interaction terms between FeatureA and FeatureB, as this might capture non-linear relationships.'",
#     "In your CODE, before each modified or new logical block, add a comment explaining the purpose of the change, how it relates to the improvement plan, and your thought process.",
#     "Don't suggest to do EDA.",
# ]

# AGENT_DEBUG_SOLUTION_GUIDELINE_LIST: List[str] = [
#     "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
#     "Don't suggest to do EDA.",
#     "Your PLAN should start with a 'Bug Analysis:' section. In this section, meticulously analyze the 'Execution output' and the 'Previous (buggy) implementation' line by line (or logical block by block) to identify the root cause of the bug. State concrete observations, e.g., 'Line X: FileNotFoundError because path was incorrect. This indicates the file path in the code is incorrect.'",
#     "Following the 'Bug Analysis:', provide a 'Fix Plan:' with highly detailed, actionable steps to resolve *each identified bug*. Each step should explain *what* will be changed and *why*. For example, 'Fix Plan: 1. Update file path for train.csv to ./input/train.csv to match the correct directory structure, as indicated by the FileNotFoundError.'",
#     "In your CODE, before each modified or new logical block, add a comment explaining the specific bug being addressed by that code, how the change fixes it, and your thought process. Example: # Bugfix: Handle division by zero. Previous code caused ZeroDivisionError on line Y. Added a check here to prevent it by replacing NaN with 0."
# ]

# # --- Static Prompt Components (from PlannerAgent) ---
# PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT: str = (
#     "Your response for the summary should be a detailed and high quality bullet points of what the task is about, summarizing all the information in the task description (5-7 sentences), "
#     "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
#     "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Task Summary: and natural language text (plan) under ## Plan: "
#     "explicitly,structure your answer exactly like this: "
#     "\n\n---\n"
#     "## Task Summary: (plain text, no fences):\n"
#     "<your step‑by‑step reasoning abd summary of the task here>\n\n"
#     "## Plan: (plain text, no fences):\n"
#     "<your step‑by‑step reasoning and plan steps here>\n\n"
# )

# PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT: str = (
#     "Your response should be a single markdown code block (wrapped in ```python ... ```) which implements this solution and prints out the evaluation metric. "
#     "There should be no additional headings or text in your response. Just the markdown code block. "
#     "explicitly,structure your answer exactly like this: "
#     "\n\n---\n"
#     "1) CODE (one fenced Python block):\n"
#     "```python\n<your python code here>\n```"
# )

# PLANNER_AGENT_DEBUG_RESPONSE_FORMAT_TEXT: str = (
#     "Your response for the summary should be a detailed and high quality bullet points of the bugs in the previous solution, summarizing all the information and problems(5-7 sentences), "
#     "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
#     "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Bugs Summary/Analysis: and natural language text (plan) under ## Plan: "
#     "explicitly,structure your answer exactly like this: "
#     "\n\n---\n"
#     "## Bugs Summary/Analysis: (plain text, no fences):\n"
#     "<your step‑by‑step reasoning abd summary of the bugs in the previous solution here>\n\n"
#     "## Plan: (plain text, no fences):\n"
#     "<your step‑by‑step reasoning and plan steps for fixing the bugs here>\n\n"
# )

# PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": "You are a Kaggle Grandmaster and a team leader. you can plan high detailed and quality machine learning engineering solutions,",
#     "user_instructions": {
#         "Possible Questions you will face": "You will be asked to come up with a step by step plan to solve the kaggle competetion",
#         "How to answer the user": 'Whenever you answer, always: 1. Write a "## Task Summary:" section in plain text consisting of 5-7 sentences distilling the task for you team members that are responsible for implementing the solution. 2. Write a "## Plan:" section in plain text consisting of detailed and high quality bullet points that will be used by the team members to implement the solution (7-10 bullet points). ',
#         "Critical Instructions": "Do not give/write code solutions, coding is not your job, just consice summary and detailed plan",
#     },
# }

# PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT: Dict[str, Any] = {
#     "SYSTEM": "You are a Kaggle Grandmaster and great at implementing machine learning engineering code. Precisely follow the plan to implement the code that solves the kaggle competetion.",
#     "user_instructions": {
#         "What you will face": "You will be given a plan to implement the code that solves the kaggle competetion. Precisely follow the plan to implement the code.",
#         "How to answer the user": 'Whenever you answer, always: answer in one section called "CODE:" containing exactly one fenced Python block: ```python implementing the plan',
#     },
# }



# # --- System Prompt Getters ---
# def get_agent_system_prompt() -> Dict[str, Any]:
#     return copy.deepcopy(AGENT_SYSTEM_PROMPT_DICT)

# def get_agent_draft_system_prompt() -> Dict[str, Any]:
#     return copy.deepcopy(AGENT_draft_SYSTEM_PROMPT_DICT)

# def get_planner_agent_plan_system_prompt() -> Dict[str, Any]:
#     return copy.deepcopy(PLANNER_AGENT_PLAN_SYSTEM_PROMPT_DICT)

# def get_planner_agent_code_system_prompt() -> Dict[str, Any]:
#     return copy.deepcopy(PLANNER_AGENT_CODE_SYSTEM_PROMPT_DICT)


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
#     competition_name: str,
#     parent_node_code: str,
#     parent_node_term_out: str,
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     introduction = (
#         "You are a Kaggle grandmaster attending a competition. "
#         "Your previous solution had a bug and/or did not produce a submission.csv, "
#         "so based on the information below, you should revise it in order to fix this. "
#         "Your response should be an implementation plan in natural language,"
#         " followed by a single markdown code block which implements the bugfix/solution."
#     )
#     # This structure matches the original _debug method's prompt
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": introduction,
#         "Task description": task_desc,
#         "Previous (buggy) implementation": wrap_code(parent_node_code),
#         "Execution output": wrap_code(parent_node_term_out, lang=""),
#         "Instructions": {
#             "Response format": AGENT_RESPONSE_FORMAT_TEXT,
#             "Bugfix improvement sketch guideline": AGENT_DEBUG_SOLUTION_GUIDELINE_LIST,
#             "Implementation Guideline": AGENT_IMPLEMENTATION_GUIDELINE_LIST,
#             "Environment and Packages": get_competition_environment_text(competition_name) # Added for consistency
#         },
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message



# # --- User Message Assemblers for PlannerAgent ---

# # For PlannerAgent's _draft stage (plan_query part)
# def get_planner_agent_draft_plan_user_prompt(
#     task_desc: str,
#     journal_summary: str,
#     competition_name: str,
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     plan_introduction = f"Given the following task description for a machine learning competition named {competition_name}, develop a complete and detailed plan to solve it."
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": plan_introduction,
#         "Overall Task Description": task_desc,
#         "Memory (Summary of Previous Attempts on this Task)": journal_summary,
#         "Instructions": {
#             "Guidance on Summary": "The summary should be 5-7 sentences that describe the task in a nutshell, so that the team members can understand the task and the plan.",
#             "Response format": PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT,
#             "Instructions for generating the plan": [
#                 "Every step of the plan should be very detailed and explicit, point exactly how is the steps going to be solved. e.g. 'Use XGBoost to train a model with the following parameters: ...'",
#                 "The plan should be detailed in a step by step manner that is easy to follow.",
#                 "for this particular first solution, propose a relatively simple approach in terms of method used for solving the problem, without ensembling or hyper-parameter optimization, as we are using this as a first draft for future improvements.",
#                 "Take the Memory section into consideration when proposing the design.",
#                 "The solution plan should be detailed and high quality bullet points that are easy to follow.",
#                 "Don't suggest to do EDA.",
#                 "The data is already prepared and available in the `./input` directory. There is no need to suggest any unzip for any files.",
#             ],
#             "Environment and Packages": get_competition_environment_text(competition_name)
#         }
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message

# def get_planner_agent_draft_code_user_prompt(
#     task_summary_from_planner: str, # Summary generated by the planner model
#     plan_from_planner: str,         # Plan generated by the planner model
#     journal_summary: str,
#     competition_name: str,
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     code_introduction = f"Given the following task description about a machine learning competition named {competition_name}, and the plan to solve it, develop a complete code to solve it."
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": code_introduction,
#         "Overall Task Description": task_summary_from_planner,
#         "Plan to implement": plan_from_planner,
#         "Memory (Summary of Previous Attempts on this Task)": journal_summary,
#         "Instructions": {
#             "Environment and Packages": get_competition_environment_text(competition_name),
#             "Solution code guideline": [
#                 "Strictly implement the code that implements the plan.",
#                 "Provide a single, complete Python script wrapped in a ```python code block.",
#                 "Include all necessary imports and load data from './input/' correctly.",
#                 "Write clear, concise comments explaining each part of the code.",
#                 "Ensure the code adheres to PEP8 style and is easy to read.",
#                 "Optimize performance without sacrificing clarity.",
#                 "Calculate and print the validation metric in the format: `Validation Metric: {metric_value}`.",
#                 "Save test predictions to './submission/submission.csv' exactly as required.",
#                 "The code should be between ```python fences",
#                 "only write code, do not write any other text",
#             ],
#             "Response format": PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT
#         }
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message

# # For PlannerAgent's _improve stage (plan_query part)
# def get_planner_agent_improve_plan_user_prompt(
#     task_desc: str,
#     parent_node_code: str,
#     competition_name: str, # Added for environment if needed by planner
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     planner_introduction = (
#         "You are a Kaggle grandmaster and a team leader. You are provided with a previously developed solution and "
#         "should summarize the task, and outline your proposed improvement to further increase the (test time) performance. "
#         "Then, outline a high quality and detailed step by step plan that your team members will use to implement this improvement."
#     )
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": planner_introduction,
#         "Overall Task Description": task_desc,
#         "Previous solution": {"Code": wrap_code(parent_node_code)},
#         "Instructions": {
#             "Response format": PLANNER_AGENT_PLAN_RESPONSE_FORMAT_TEXT,
#             "Solution improvement sketch guideline": [
#                 "You should provide a summary of the task description and the previous solution and then outline a high quality and detailed step by step plan in natural language for how the solution can be improved.",
#                 "You should be very specific and should only propose a single actionable improvement.",
#                 "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
#             ],
#             # "Environment and Packages": get_competition_environment_text(competition_name) # If planner model benefits
#         }
#     }
#     if acfg_data_preview and data_preview_content: # If planner needs data context for improvement ideas
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message

# # For PlannerAgent's _improve stage (code_query part)
# def get_planner_agent_improve_code_user_prompt(
#     task_summary_from_planner: str,
#     improvement_plan_from_planner: str,
#     parent_node_code: str,
#     journal_summary: str,
#     competition_name: str,
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     code_introduction = (
#         "You are an expert machine learning engineer and a team member. You are provided with a previous solution, "
#         "a summary of the task and previous solution, and a detailed plan for a single, atomic improvement. "
#         "Your task is to implement this improvement."
#     )
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": code_introduction,
#         "Task description summary and previous solution": task_summary_from_planner,
#         "Improvement plan": {"Plan": improvement_plan_from_planner},
#         "Previous solution code": {"Code": wrap_code(parent_node_code)},
#         "Memory": journal_summary,
#         "Instructions": {
#             "Environment and Packages": get_competition_environment_text(competition_name),
#             "Response format": PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT,
#             "code improvement guideline": [
#                 "You should precisely follow the plan for improvement and implement the code that implements the improvement.",
#                 "The final code should be a single code block, complete, and self-contained.",
#                 "The code should be well documented and easy to understand.",
#                 "Strictly follow the plan for improvement.",
#                 "Take the Memory section into consideration during implementation to avoid bugs.",
#                 "Code should be between ```python fences.",
#                 "Only write code; do not write any other text.",
#             ],
#             "additional guidelines": AGENT_IMPLEMENTATION_GUIDELINE_LIST # Reusing agent's guidelines
#         }
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message

# # For PlannerAgent's _debug stage (plan_query part)
# def get_planner_agent_debug_plan_user_prompt(
#     task_desc: str,
#     parent_node_code: str,
#     parent_node_term_out: str,
#     # competition_name: str, # If planner needs environment for debug plan
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     plan_introduction = (
#         "You are a Kaggle grandmaster AND A TEAM LEADER. "
#         "Your team's previous solution had a bug and/or did not produce a submission.csv. "
#         "Based on the information below, you should first provide a detailed summary of the bugs "
#         "and then outline a detailed step-by-step plan to fix them. This plan will be given to a team member to implement."
#     )
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": plan_introduction,
#         "Task description": task_desc,
#         "Previous (buggy) implementation": wrap_code(parent_node_code),
#         "Execution output": wrap_code(parent_node_term_out, lang=""),
#         "Instructions": {
#             "Response format": PLANNER_AGENT_DEBUG_RESPONSE_FORMAT_TEXT,
#             # Guidelines are embedded in the response format and intro for planner debug plan
#             # "Environment and Packages": get_competition_environment_text(competition_name) # If planner benefits
#         }
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message

# # For PlannerAgent's _debug stage (code_query part)
# def get_planner_agent_debug_code_user_prompt(
#     bug_summary_from_planner: str,
#     fix_plan_from_planner: str,
#     parent_node_code: str,
#     parent_node_term_out: str, # For coder's context
#     competition_name: str,
#     acfg_data_preview: bool,
#     data_preview_content: str = None
# ) -> Dict[str, Any]:
#     code_introduction = (
#         "You are a Kaggle grandmaster AND A TEAM MEMBER. Your previous solution had bugs. "
#         "You are provided with an analysis of the bugs, a detailed plan to fix them, the original buggy code, and its execution output. "
#         "Your task is to implement the bugfix."
#     )
#     prompt_user_message: Dict[str, Any] = {
#         "Introduction": code_introduction,
#         "Problem Description and Analysis": bug_summary_from_planner,
#         "Plan for fixing the bug": fix_plan_from_planner,
#         "Previous (buggy) implementation": wrap_code(parent_node_code),
#         "Execution output of buggy code": wrap_code(parent_node_term_out, lang=""),
#         "Instructions": {
#             "Environment and Packages": get_competition_environment_text(competition_name),
#             "Response format": PLANNER_AGENT_CODE_RESPONSE_FORMAT_TEXT,
#             "Bugfix implementation guideline": [
#                 "Precisely follow the plan for fixing the bugs and implement the code that implements the fix.",
#                 "The final code should be a single code block, complete, and self-contained.",
#             ],
#             "additional guidelines": AGENT_IMPLEMENTATION_GUIDELINE_LIST # Reusing agent's guidelines
#         }
#     }
#     if acfg_data_preview and data_preview_content:
#         prompt_user_message["Data Overview"] = data_preview_content
#     return prompt_user_message

