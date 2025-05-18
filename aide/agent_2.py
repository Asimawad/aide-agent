import shutil
import logging
import random
import time
from rich.syntax import Syntax
from rich.console import Console
from typing import Any, Callable, cast
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.pretty_logging import log_step, logger        
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code,trim_long_string, format_code
from .utils.self_reflection import perform_two_step_reflection  , perform_two_step_reflection_with_fewshot

try:
    import wandb
except ImportError:
    wandb = None


logger = logging.getLogger("aide")  # A separate logger for agent.py



console = Console()
def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


ExecCallbackType = Callable[[str, bool], ExecutionResult]

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


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_run=None,
        competition_benchmarks=None
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        self._prev_buggy: bool = False
        self.wandb_run = wandb_run
        self.competition_benchmarks = competition_benchmarks
        self.competition_name = self.cfg.competition_name

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "1. Write a complete, single-file Python script. ",
            "2. starting with imports, and load necessary data from the './input/' directory.",
            "3. Implement the simple solution proposed in your plan.",
            "4. Calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
            "5. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",
            "6. The script must run without errors. Focus on correctness first.",
        ]
        return {"Implementation Guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        fmt = (
                    "\n\n---\n"
                    "1) PLAN (plain text, no fences):\n"
                    "<your step‑by‑step reasoning here>\n\n"
                    "2) CODE (one fenced Python block):\n"
                    "```python\n"
                    "<your python code here>\n"
                    "```"
                )
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "explicitly,structure your answer exactly like this: ") + fmt
        }
    def plan_and_code_query(self, prompt,excute, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        system_prompt = {
            "SYSTEM":"You are a Kaggle Grandmaster. you can plan , implement, debug and improve and machine learning engineering code,",
            "user_instructions": {
               "Possible Questions you will face": "You will be asked to either come up with a plan and a code to solve the kaggle competetion, or debug a code or improve a working code to get better results",
               "How to answer the user": "Whenever you answer, always: 1. Write a \"PLAN:\" section in plain text—3–5 concise bullet points. 2. Then write a \"CODE:\" section containing exactly one fenced Python block: ```python"
            }
        }

        completion_text = None
        execution_summary= None
        for _ in range(retries):
            if self.cfg.inference_engine == "HF" and self.acfg.code.model != "o3-mini" :
                completion_text = query(
                system_message=system_prompt,
                user_message=prompt,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                max_tokens=self.acfg.code.max_new_tokens,
                top_p=self.acfg.code.top_p,
                top_k=self.acfg.code.top_k,
                excute=excute,
                current_step=self.current_step,
                inference_engine = self.cfg.inference_engine,
                num_responses=self.acfg.code.num_return_sequences,
                convert_system_to_user=self.acfg.convert_system_to_user,
                )
            else:
                completion_text  = query(
                system_message=system_prompt,
                user_message=prompt,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                current_step=self.current_step,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )


            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code , "execution_summary"

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text, "None"  # type: ignore
    # Inside aide-ds/aide/agent.py, within the Agent class

    def _draft(self, parent_node=None) -> Node: # Removed initial_high_level_plan for now, as we focus on templated full draft
        # console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Drafting") # Keep if you like console output here
        logger.info(f"Agent step {self.current_step}: Drafting new solution (parent: {parent_node})")

        # --- Try to get the competition-specific template ---
        comp_data = self.competition_benchmarks

        code_template = None
        if self.competition_benchmarks and self.competition_name and self.cfg.use_template:
            if comp_data and comp_data["template"]:
                code_template = comp_data["template"]
                logger.info(f"Found code template for competition: {self.competition_name}")
            else:
                logger.warning(f"No template found for competition: {self.competition_name} in competition_benchmarks. Proceeding without template.")
        else:
            logger.warning("Competition benchmarks or competition name not available or not enabled. Proceeding without template.")

        # --- Construct the prompt ---
        introduction = (
            "You are a Kaggle grandmaster. Your task is to develop a complete Python script to solve the described machine learning competition."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer. Your task is to develop a complete Python script to solve the described machine learning problem."
            )

        prompt_user_message: Any = {
            "Introduction": introduction,
            "Overall Task Description": self.task_desc, # This is the markdown/text from the competition
            "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(),
            "Instructions": {},
        }

        if code_template:
            prompt_user_message["Code Template to Complete"] = f"```python\n{code_template}\n```"
            prompt_user_message["Instructions"]["Template Guidance"] = [
                "You are provided with a Python code template above. Your primary goal is to complete the sections marked with `{{PLACEHOLDER_NAME}}`.",
                "Specifically, you need to provide Python code for:",
                "  1. `{{FEATURE_ENGINEERING_CODE}}`: Load data as per template, preprocess, and create features necessary for your model.",
                "  2. `{{MODEL_TRAINING_VALIDATION_CODE}}`: Define your model, set up training loops, and perform validation. This section *must* print the primary validation metric in the format: `print(f\"Validation Metric: {your_validation_score:.4f}\")` (adjust precision as needed for the metric).",
                "  3. `{{PREDICTION_CODE}}`: Use your trained model to generate predictions on the test data.",
                "  4. `{{CREATE_FINAL_SUBMISSION_DATAFRAME_CODE}}`: Construct a pandas DataFrame named `final_submission_df`. This DataFrame must strictly follow the submission format specified by the `sample_submission.csv` for this competition (column names, number of rows, ID column).",
                "Do NOT modify the pre-filled data loading sections or the final submission saving logic in the template unless absolutely critical for your approach and you explain why.",
                "Ensure all necessary libraries not already in the template are imported at the beginning of the relevant placeholder or at the top of the script if globally needed.",
                "Focus on creating a robust, runnable first version of the complete solution based on the template."
            ]
            prompt_user_message["Instructions"]["Output Format"] = (
                "Your response should be a brief natural language PLAN (3-5 sentences) outlining your approach to filling the template placeholders, "
                "followed by a SINGLE markdown code block containing the *complete, filled-in Python script* based on the provided template."
                "There should be no additional headings or text in your response. Just the PLAN, a newline, and then the markdown code block."
                "explicitly,structure your answer exactly like this:"
                "\n\n---\n"
                "1) PLAN (plain text, no fences):\n"
                "<your step‑by‑step reasoning for filling the placeholders>\n\n"
                "2) CODE (one fenced Python block):\n"
                "```python\n"
                "<your COMPLETE python code here, with template placeholders filled>\n"
                "```"
            )
        else: # Fallback if no template is found - revert to original _draft prompting style
            prompt_user_message["Instructions"] |= self._prompt_resp_fmt # Original response format
            prompt_user_message["Instructions"] |= { # Original sketch guidelines
                "Solution sketch guideline": [
                    "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                    "Take the Memory section into consideration when proposing the design.",
                    "The solution sketch should be 3-5 sentences.",
                    "Propose an evaluation metric that is reasonable for this task.",
                    "Don't suggest to do EDA.",
                    "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                ],
            }
            prompt_user_message["Instructions"] |= self._prompt_impl_guideline # Original implementation guidelines

        prompt_user_message["Instructions"] |= self._prompt_environment # Common environment prompt

        if self.acfg.data_preview:
            prompt_user_message["Data Overview"] = self.data_preview
        
        # The `plan` from LLM here is its plan for filling the template / solving the task.
        # The `code` is the complete script (either filled template or from scratch).
        agent_plan_for_step, generated_code, execution_summary = self.plan_and_code_query(prompt_user_message, excute=False)
        
        formatted_extracted_code = format_code(generated_code)
        if formatted_extracted_code:
            # console.print(f"[bold green]Extracted a valid Code for step {self.current_step}[/bold green]")
            # console.print(Syntax(formatted_extracted_code, "python", theme="default", line_numbers=True))
            logger.info("Code generated for drafting stage:", extra={"verbose": True}) # General log
            logger.debug(f"{Syntax(formatted_extracted_code, 'python', theme='default', line_numbers=True)}",  extra={"verbose": True}) # Verbose log with code
            # console.print("-" * 60)
        
        new_node = Node(
            plan=agent_plan_for_step, 
            code=generated_code,
            summary=execution_summary, # This field seems not heavily used, but kept for consistency
            # high_level_plan will be None if we are not doing the hierarchical plan for now
            # current_hl_step_index will be None
        )
        # Parent will be set by the caller if this isn't a root draft
        if parent_node:
            new_node.parent = parent_node

        logger.info(f"Drafted new node {new_node.id} (Template used: {bool(code_template)})")
        return new_node


    def _improve(self, parent_node: Node) -> Node:
        console.rule(f"[cyan]Stage : Improving")
        logger.info(f"Agent step {self.current_step}: Generating code (parent type: {parent_node.stage_name})",extra={"verbose": True})
        introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code , _ = self.plan_and_code_query(prompt,excute=False)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        console.rule(f"[cyan]Stage : Debugging")
        logger.info(f"Agent step {self.current_step}: Generating code (parent type: {parent_node.stage_name})", extra={"verbose": True})
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code, _ = self.plan_and_code_query(prompt,excute=False)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def reflect(self, node: Node) -> tuple[str, str]:
        """
        Performs a two-step self-reflection using the external utility function.

        Returns:
            Tuple: (reflection_plan, revised_code)
        """
        logger.info("Initiating two-step self-reflection...")
        reflection_plan, revised_code = perform_two_step_reflection(
            code=node.code,
            analysis=node.analysis,
            term_out=node.term_out,
            task_desc=self.task_desc,
            model_name=self.acfg.code.model,
            temperature=self.acfg.code.temp,
            convert_system_to_user=self.acfg.convert_system_to_user,
            query_func=query,  # 
            wrap_code_func=wrap_code,  # 
            extract_code_func=extract_code,  # 
        )

        if revised_code != node.code and revised_code:  # Check if code actually changed
            logger.info("Self-reflection resulted in code changes.")
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info("Self-reflection found no errors requiring changes.")
        else:
            logger.warning(
                "Self-reflection finished, but revised code is same as original or empty."
            )

        return reflection_plan, revised_code

    def double_reflect(self, code: str) -> tuple[str, str]:
        """
        Performs a two-step self-reflection using the external utility function.

        Returns:
            Tuple: (reflection_plan, revised_code)
        """
        logger.info("Initiating two-step self-reflection...")
        reflection_plan, revised_code = perform_two_step_reflection(
            code=code,
            task_desc=self.task_desc,
            model_name=self.acfg.code.model,
            temperature=self.acfg.code.temp,
            convert_system_to_user=self.acfg.convert_system_to_user,
            query_func=query,  # 
            wrap_code_func=wrap_code,  # 
            extract_code_func=extract_code,  #
        )

        if revised_code != code and revised_code:  # Check if code actually changed
            logger.info("Self-reflection resulted in code changes.")
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info("Self-reflection found no errors requiring changes.")
        else:
            logger.warning(
                "Self-reflection finished, but revised code is same as original or empty."
            )

        return reflection_plan, revised_code
    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)
        logger.info(f"Data preview updated to {self.data_preview}")

    def step(self, exec_callback: ExecCallbackType, current_step_number: int): 

        t0 = time.time()

        # clear the submission dir from previous steps
        submission_dir = self.cfg.workspace_dir / "submission" # Define once
        shutil.rmtree(submission_dir, ignore_errors=True)
        submission_dir.mkdir(exist_ok=True)

        last = time.time()
        self.current_step = current_step_number

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()

        draft_flag = False
        if parent_node is None:
            draft_flag = True
            node_stage = "draft"
            result_node = self._draft(parent_node)
        elif parent_node.is_buggy:
            node_stage = "debug"
            result_node = self._debug(parent_node)

        else:
            node_stage = "improve"
            result_node = self._improve(parent_node)



        logger.info(f"Agent step {current_step_number}: Executing code for node {result_node.id} (stage: {node_stage}")
        exec_start_time = time.time()

        exec_result = exec_callback(
            result_node.code,
            reset_session=True
        )
        # Flag if execution threw any exception
        exec_duration = time.time() - exec_start_time

        # Parse execution result
        logger.info(f"Agent step {current_step_number}: Parsing execution results for node {result_node.id}")

        result_node = self.parse_exec_result(
            node=result_node, exec_result=exec_result,
            )
        self._prev_buggy = result_node.is_buggy

        # Apply reflection if applicable
        reflection_applied = False
        if draft_flag and self.acfg.ITS_Strategy=="self-reflection" and result_node.is_buggy:  
            try:
                console.rule(f"[cyan]Stage : Self Reflection")
                reflection_plan, reflection_code = self.reflect(node=result_node)
                if (
                    reflection_code
                    and reflection_code.strip()
                    and reflection_code != result_node.code
                ):
                    result_node.code = reflection_code
                    logger.info(
                        f"Node {result_node.id} self-reflected and updated code"
                    )
                    reflection_applied = True

                elif reflection_plan != "No specific errors found requiring changes.":
                    logger.info(
                        f"Node {result_node.id} self-reflection completed, but no changes applied."
                    )
                else:
                    logger.info("No errors found by reflection.")
            except Exception as e:
                logger.error(
                    f"Error during self-reflection for node {result_node.id}: {e}",
                    exc_info=True,
                )
        if reflection_applied:
            logger.info(f"Agent is executing the reflect code for node {result_node.id}")
            exec_start_time = time.time()

            exec_result = exec_callback(
                result_node.code,
                reset_session=True
            )
            # Flag if execution threw any exception
            exec_duration = time.time() - exec_start_time

            # Parse execution result
            logger.info(f"Agent step {current_step_number}: Parsing execution results for node {result_node.id}")

            result_node = self.parse_exec_result(
                node=result_node, exec_result=exec_result,
                )

        if self._prev_buggy and not result_node.is_buggy:
            result_node.effective_debug_step = True
            if reflection_applied:
                result_node.effective_reflections = True
            else:
                result_node.effective_reflections = False
        else:
            result_node.effective_debug_step = False
            result_node.effective_reflections = False
        self._prev_buggy = result_node.is_buggy

        step_log_data=({
            f"exec/exec_time_s": exec_duration,
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name":self.competition_name,
            "exec/exception_type": result_node.exc_type if  result_node.exc_type  else 0,
            f"code/estimated_quality":int(self._code_quality),
            f"eval/reflection_usage": 1 if reflection_applied and not result_node.is_buggy else 0,
            f"eval/effective_debug_step": 1 if result_node.effective_debug_step else 0,
            f"eval/effective_reflections": 1 if result_node.effective_reflections else 0,
        })
        if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None:
            step_log_data[f"eval/validation_metric"] = result_node.metric.value
            agent_validation_metrics = {'value': result_node.metric.value, 'step': current_step_number ,
                                         'competition_name': self.competition_name,
                                         "above_median": 1 if result_node.metric.value > self.competition_benchmarks["median_threshold"] else 0,
                                         "gold_medal": 1 if result_node.metric.value > self.competition_benchmarks["gold_threshold"] else 0,
                                         "silver_medal": 1 if result_node.metric.value > self.competition_benchmarks["silver_threshold"] else 0,
                                         "bronze_medal": 1 if result_node.metric.value > self.competition_benchmarks["bronze_threshold"] else 0,
                                         }
            # --- Bar charts for threshold flags ---
            # Above Median
            self._above_median_flags = getattr(self, "_above_median_flags", [])
            self._above_median_flags.append(agent_validation_metrics["above_median"])
            above_true = sum(self._above_median_flags)
            above_false = len(self._above_median_flags) - above_true
            above_table = wandb.Table(
                data=[["Above Median", above_true], ["Below Median", above_false]],
                columns=["label","count"]
            )
            step_log_data["plots/above_median_bar"] = wandb.plot.bar(
                above_table, "label", "count", title="Above Median Steps"
            )
            # Gold Medal
            self._gold_medal_flags = getattr(self, "_gold_medal_flags", [])
            self._gold_medal_flags.append(agent_validation_metrics["gold_medal"])
            gold_true = sum(self._gold_medal_flags)
            gold_false = len(self._gold_medal_flags) - gold_true
            gold_table = wandb.Table(
                data=[["Gold Medal", gold_true], ["No Gold Medal", gold_false]],
                columns=["label","count"]
            )
            step_log_data["plots/gold_medal_bar"] = wandb.plot.bar(
                gold_table, "label", "count", title="Gold Medal Steps"
            )
            # Silver Medal
            self._silver_medal_flags = getattr(self, "_silver_medal_flags", [])
            self._silver_medal_flags.append(agent_validation_metrics["silver_medal"])
            silver_true = sum(self._silver_medal_flags)
            silver_false = len(self._silver_medal_flags) - silver_true
            silver_table = wandb.Table(
                data=[["Silver Medal", silver_true], ["No Silver Medal", silver_false]],
                columns=["label","count"]
            )
            step_log_data["plots/silver_medal_bar"] = wandb.plot.bar(
                silver_table, "label", "count", title="Silver Medal Steps"
            )
            # Bronze Medal
            self._bronze_medal_flags = getattr(self, "_bronze_medal_flags", [])
            self._bronze_medal_flags.append(agent_validation_metrics["bronze_medal"])
            bronze_true = sum(self._bronze_medal_flags)
            bronze_false = len(self._bronze_medal_flags) - bronze_true
            bronze_table = wandb.Table(
                data=[["Bronze Medal", bronze_true], ["No Bronze Medal", bronze_false]],
                columns=["label","count"]
            )
            step_log_data["plots/bronze_medal_bar"] = wandb.plot.bar(
                bronze_table, "label", "count", title="Bronze Medal Steps"
            )
        else:
            step_log_data[f"eval/validation_metric"] = float('nan') # W&B handles NaN well

        # Final check for submission file existence
        submission_path = submission_dir / "submission.csv"
        submission_exists = submission_path.exists()
        if not result_node.is_buggy and not submission_exists:
            result_node.is_buggy = True
            result_node.metric = WorstMetricValue()
            logger.info(
                f"Actually, node {result_node.id} did not produce a submission.csv"
            )
# 
        step_log_data[f"eval/submission_produced"] = 1 if submission_exists else 0



        # --- Histogram of validation metric 
        self._metric_hist = getattr(self, "_metric_hist", [])
        if result_node.metric and result_node.metric.value is not None:
            self._metric_hist.append(result_node.metric.value)

        if len(self._metric_hist) >= 3:          # wait until we have a few points
            tbl = wandb.Table(
                data=[[v] for v in self._metric_hist], columns=["val"]
            )
            step_log_data["plots/val_metric_hist"] = wandb.plot.scatter(
                tbl, "val", "step", title="Validation-metric distribution"
            )

        # Keep a rolling list of 0/1 flags for every step
        self._bug_flags = getattr(self, "_bug_flags", [])
        self._bug_flags.append(1 if result_node.is_buggy else 0)

        bug_count   = sum(self._bug_flags)          
        clean_count = len(self._bug_flags) - bug_count

        bug_table = wandb.Table(
            data=[["Buggy", bug_count], ["Clean", clean_count]],
            columns=["label", "count"],
        )
        step_log_data["plots/bug_vs_clean"] = wandb.plot.bar(
            bug_table, "label", "count", title="Buggy vs clean steps"
        )                                           
        # --- Bar chart: Submission produced vs missing 
        self._sub_flags = getattr(self, "_sub_flags", [])

        self._sub_flags.append(1 if submission_exists else 0)

        with_sub   = sum(self._sub_flags)                 # steps that made a CSV
        without_sub = len(self._sub_flags) - with_sub

        sub_table = wandb.Table(
            data=[["Has submission", with_sub], ["No submission", without_sub]],
            columns=["label", "count"],
        )
        step_log_data["plots/submission_presence"] = wandb.plot.bar(
            sub_table, "label", "count", title="Submission produced vs missing"
        )                                          
 
        # --- Send log data to W&B ---
        if self.wandb_run:
            t_wandb_start = time.time()
            self.wandb_run.log(step_log_data, step=current_step_number)

            last = time.time()
        # --- End Send log data ---
        self.journal.append(result_node)

        # Log best solution artifacts *immediately* when a new best is found
        best_node = self.journal.get_best_node()
        if best_node is not None and best_node.id == result_node.id:
             logger.debug(f"Node {result_node.id} is the best node so far (Metric: {best_node.metric.value:.4f})")
             best_solution_dir = self.cfg.workspace_dir / "best_solution"
             best_submission_dir = self.cfg.workspace_dir / "best_submission"
             best_solution_dir.mkdir(exist_ok=True, parents=True)
             best_submission_dir.mkdir(exist_ok=True, parents=True)

             if submission_exists:
                 shutil.copy(submission_path, best_submission_dir)
             else:
                  logger.warning(f"Best node {result_node.id} did not produce submission.csv, cannot cache/log artifact.")


             # Cache best solution code locally
             best_code_path = best_solution_dir / "solution.py"
             with open(best_code_path, "w") as f:
                 f.write(result_node.code)
             with open(best_solution_dir / "node_id.txt", "w") as f:
                 f.write(str(result_node.id))


        elif best_node:
             logger.debug(f"This Node is not the best node (Best: {best_node.id} with metric {best_node.metric.value:.4f})")
            # …existing code that fills exec_duration / result_node.metric / etc.

        result_node.stage      = node_stage
        result_node.exec_time  = exec_duration

        log_step(
            step   = current_step_number,
            total  = self.acfg.steps,
            stage  = node_stage,
            is_buggy = result_node.is_buggy,
            exec_time = exec_duration,
            metric = (result_node.metric.value
                    if result_node.metric and result_node.metric.value else None),
        )


    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:

        node.absorb_exec_result(exec_result)

        # Original complex prompt
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        )
        if self.acfg.obfuscate:
                introduction = (
                    "You are an expert machine learning engineer attempting a task. "
                    "You have written code to solve this task and now need to evaluate the output of the code execution. "
                    "You should determine if there were any bugs as well as report the empirical findings."
                )

        prompt = {
            "Introduction": introduction,
            "Task Description": self.task_desc, # Provide task context
            "Code Executed": wrap_code(node.code),
            "Execution Output Log": wrap_code(node.term_out, lang=""), # Use raw term_out
        }
        
        # Retry mechanism for the feedback LLM call (optional but good)
        max_retries = 3
        review_response = None
        
        for attempt in range(max_retries):
            try:
                review_response = cast(
                    dict,
                    query(
                        system_message=prompt,
                        user_message=None,
                        func_spec=review_func_spec,
                        model=self.acfg.feedback.model,
                        temperature=self.acfg.feedback.temp,
                        excute = False,
                        convert_system_to_user=self.acfg.convert_system_to_user,
                    ),
                )
                # Check if required keys are present
                if all(k in review_response for k in ["is_bug", "has_csv_submission", "summary", "metric", "lower_is_better","code_quality"]):
                    break # Success
                else:
                    logger.warning(f"Feedback LLM response missing keys (attempt {attempt+1}/{max_retries}). Response: {review_response}")
                    review_response = None # Force retry
            except Exception as e:
                logger.error(f"Error querying feedback LLM (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error("Feedback LLM query failed after multiple retries.")
                    # Handle failure: maybe default to buggy?
                    review_response = {
                         "is_bug": True,
                         "has_csv_submission": False,
                         "summary": "Failed to get feedback from LLM.",
                         "metric": None,
                         "lower_is_better": True, # Default assumption
                         "code_quality": 0,
                    }
                    break 

        # if the metric isn't a float then fill the metric with the worst metric
        metric_value = review_response.get("metric") # Use .get for safety
        if not isinstance(metric_value, (float, int)):
            metric_value = None # Set to None if not a valid number

        self._code_quality = review_response.get("code_quality",0)
        # do an extra check, to catch cases where judge fails
        submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
        has_csv_submission_actual = submission_path.exists()
        has_csv_submission_reported = review_response.get("has_csv_submission", False)


        node.analysis = review_response.get("summary", "Feedback LLM failed.") # Default value
        # Determine buggy status based on multiple factors
        logger.info(f"summary: {node.analysis}")
        node.is_buggy = (
            review_response.get("is_bug", True) # Default to True if key missing
            or node.exc_type is not None
            or metric_value is None # Use the validated metric_value
            or not has_csv_submission_reported # Judge's report
            or not has_csv_submission_actual # Actual file existence
        )

        if node.is_buggy:
            logger.info(
                f"Feedback results: Current Node is buggy."
            )
            # Log reasons for being buggy
            bug_reasons = []
            if review_response.get("is_bug", True): bug_reasons.append("LLM judged buggy") ; bug_reasons.append(review_response.get("summary", "Feedback LLM failed."))
            if node.exc_type is not None: bug_reasons.append(f"Exception ({node.exc_type})")
            if metric_value is None: bug_reasons.append("Metric missing/invalid")
            logger.info(f"Buggy reasons: {'; '.join(bug_reasons)}")

            node.metric = WorstMetricValue()

        else:
            logger.info(f"Feedback results: Current Node is not buggy")
            node.metric = MetricValue(
                metric_value, maximize=not review_response.get("lower_is_better", True) # Default lower is better
            )

        return node
