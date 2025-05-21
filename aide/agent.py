import shutil
import logging
import random
import time
from rich.syntax import Syntax
from rich.console import Console
from typing import Any, Callable, cast, Optional
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.pretty_logging import log_step, logger
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code,
    trim_long_string,
    format_code,
    extract_plan,
    extract_summary,
)
from .utils.self_reflection import (
    perform_two_step_reflection,
)


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
        competition_benchmarks=None,
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
                "explicitly,structure your answer exactly like this: "
            )
            + fmt
        }

    def plan_and_code_query(self, prompt, excute, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        system_prompt = {
            "SYSTEM": "You are a Kaggle Grandmaster. you can plan , implement, debug and improve and machine learning engineering code,",
            "user_instructions": {
                "Possible Questions you will face": "You will be asked to either come up with a plan and a code to solve the kaggle competetion, or debug a code or improve a working code to get better results",
                "How to answer the user": 'Whenever you answer, always: 1. Write a "PLAN:" section in plain text—3–5 concise bullet points. 2. Then write a "CODE:" section containing exactly one fenced Python block: ```python',
            },
        }

        completion_text = None
        execution_summary = None
        for _ in range(retries):
            if self.cfg.inference_engine == "HF" and self.acfg.code.model != "o3-mini":
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
                    inference_engine=self.cfg.inference_engine,
                    num_responses=self.acfg.code.num_return_sequences,
                    convert_system_to_user=self.acfg.convert_system_to_user,
                )
            else:
                completion_text = query(
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
                return nl_text, code, "execution_summary"

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text, "None"  # type: ignore

    # Inside aide-ds/aide/agent.py, within the Agent class

    def _draft(
        self, parent_node=None
    ) -> (
        Node
    ):  # Removed initial_high_level_plan for now, as we focus on templated full draft
        # console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Drafting") # Keep if you like console output here
        logger.info(
            f"Agent step {self.current_step}: Drafting new solution (parent: {parent_node})"
        )

        # --- Try to get the competition-specific template ---
        comp_data = self.competition_benchmarks

        code_template = None
        if (
            self.competition_benchmarks
            and self.competition_name
            and self.cfg.use_template
        ):
            if comp_data and comp_data["template"]:
                code_template = comp_data["template"]
                logger.info(
                    f"Found code template for competition: {self.competition_name}"
                )
            else:
                logger.warning(
                    f"No template found for competition: {self.competition_name} in competition_benchmarks. Proceeding without template."
                )
        else:
            logger.warning(
                "Competition benchmarks or competition name not available or not enabled. Proceeding without template."
            )

        # --- Construct the prompt ---
        introduction = "You are a Kaggle grandmaster. Your task is to develop a complete Python script to solve the described machine learning competition."
        if self.acfg.obfuscate:
            introduction = "You are an expert machine learning engineer. Your task is to develop a complete Python script to solve the described machine learning problem."

        prompt_user_message: Any = {
            "Introduction": introduction,
            "Overall Task Description": self.task_desc,  # This is the markdown/text from the competition
            "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(),
            "Instructions": {},
        }

        if code_template:
            prompt_user_message["Code Template to Complete"] = (
                f"```python\n{code_template}\n```"
            )
            prompt_user_message["Instructions"]["Template Guidance"] = [
                "You are provided with a Python code template above. Your primary goal is to complete the sections marked with `{{PLACEHOLDER_NAME}}`.",
                "Specifically, you need to provide Python code for:",
                "  1. `{{FEATURE_ENGINEERING_CODE}}`: Load data as per template, preprocess, and create features necessary for your model.",
                '  2. `{{MODEL_TRAINING_VALIDATION_CODE}}`: Define your model, set up training loops, and perform validation. This section *must* print the primary validation metric in the format: `print(f"Validation Metric: {your_validation_score:.4f}")` (adjust precision as needed for the metric).',
                "  3. `{{PREDICTION_CODE}}`: Use your trained model to generate predictions on the test data.",
                "  4. `{{CREATE_FINAL_SUBMISSION_DATAFRAME_CODE}}`: Construct a pandas DataFrame named `final_submission_df`. This DataFrame must strictly follow the submission format specified by the `sample_submission.csv` for this competition (column names, number of rows, ID column).",
                "Do NOT modify the pre-filled data loading sections or the final submission saving logic in the template unless absolutely critical for your approach and you explain why.",
                "Ensure all necessary libraries not already in the template are imported at the beginning of the relevant placeholder or at the top of the script if globally needed.",
                "Focus on creating a robust, runnable first version of the complete solution based on the template.",
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
        else:  # Fallback if no template is found - revert to original _draft prompting style
            prompt_user_message[
                "Instructions"
            ] |= self._prompt_resp_fmt  # Original response format
            prompt_user_message["Instructions"] |= {  # Original sketch guidelines
                "Solution sketch guideline": [
                    "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                    "Take the Memory section into consideration when proposing the design.",
                    "The solution sketch should be 3-5 sentences.",
                    "Propose an evaluation metric that is reasonable for this task.",
                    "Don't suggest to do EDA.",
                    "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                ],
            }
            prompt_user_message[
                "Instructions"
            ] |= self._prompt_impl_guideline  # Original implementation guidelines

        prompt_user_message[
            "Instructions"
        ] |= self._prompt_environment  # Common environment prompt

        if self.acfg.data_preview:
            prompt_user_message["Data Overview"] = self.data_preview

        # The `plan` from LLM here is its plan for filling the template / solving the task.
        # The `code` is the complete script (either filled template or from scratch).
        agent_plan_for_step, generated_code, execution_summary = (
            self.plan_and_code_query(prompt_user_message, excute=False)
        )

        formatted_extracted_code = format_code(generated_code)
        if formatted_extracted_code:
            # console.print(f"[bold green]Extracted a valid Code for step {self.current_step}[/bold green]")
            # console.print(Syntax(formatted_extracted_code, "python", theme="default", line_numbers=True))
            logger.info(
                "Code generated for drafting stage:", extra={"verbose": True}
            )  # General log
            logger.debug(
                f"{Syntax(formatted_extracted_code, 'python', theme='default', line_numbers=True)}",
                extra={"verbose": True},
            )  # Verbose log with code
            # console.print("-" * 60)

        new_node = Node(
            plan=agent_plan_for_step,
            code=generated_code,
            summary=execution_summary,  # This field seems not heavily used, but kept for consistency
            # high_level_plan will be None if we are not doing the hierarchical plan for now
            # current_hl_step_index will be None
        )
        # Parent will be set by the caller if this isn't a root draft
        if parent_node:
            new_node.parent = parent_node

        logger.info(
            f"Drafted new node {new_node.id} (Template used: {bool(code_template)})"
        )
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        console.rule(f"[cyan]Stage : Improving")
        logger.info(
            f"Agent step {self.current_step}: Generating code (parent type: {parent_node.stage_name})",
            extra={"verbose": True},
        )
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

        plan, code, _ = self.plan_and_code_query(prompt, excute=False)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        console.rule(f"[cyan]Stage : Debugging")
        logger.info(
            f"Agent step {self.current_step}: Generating code (parent type: {parent_node.stage_name})",
            extra={"verbose": True},
        )
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

        plan, code, _ = self.plan_and_code_query(prompt, excute=False)
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
        submission_dir = self.cfg.workspace_dir / "submission"  # Define once
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

        logger.info(
            f"Agent step {current_step_number}: Executing code for node {result_node.id} (stage: {node_stage}"
        )
        exec_start_time = time.time()

        exec_result = exec_callback(result_node.code, reset_session=True)
        # Flag if execution threw any exception
        exec_duration = time.time() - exec_start_time

        # Parse execution result
        logger.info(
            f"Agent step {current_step_number}: Parsing execution results for node {result_node.id}"
        )

        result_node = self.parse_exec_result(
            node=result_node,
            exec_result=exec_result,
        )
        self._prev_buggy = result_node.is_buggy

        # Apply reflection if applicable
        reflection_applied = False
        if (
            draft_flag
            and self.acfg.ITS_Strategy == "self-reflection"
            and result_node.is_buggy
        ):
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
            logger.info(
                f"Agent is executing the reflect code for node {result_node.id}"
            )
            exec_start_time = time.time()

            exec_result = exec_callback(result_node.code, reset_session=True)
            # Flag if execution threw any exception
            exec_duration = time.time() - exec_start_time

            # Parse execution result
            logger.info(
                f"Agent step {current_step_number}: Parsing execution results for node {result_node.id}"
            )

            result_node = self.parse_exec_result(
                node=result_node,
                exec_result=exec_result,
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

        step_log_data = {
            f"exec/exec_time_s": exec_duration,
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name": self.competition_name,
            "exec/exception_type": result_node.exc_type if result_node.exc_type else 0,
            f"code/estimated_quality": int(self._code_quality),
            f"eval/reflection_usage": (
                1 if reflection_applied and not result_node.is_buggy else 0
            ),
            f"eval/effective_debug_step": 1 if result_node.effective_debug_step else 0,
            f"eval/effective_reflections": (
                1 if result_node.effective_reflections else 0
            ),
        }
        if (
            not result_node.is_buggy
            and result_node.metric
            and result_node.metric.value is not None
        ):
            step_log_data[f"eval/validation_metric"] = result_node.metric.value
            agent_validation_metrics = {
                "value": result_node.metric.value,
                "step": current_step_number,
                "competition_name": self.competition_name,
                "above_median": (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks["median_threshold"]
                    else 0
                ),
                "gold_medal": (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks["gold_threshold"]
                    else 0
                ),
                "silver_medal": (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks["silver_threshold"]
                    else 0
                ),
                "bronze_medal": (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks["bronze_threshold"]
                    else 0
                ),
            }
            # --- Bar charts for threshold flags ---
            # Above Median
            self._above_median_flags = getattr(self, "_above_median_flags", [])
            self._above_median_flags.append(agent_validation_metrics["above_median"])
            above_true = sum(self._above_median_flags)
            above_false = len(self._above_median_flags) - above_true
            above_table = wandb.Table(
                data=[["Above Median", above_true], ["Below Median", above_false]],
                columns=["label", "count"],
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
                columns=["label", "count"],
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
                columns=["label", "count"],
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
                columns=["label", "count"],
            )
            step_log_data["plots/bronze_medal_bar"] = wandb.plot.bar(
                bronze_table, "label", "count", title="Bronze Medal Steps"
            )
        else:
            step_log_data[f"eval/validation_metric"] = float(
                "nan"
            )  # W&B handles NaN well

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

        if len(self._metric_hist) >= 3:  # wait until we have a few points
            tbl = wandb.Table(data=[[v] for v in self._metric_hist], columns=["val"])
            step_log_data["plots/val_metric_hist"] = wandb.plot.scatter(
                tbl, "val", "step", title="Validation-metric distribution"
            )

        # Keep a rolling list of 0/1 flags for every step
        self._bug_flags = getattr(self, "_bug_flags", [])
        self._bug_flags.append(1 if result_node.is_buggy else 0)

        bug_count = sum(self._bug_flags)
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

        with_sub = sum(self._sub_flags)  # steps that made a CSV
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
            logger.debug(
                f"Node {result_node.id} is the best node so far (Metric: {best_node.metric.value:.4f})"
            )
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_submission_dir = self.cfg.workspace_dir / "best_submission"
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            best_submission_dir.mkdir(exist_ok=True, parents=True)

            if submission_exists:
                shutil.copy(submission_path, best_submission_dir)
            else:
                logger.warning(
                    f"Best node {result_node.id} did not produce submission.csv, cannot cache/log artifact."
                )

            # Cache best solution code locally
            best_code_path = best_solution_dir / "solution.py"
            with open(best_code_path, "w") as f:
                f.write(result_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f:
                f.write(str(result_node.id))

        elif best_node:
            logger.debug(
                f"This Node is not the best node (Best: {best_node.id} with metric {best_node.metric.value:.4f})"
            )
        # …existing code that fills exec_duration / result_node.metric / etc.

        result_node.stage = node_stage
        result_node.exec_time = exec_duration

        log_step(
            step=current_step_number,
            total=self.acfg.steps,
            stage=node_stage,
            is_buggy=result_node.is_buggy,
            exec_time=exec_duration,
            metric=(
                result_node.metric.value
                if result_node.metric and result_node.metric.value
                else None
            ),
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
            "Task Description": self.task_desc,  # Provide task context
            "Code Executed": wrap_code(node.code),
            "Execution Output Log": wrap_code(
                node.term_out, lang=""
            ),  # Use raw term_out
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
                        excute=False,
                        convert_system_to_user=self.acfg.convert_system_to_user,
                    ),
                )
                # Check if required keys are present
                if all(
                    k in review_response
                    for k in [
                        "is_bug",
                        "has_csv_submission",
                        "summary",
                        "metric",
                        "lower_is_better",
                        "code_quality",
                    ]
                ):
                    break  # Success
                else:
                    logger.warning(
                        f"Feedback LLM response missing keys (attempt {attempt+1}/{max_retries}). Response: {review_response}"
                    )
                    review_response = None  # Force retry
            except Exception as e:
                logger.error(
                    f"Error querying feedback LLM (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error("Feedback LLM query failed after multiple retries.")

                    review_response = {
                        "is_bug": True,
                        "has_csv_submission": False,
                        "summary": "Failed to get feedback from LLM.",
                        "metric": None,
                        "lower_is_better": True,  # Default assumption
                        "code_quality": 0,
                    }
                    break

        # if the metric isn't a float then fill the metric with the worst metric
        metric_value = review_response.get("metric")  # Use .get for safety
        if not isinstance(metric_value, (float, int)):
            metric_value = None  # Set to None if not a valid number

        self._code_quality = review_response.get("code_quality", 0)
        # do an extra check, to catch cases where judge fails
        submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
        has_csv_submission_actual = submission_path.exists()
        has_csv_submission_reported = review_response.get("has_csv_submission", False)

        node.analysis = review_response.get(
            "summary", "Feedback LLM failed."
        )  # Default value
        # Determine buggy status based on multiple factors
        logger.info(f"summary: {node.analysis}")
        node.is_buggy = (
            review_response.get("is_bug", True)  # Default to True if key missing
            or node.exc_type is not None
            or metric_value is None  # Use the validated metric_value
            or not has_csv_submission_reported  # Judge's report
            or not has_csv_submission_actual  # Actual file existence
        )

        if node.is_buggy:
            logger.info(f"Feedback results: Current Node is buggy.")
            # Log reasons for being buggy
            bug_reasons = []
            if review_response.get("is_bug", True):
                bug_reasons.append("LLM judged buggy")
                bug_reasons.append(
                    review_response.get("summary", "Feedback LLM failed.")
                )
            if node.exc_type is not None:
                bug_reasons.append(f"Exception ({node.exc_type})")
            if metric_value is None:
                bug_reasons.append("Metric missing/invalid")
            logger.info(f"Buggy reasons: {'; '.join(bug_reasons)}")

            node.metric = WorstMetricValue()

        else:
            logger.info(f"Feedback results: Current Node is not buggy")
            node.metric = MetricValue(
                metric_value,
                maximize=not review_response.get(
                    "lower_is_better", True
                ),  # Default lower is better
            )

        return node


#############################################################################
#############################################################################
# -*- coding: utf-8 -*-


class PlannerAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_run=None,
        competition_benchmarks=None,
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
        self._code_quality = 0  # Initialize code quality

        # Initialize W&B plot data lists
        self._metric_hist = []
        self._bug_flags = []
        self._sub_flags = []
        self._above_median_flags = []
        self._gold_medal_flags = []
        self._silver_medal_flags = []
        self._bronze_medal_flags = []

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search
        logger.info("[search_policy] Determining next action.", extra={"verbose": True})

        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info(
                "[search_policy] Selected: Draft new node (not enough drafts).",
                extra={"verbose": True},
            )
            return None

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(
                    f"[search_policy] Selected: Debug node {node_to_debug.id}.",
                    extra={"verbose": True},
                )
                return node_to_debug
            else:
                logger.info(
                    "[search_policy] Attempted debug, but no debuggable nodes found.",
                    extra={"verbose": True},
                )

        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info(
                "[search_policy] Selected: Draft new node (no good nodes to improve).",
                extra={"verbose": True},
            )
            return None

        greedy_node = self.journal.get_best_node()
        if greedy_node:  # Ensure greedy_node is not None
            logger.info(
                f"[search_policy] Selected: Improve greedy node {greedy_node.id}.",
                extra={"verbose": True},
            )
            return greedy_node
        else:  # Should ideally not happen if good_nodes exist, but as a fallback
            logger.info(
                "[search_policy] Selected: Draft new node (no best node found, fallback).",
                extra={"verbose": True},
            )
            return None

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
        return {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }

    @property
    def _prompt_impl_guideline(self):
        return {
            "Implementation Guideline": [
                "1. Write a complete, single-file Python script. ",
                "2. starting with imports, and load necessary data from the './input/' directory.",
                "3. Implement the solution proposed in the plan.",
                "4. Calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
                "5. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",
                "6. The script must run without errors. Focus on correctness first.",
                "7. The code should be clean and easy to understand. It should be well-documented and well-structured.",
            ]
        }

    @property
    def _prompt_resp_fmt(
        self,
    ):  # This seems unused now with specific plan/code/debug formats
        fmt = (
            "\n\n---\n"
            "1) PLAN (plain text, no fences):\n"
            "<your step‑by‑step reasoning here>\n\n"
            "2) CODE (one fenced Python block):\n"
            "```python\n<your python code here>\n```"
        )
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "explicitly,structure your answer exactly like this: "
            )
            + fmt
        }

    @property
    def debug_prompt_resp_fmt(self):
        fmt = (
            "\n\n---\n"
            "## Bugs Summary/Analysis: (plain text, no fences):\n"
            "<your step‑by‑step reasoning abd summary of the bugs in the previous solution here>\n\n"
            "## Plan: (plain text, no fences):\n"
            "<your step‑by‑step reasoning and plan steps for fixing the bugs here>\n\n"
        )
        return {
            "Response format": (
                "Your response for the summary should be a detailed and high quality bullet points of the bugs in the previous solution, summarizing all the information and problems(5-7 sentences), "
                "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
                "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Bugs Summary/Analysis: and natural language text (plan) under ## Plan: "
                "explicitly,structure your answer exactly like this: "
            )
            + fmt
        }

    @property
    def code_prompt_resp_fmt(self):
        fmt = (
            "\n\n---\n"
            "1) CODE (one fenced Python block):\n"
            "```python\n<your python code here>\n```"
        )
        return {
            "Response format": (
                "Your response should be a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just the markdown code block. "
                "explicitly,structure your answer exactly like this: "
            )
            + fmt
        }

    @property
    def plan_prompt_resp_fmt(self):
        fmt = (
            "\n\n---\n"
            "## Task Summary: (plain text, no fences):\n"
            "<your step‑by‑step reasoning abd summary of the task here>\n\n"
            "## Plan: (plain text, no fences):\n"
            "<your step‑by‑step reasoning and plan steps here>\n\n"
        )
        return {
            "Response format": (
                "Your response for the summary should be a detailed and high quality bullet points of what the task is about, summarizing all the information in the task description (5-7 sentences), "
                "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
                "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Task Summary: and natural language text (plan) under ## Plan: "
                "explicitly,structure your answer exactly like this: "
            )
            + fmt
        }

    def _query_llm_with_retries(
        self,
        query_type: str,
        system_prompt: Any,
        user_prompt: Any,
        model: str,
        temperature: float,
        planner_flag: bool,
        current_step: int,
        convert_system_to_user: bool,
        retries: int = 3,
    ) -> Any:
        """Helper function to query LLM with retries and detailed logging."""
        completion_text = None
        log_prefix = f"LLM_QUERY_{query_type.upper()}_STEP{current_step}"

        for attempt in range(retries):
            logger.info(
                f"{log_prefix}_ATTEMPT{attempt+1}: Sending request. Model: {model}, Temp: {temperature}, Planner: {planner_flag}",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_ATTEMPT{attempt+1}_SYSTEM_PROMPT_START\n{system_prompt}\n{log_prefix}_ATTEMPT{attempt+1}_SYSTEM_PROMPT_END",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_ATTEMPT{attempt+1}_USER_PROMPT_START\n{user_prompt}\n{log_prefix}_ATTEMPT{attempt+1}_USER_PROMPT_END",
                extra={"verbose": True},
            )

            try:
                completion_text = query(
                    system_message=system_prompt,
                    user_message=user_prompt,
                    model=model,
                    planner=planner_flag,
                    temperature=temperature,
                    current_step=current_step,  # Pass current_step for backend logging
                    convert_system_to_user=convert_system_to_user,
                )
                logger.info(
                    f"{log_prefix}_ATTEMPT{attempt+1}: Received response.",
                    extra={"verbose": True},
                )
                logger.debug(
                    f"{log_prefix}_ATTEMPT{attempt+1}_RAW_RESPONSE_START\n{completion_text}\n{log_prefix}_ATTEMPT{attempt+1}_RAW_RESPONSE_END",
                    extra={"verbose": True},
                )
                return completion_text  # Return on successful query
            except Exception as e:
                logger.error(
                    f"{log_prefix}_ATTEMPT{attempt+1}: Error during LLM query: {e}",
                    exc_info=True,
                    extra={"verbose": True},
                )
                if attempt == retries - 1:
                    logger.error(
                        f"{log_prefix}: All {retries} retries failed.",
                        extra={"verbose": True},
                    )
                    return None  # Or raise the exception e
        return (
            None  # Should be unreachable if retries > 0 and an exception isn't raised
        )

    def plan_query(self, prompt_user_message: Any, retries=3) -> tuple[str, str, str]:
        """Generate a step by step natural language plan that will be fed to the coder model."""
        system_prompt = {
            "SYSTEM": "You are a Kaggle Grandmaster and a team leader. you can plan high detailed and quality machine learning engineering solutions,",
            "user_instructions": {
                "Possible Questions you will face": "You will be asked to come up with a step by step plan to solve the kaggle competetion",
                "How to answer the user": 'Whenever you answer, always: 1. Write a "## Task Summary:" section in plain text consisting of 5-7 sentences distilling the task for you team members that are responsible for implementing the solution. 2. Write a "## Plan:" section in plain text consisting of detailed and high quality bullet points that will be used by the team members to implement the solution (7-10 bullet points). ',
                "Critical Instructions": "Do not give/write code solutions, coding is not your job, just consice summary and detailed plan",
            },
        }
        log_prefix = f"LLM_PLAN_QUERY_STEP{self.current_step}"

        completion_text = self._query_llm_with_retries(
            query_type="PLANNER",
            system_prompt=system_prompt,
            user_prompt=prompt_user_message,
            model=self.acfg.code.planner_model,
            temperature=self.acfg.code.temp,
            planner_flag=True,
            current_step=self.current_step,
            convert_system_to_user=self.acfg.convert_system_to_user,
            retries=retries,
        )

        if completion_text is None:
            logger.error(
                f"{log_prefix}: Failed to get response after retries. Returning empty plan and summary.",
                extra={"verbose": True},
            )
            return "", "", ""

        summary = extract_summary(completion_text)
        plan = extract_plan(completion_text)

        if plan and summary:
            logger.info(
                f"{log_prefix}: Successfully extracted plan and summary.",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_EXTRACTED_SUMMARY_START\n{summary}\n{log_prefix}_EXTRACTED_SUMMARY_END",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_EXTRACTED_PLAN_START\n{plan}\n{log_prefix}_EXTRACTED_PLAN_END",
                extra={"verbose": True},
            )
            return summary, plan, ""
        else:
            logger.warning(
                f"{log_prefix}: Plan or summary extraction failed. Raw text: '{trim_long_string(completion_text if isinstance(completion_text, str) else str(completion_text))}'",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_EXTRACTION_FAILED_RAW_COMPLETION_START\n{completion_text}\n{log_prefix}_EXTRACTION_FAILED_RAW_COMPLETION_END",
                extra={"verbose": True},
            )
            return (
                "",
                (
                    completion_text
                    if isinstance(completion_text, str)
                    else str(completion_text)
                ),
                "",
            )  # Return raw text as plan if extraction fails

    def code_query(self, prompt_user_message: Any, retries=3) -> tuple[str, str, str]:
        """Follow a predefined plan and implement the code that solves the kaggle competetion."""
        system_prompt = {
            "SYSTEM": "You are a Kaggle Grandmaster and great at implementing machine learning engineering code. Precisely follow the plan to implement the code that solves the kaggle competetion.",
            "user_instructions": {
                "What you will face": "You will be given a plan to implement the code that solves the kaggle competetion. Precisely follow the plan to implement the code.",
                "How to answer the user": 'Whenever you answer, always: answer in one section called "CODE:" containing exactly one fenced Python block: ```python implementing the plan',
            },
        }
        log_prefix = f"LLM_CODE_QUERY_STEP{self.current_step}"

        completion_text = self._query_llm_with_retries(
            query_type="CODER",
            system_prompt=system_prompt,
            user_prompt=prompt_user_message,
            model=self.acfg.code.model,
            temperature=self.acfg.code.temp,
            planner_flag=False,  # Coder model is not planner
            current_step=self.current_step,
            convert_system_to_user=self.acfg.convert_system_to_user,
            retries=retries,
        )

        if completion_text is None:
            logger.error(
                f"{log_prefix}: Failed to get response after retries. Returning empty code.",
                extra={"verbose": True},
            )
            return "", "", ""

        code = extract_code(completion_text)
        # nl_text = extract_text_up_to_code(completion_text) # Often empty with current prompts

        if code:
            logger.info(
                f"{log_prefix}: Successfully extracted code.", extra={"verbose": True}
            )
            # logger.debug(f"{log_prefix}_EXTRACTED_NL_TEXT_START\n{nl_text}\n{log_prefix}_EXTRACTED_NL_TEXT_END", extra={"verbose": True})
            logger.debug(
                f"{log_prefix}_EXTRACTED_CODE_START\n{code}\n{log_prefix}_EXTRACTED_CODE_END",
                extra={"verbose": True},
            )
            return "", code, ""  # Assuming nl_text is not critical here
        else:
            logger.warning(
                f"{log_prefix}: Code extraction failed. Raw text: '{trim_long_string(completion_text if isinstance(completion_text, str) else str(completion_text))}'",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_EXTRACTION_FAILED_RAW_COMPLETION_START\n{completion_text}\n{log_prefix}_EXTRACTION_FAILED_RAW_COMPLETION_END",
                extra={"verbose": True},
            )
            # Return raw text as code if extraction fails, might be useful for debugging LLM output format
            return (
                "",
                (
                    completion_text
                    if isinstance(completion_text, str)
                    else str(completion_text)
                ),
                "",
            )

    def _draft(self, parent_node=None) -> Node:
        # console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Drafting")
        logger.info(
            f"AGENT_DRAFT_STEP{self.current_step}: Starting drafting process. Parent: {parent_node.id if parent_node else 'None'}",
            extra={"verbose": True},
        )

        comp_data = self.competition_benchmarks
        code_template = None  # Not used in current logic based on prompt construction

        plan_introduction = f"Given the following task description for a machine learning competition named {self.competition_name}, develop a complete and detailed plan to solve it."
        code_introduction = f"Given the following task description about a machine learning competition named {self.competition_name}, and the plan to solve it, develop a complete code to solve it."

        # --- Plan Generation ---
        plan_prompt_user_message: Any = {
            "Introduction": plan_introduction,
            "Overall Task Description": self.task_desc,
            "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(),
            "Instructions": {},
        }
        plan_prompt_user_message["Instructions"] |= self.plan_prompt_resp_fmt
        plan_prompt_user_message["Instructions"] |= {
            "Solution plan guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization, as we are using this as a first draft for future improvements.",
                "The summary should be 5-7 sentences that describe the task in a nutshell, so that the team members can understand the task and the plan.",
                "Take the Memory section into consideration when proposing the design.",
                "The solution plan should be detailed and high quality bullet points that are easy to follow.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        plan_prompt_user_message["Instructions"] |= self._prompt_environment
        if self.acfg.data_preview:
            plan_prompt_user_message["Data Overview"] = self.data_preview

        logger.info(
            f"AGENT_DRAFT_STEP{self.current_step}: Calling plan_query for drafting.",
            extra={"verbose": True},
        )
        agent_summary_for_step, agent_plan_for_step, _ = self.plan_query(
            plan_prompt_user_message
        )

        if not agent_plan_for_step:  # If plan generation failed
            logger.error(
                f"AGENT_DRAFT_STEP{self.current_step}: Plan generation failed. Cannot proceed with code generation.",
                extra={"verbose": True},
            )
            # Create a dummy node indicating failure
            return Node(
                plan="PLAN GENERATION FAILED",
                code="# PLAN GENERATION FAILED",
                summary=agent_summary_for_step or "PLAN GENERATION FAILED",
                parent=parent_node,
            )

        # --- Code Generation ---
        code_prompt_user_message: Any = {
            "Introduction": code_introduction,
            "Overall Task Description": agent_summary_for_step,  # Use summary from planning
            "Plan to implement": agent_plan_for_step,  # Crucial: Add the generated plan here
            "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(),
            "Instructions": {},
        }
        code_prompt_user_message["Instructions"] |= self._prompt_environment
        code_prompt_user_message["Instructions"] |= {
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
        }
        code_prompt_user_message["Instructions"] |= self.code_prompt_resp_fmt
        if self.acfg.data_preview:
            code_prompt_user_message["Data Overview"] = self.data_preview

        logger.info(
            f"AGENT_DRAFT_STEP{self.current_step}: Calling code_query for drafting.",
            extra={"verbose": True},
        )
        _, generated_code, _ = self.code_query(code_prompt_user_message)

        # Log formatted code for easier reading in verbose logs if successful
        formatted_extracted_code = format_code(generated_code)
        if formatted_extracted_code:
            logger.debug(
                f"AGENT_DRAFT_STEP{self.current_step}_FORMATTED_GENERATED_CODE_START\n{formatted_extracted_code}\nAGENT_DRAFT_STEP{self.current_step}_FORMATTED_GENERATED_CODE_END",
                extra={"verbose": True},
            )
        else:
            logger.warning(
                f"AGENT_DRAFT_STEP{self.current_step}: Code formatting failed for generated code.",
                extra={"verbose": True},
            )

        new_node = Node(
            plan=agent_plan_for_step,
            code=generated_code,
            summary=agent_summary_for_step,
            parent=parent_node,
        )
        logger.info(
            f"AGENT_DRAFT_STEP{self.current_step}: Drafted new node {new_node.id}.",
            extra={"verbose": True},
        )
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        # console.rule(f"[cyan]Stage : Improving")
        logger.info(
            f"AGENT_IMPROVE_STEP{self.current_step}: Starting improvement process for node {parent_node.id}.",
            extra={"verbose": True},
        )

        planner_introduction = "You are a Kaggle grandmaster ... improve it ... summarize the task, and outline your proposed improvement ... then outline a high quality and detailed step by step plan ..."  # Truncated for brevity
        code_introduction = "You are an expert machine learning engineer ... implement the improvement ..."  # Truncated

        # --- Plan Generation for Improvement ---
        plan_prompt_user_message: Any = {
            "Introduction": planner_introduction,
            "Overall Task Description": self.task_desc,
            "Instructions": {},
            "Previous solution": {"Code": wrap_code(parent_node.code)},
            # "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(), # Consider if needed for plan
        }
        plan_prompt_user_message["Instructions"] |= self.plan_prompt_resp_fmt
        plan_prompt_user_message["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "You should provide a summary of the task description and the previous solution and then outline a high quality and detailed step by step plan in natural language for how the solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                # "Take the Memory section into consideration when proposing the improvement.", # Re-added
            ],
        }
        # Add environment if planner model might benefit
        # plan_prompt_user_message["Instructions"] |= self._prompt_environment
        if (
            self.acfg.data_preview
        ):  # If planner needs data context for improvement ideas
            plan_prompt_user_message["Data Overview"] = self.data_preview

        logger.info(
            f"AGENT_IMPROVE_STEP{self.current_step}: Calling plan_query for improvement.",
            extra={"verbose": True},
        )
        agent_summary_for_step, agent_plan_for_step, _ = self.plan_query(
            plan_prompt_user_message
        )

        if not agent_plan_for_step:
            logger.error(
                f"AGENT_IMPROVE_STEP{self.current_step}: Improvement plan generation failed for node {parent_node.id}.",
                extra={"verbose": True},
            )
            return Node(
                plan="IMPROVEMENT PLAN GENERATION FAILED",
                code=parent_node.code,
                summary=agent_summary_for_step or "IMPROVEMENT PLAN FAILED",
                parent=parent_node,
            )

        # --- Code Generation for Improvement ---
        code_prompt_user_message: Any = (
            {  # Renamed from 'prompt' to 'code_prompt_user_message'
                "Introduction": code_introduction,
                "Task description summary and previous solution": agent_summary_for_step,  # Use summary from planning
                "Improvement plan": {
                    "Plan": agent_plan_for_step
                },  # Crucial: Add the generated plan
                "Previous solution code": {
                    "Code": wrap_code(parent_node.code)
                },  # Keep original code for context
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
        )
        code_prompt_user_message["Instructions"] |= self._prompt_environment
        code_prompt_user_message["Instructions"] |= self.code_prompt_resp_fmt
        code_prompt_user_message["Instructions"] |= {
            "code improvement guideline": [
                "You should precisely follow the plan for improvement and implement the code that implements the improvement.",
                "The final code should be a single code block, complete, and self-contained.",
                "The code should be well documented and easy to understand.",
                "Strictly follow the plan for improvement.",
                "Take the Memory section into consideration during implementation to avoid bugs.",
                "Code should be between ```python fences.",
                "Only write code; do not write any other text.",
            ],
            "additional guidelines": self._prompt_impl_guideline[
                "Implementation Guideline"
            ],  # Reuse
        }
        if self.acfg.data_preview:
            code_prompt_user_message["Data Overview"] = self.data_preview

        logger.info(
            f"AGENT_IMPROVE_STEP{self.current_step}: Calling code_query for improvement.",
            extra={"verbose": True},
        )
        _, generated_code, _ = self.code_query(code_prompt_user_message)

        formatted_extracted_code = format_code(generated_code)
        if formatted_extracted_code:
            logger.debug(
                f"AGENT_IMPROVE_STEP{self.current_step}_FORMATTED_GENERATED_CODE_START\n{formatted_extracted_code}\nAGENT_IMPROVE_STEP{self.current_step}_FORMATTED_GENERATED_CODE_END",
                extra={"verbose": True},
            )

        new_node = Node(
            plan=agent_plan_for_step,
            code=generated_code,
            summary=agent_summary_for_step,
            parent=parent_node,
        )
        logger.info(
            f"AGENT_IMPROVE_STEP{self.current_step}: Improved node {parent_node.id} to create new node {new_node.id}.",
            extra={"verbose": True},
        )
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        # console.rule(f"[cyan]Stage : Debugging")
        logger.info(
            f"AGENT_DEBUG_STEP{self.current_step}: Starting debugging process for node {parent_node.id}.",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_DEBUG_STEP{self.current_step}_PARENT_CODE_START\n{parent_node.code}\nAGENT_DEBUG_STEP{self.current_step}_PARENT_CODE_END",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_DEBUG_STEP{self.current_step}_PARENT_TERM_OUT_START\n{parent_node.term_out}\nAGENT_DEBUG_STEP{self.current_step}_PARENT_TERM_OUT_END",
            extra={"verbose": True},
        )

        plan_introduction = "You are a Kaggle grandmaster AND A TEAM LEADER. ... revise it in order to fix this. Your response should be a summary ... followed by a detailed plan ..."  # Truncated
        code_introduction = "You are a Kaggle grandmaster AND A TEAM MEMBER. ... implement the bugfix/solution ..."  # Truncated

        # --- Plan Generation for Debugging ---
        plan_prompt_user_message: Any = {  # Renamed from plan_prompt
            "Introduction": plan_introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        plan_prompt_user_message["Instructions"] |= self.debug_prompt_resp_fmt
        if self.acfg.data_preview:
            plan_prompt_user_message["Data Overview"] = self.data_preview
        # Environment might be useful if bugs are env-related
        # plan_prompt_user_message["Instructions"] |= self._prompt_environment

        logger.info(
            f"AGENT_DEBUG_STEP{self.current_step}: Calling plan_query for debugging.",
            extra={"verbose": True},
        )
        agent_summary_for_step, agent_plan_for_step, _ = self.plan_query(
            plan_prompt_user_message
        )

        if not agent_plan_for_step:
            logger.error(
                f"AGENT_DEBUG_STEP{self.current_step}: Debug plan generation failed for node {parent_node.id}.",
                extra={"verbose": True},
            )
            return Node(
                plan="DEBUG PLAN GENERATION FAILED",
                code=parent_node.code,
                summary=agent_summary_for_step or "DEBUG PLAN FAILED",
                parent=parent_node,
            )

        # --- Code Generation for Debugging ---
        code_prompt_user_message: Any = {  # Renamed from code_prompt
            "Introduction": code_introduction,
            "Problem Description and Analysis": agent_summary_for_step,  # Use summary from planning
            "Plan for fixing the bug": agent_plan_for_step,  # Crucial: Add the plan
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output of buggy code": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        code_prompt_user_message["Instructions"] |= self._prompt_environment
        code_prompt_user_message["Instructions"] |= self.code_prompt_resp_fmt
        code_prompt_user_message["Instructions"] |= {
            "Bugfix implementation guideline": [  # Changed from sketch guideline
                "Precisely follow the plan for fixing the bugs and implement the code that implements the fix.",
                "The final code should be a single code block, complete, and self-contained.",
                # "Take the Memory section into consideration during implementation to avoid repeating previous bugs.", # If memory is useful for debugging
            ],
            "additional guidelines": self._prompt_impl_guideline[
                "Implementation Guideline"
            ],
        }
        if self.acfg.data_preview:
            code_prompt_user_message["Data Overview"] = self.data_preview

        logger.info(
            f"AGENT_DEBUG_STEP{self.current_step}: Calling code_query for debugging.",
            extra={"verbose": True},
        )
        _, generated_code, _ = self.code_query(code_prompt_user_message)

        formatted_extracted_code = format_code(generated_code)
        if formatted_extracted_code:
            logger.debug(
                f"AGENT_DEBUG_STEP{self.current_step}_FORMATTED_GENERATED_CODE_START\n{formatted_extracted_code}\nAGENT_DEBUG_STEP{self.current_step}_FORMATTED_GENERATED_CODE_END",
                extra={"verbose": True},
            )

        new_node = Node(
            plan=agent_plan_for_step,
            code=generated_code,
            summary=agent_summary_for_step,
            parent=parent_node,
        )
        logger.info(
            f"AGENT_DEBUG_STEP{self.current_step}: Debugged node {parent_node.id} to create new node {new_node.id}.",
            extra={"verbose": True},
        )
        return new_node

    def reflect(self, node: Node) -> tuple[str, str]:
        logger.info(
            f"AGENT_REFLECT_STEP{self.current_step}: Initiating self-reflection for node {node.id}.",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_REFLECT_STEP{self.current_step}_NODE_CODE_START\n{node.code}\nAGENT_REFLECT_STEP{self.current_step}_NODE_CODE_END",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_REFLECT_STEP{self.current_step}_NODE_ANALYSIS_START\n{node.analysis}\nAGENT_REFLECT_STEP{self.current_step}_NODE_ANALYSIS_END",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_REFLECT_STEP{self.current_step}_NODE_TERM_OUT_START\n{node.term_out}\nAGENT_REFLECT_STEP{self.current_step}_NODE_TERM_OUT_END",
            extra={"verbose": True},
        )

        # The perform_two_step_reflection function itself needs to be instrumented
        # with verbose logging if you want to see its internal LLM calls.
        # For now, we log what we pass to it and what it returns.
        try:
            reflection_plan, revised_code = perform_two_step_reflection(
                code=node.code,
                analysis=node.analysis,  # Make sure this is the textual analysis string
                term_out=node.term_out,
                task_desc=self.task_desc,
                model_name=self.acfg.code.model,  # Which model for reflection? Planner or Coder?
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query,  # Pass the master query
                wrap_code_func=wrap_code,
                extract_code_func=extract_code,
                current_step=self.current_step,  # Add this to the function signature
                # ADD current_step for logging within perform_two_step_reflection if modified
                # current_step = self.current_step
            )
            logger.info(
                f"AGENT_REFLECT_STEP{self.current_step}: Self-reflection completed.",
                extra={"verbose": True},
            )
            logger.debug(
                f"AGENT_REFLECT_STEP{self.current_step}_REFLECTION_PLAN_START\n{reflection_plan}\nAGENT_REFLECT_STEP{self.current_step}_REFLECTION_PLAN_END",
                extra={"verbose": True},
            )
            logger.debug(
                f"AGENT_REFLECT_STEP{self.current_step}_REVISED_CODE_START\n{revised_code}\nAGENT_REFLECT_STEP{self.current_step}_REVISED_CODE_END",
                extra={"verbose": True},
            )

        except Exception as e:
            logger.error(
                f"AGENT_REFLECT_STEP{self.current_step}: Error during self-reflection: {e}",
                exc_info=True,
                extra={"verbose": True},
            )
            return (
                "REFLECTION_ERROR: " + str(e),
                node.code,
            )  # Return original code on error

        if revised_code and revised_code.strip() and revised_code != node.code:
            logger.info(
                f"AGENT_REFLECT_STEP{self.current_step}: Self-reflection resulted in code changes.",
                extra={"verbose": True},
            )
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info(
                f"AGENT_REFLECT_STEP{self.current_step}: Self-reflection found no errors requiring changes.",
                extra={"verbose": True},
            )
        else:
            logger.warning(
                f"AGENT_REFLECT_STEP{self.current_step}: Self-reflection finished, but revised code is same as original or empty. Plan: {reflection_plan}",
                extra={"verbose": True},
            )
        return reflection_plan, revised_code

    # double_reflect seems unused, if it is, instrument it similarly to reflect()

    def summarize_task(self, task_desc: str) -> str:
        log_prefix = f"LLM_SUMMARIZE_TASK_STEP{self.current_step}"
        logger.info(
            f"{log_prefix}: Summarizing task description.", extra={"verbose": True}
        )

        system_prompt = {
            "SYSTEM": "You are an expert summarization assistant. ... focus on the goal, evaluation metric, dataset ..."  # Truncated
        }
        user_prompt = {
            "Task Description": task_desc,
            "Instructions": (
                "Please provide a concise summary ... Limit the summary to around 7 sentences."
            ),  # Truncated
        }

        summary_completion = self._query_llm_with_retries(
            query_type="SUMMARIZER",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.acfg.code.planner_model,  # Use planner model for summarization
            temperature=0.3,
            planner_flag=True,  # Treat as a planning-like task
            current_step=self.current_step,
            convert_system_to_user=self.acfg.convert_system_to_user,
        )

        if summary_completion is None:
            logger.error(
                f"{log_prefix}: Failed to get summary after retries. Returning original task description.",
                extra={"verbose": True},
            )
            return task_desc  # Fallback

        # from .utils.response import extract_summary # Already imported
        concise_summary = extract_summary(summary_completion) or summary_completion
        logger.info(f"{log_prefix}: Task summarized.", extra={"verbose": True})
        logger.debug(
            f"{log_prefix}_CONCISE_SUMMARY_START\n{concise_summary.strip()}\n{log_prefix}_CONCISE_SUMMARY_END",
            extra={"verbose": True},
        )
        return concise_summary.strip()

    def update_data_preview(self):
        logger.info(
            f"AGENT_STEP{self.current_step}: Updating data preview.",
            extra={"verbose": True},
        )
        try:
            self.data_preview = data_preview.generate(self.cfg.workspace_dir)
            logger.info(
                f"AGENT_STEP{self.current_step}: Data preview updated.",
                extra={"verbose": True},
            )
            logger.debug(
                f"AGENT_STEP{self.current_step}_DATA_PREVIEW_START\n{self.data_preview}\nAGENT_STEP{self.current_step}_DATA_PREVIEW_END",
                extra={"verbose": True},
            )
        except Exception as e:
            logger.error(
                f"AGENT_STEP{self.current_step}: Failed to update data preview: {e}",
                exc_info=True,
                extra={"verbose": True},
            )
            self.data_preview = "Error generating data preview."

    def step(self, exec_callback: ExecCallbackType, current_step_number: int):
        logger.info(
            f"AGENT_STEP_START: {current_step_number}, Total Steps: {self.acfg.steps}",
            extra={"verbose": True},
        )
        t_step_start = time.time()

        submission_dir = self.cfg.workspace_dir / "submission"
        logger.info(
            f"AGENT_STEP{current_step_number}: Clearing submission directory: {submission_dir}",
            extra={"verbose": True},
        )
        shutil.rmtree(submission_dir, ignore_errors=True)
        submission_dir.mkdir(exist_ok=True)

        self.current_step = current_step_number

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
        # Consider if task summarization should happen only once or under certain conditions
        # if (
        #     self.journal.task_summary is None and self.cfg.goal is None
        # ):  # Only summarize if not already done
        #     logger.info(
        #         f"AGENT_STEP{current_step_number}: Task summary not found in journal, generating new one.",
        #         extra={"verbose": True},
        #     )
        #     self.journal.task_summary = self.summarize_task(self.task_desc)
        #     self.task_desc = (
        #         self.journal.task_summary
        #     )  # Update internal task_desc to summarized version

        logger.info(
            f"AGENT_STEP{current_step_number}: Calling search_policy.",
            extra={"verbose": True},
        )
        parent_node = self.search_policy()
        result_node: Node  # Type hint

        draft_flag = False
        if parent_node is None:
            draft_flag = True
            node_stage = "draft"
            logger.info(
                f"AGENT_STEP{current_step_number}: Stage selected: DRAFTING.",
                extra={"verbose": True},
            )
            result_node = self._draft(parent_node)
        elif parent_node.is_buggy:
            node_stage = "debug"
            logger.info(
                f"AGENT_STEP{current_step_number}: Stage selected: DEBUGGING node {parent_node.id}.",
                extra={"verbose": True},
            )
            result_node = self._debug(parent_node)
        else:
            node_stage = "improve"
            logger.info(
                f"AGENT_STEP{current_step_number}: Stage selected: IMPROVING node {parent_node.id}.",
                extra={"verbose": True},
            )
            result_node = self._improve(parent_node)

        logger.info(
            f"AGENT_STEP{current_step_number}: Executing code for node {result_node.id} (stage: {node_stage}). Code length: {len(result_node.code)}",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_STEP{current_step_number}_CODE_TO_EXECUTE_START\n{result_node.code}\nAGENT_STEP{current_step_number}_CODE_TO_EXECUTE_END",
            extra={"verbose": True},
        )
        exec_start_time = time.time()
        exec_result = exec_callback(result_node.code, reset_session=True)
        exec_duration = time.time() - exec_start_time
        logger.info(
            f"AGENT_STEP{current_step_number}: Code execution finished in {exec_duration:.2f}s. Success: {exec_result.term_out}",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_STEP{current_step_number}_EXEC_RESULT_STDOUT_START\n{exec_result.term_out}\nAGENT_STEP{current_step_number}_EXEC_RESULT_STDOUT_END",
            extra={"verbose": True},
        )
        logger.debug(
            f"AGENT_STEP{current_step_number}_EXEC_RESULT_STDERR_START\n{exec_result.term_out}\nAGENT_STEP{current_step_number}_EXEC_RESULT_STDERR_END",
            extra={"verbose": True},
        )

        logger.info(
            f"AGENT_STEP{current_step_number}: Parsing execution results for node {result_node.id}.",
            extra={"verbose": True},
        )
        result_node = self.parse_exec_result(node=result_node, exec_result=exec_result)
        self._prev_buggy = result_node.is_buggy  # Update before potential reflection

        reflection_applied = False
        if (
            draft_flag
            and self.acfg.ITS_Strategy == "self-reflection"
            and result_node.is_buggy
        ):
            logger.info(
                f"AGENT_STEP{current_step_number}: Condition met for self-reflection on drafted buggy node {result_node.id}.",
                extra={"verbose": True},
            )
            # console.rule(f"[cyan]Stage : Self Reflection") # Keep if useful for interactive runs
            try:
                reflection_plan, reflection_code = self.reflect(node=result_node)
                if (
                    reflection_code
                    and reflection_code.strip()
                    and reflection_code != result_node.code
                ):
                    logger.info(
                        f"AGENT_STEP{current_step_number}: Self-reflection yielded new code for node {result_node.id}. Re-executing.",
                        extra={"verbose": True},
                    )
                    result_node.code = reflection_code  # Update node's code
                    reflection_applied = True

                    logger.info(
                        f"AGENT_STEP{current_step_number}: Re-executing reflected code for node {result_node.id}. Code length: {len(result_node.code)}",
                        extra={"verbose": True},
                    )
                    logger.debug(
                        f"AGENT_STEP{current_step_number}_REFLECTED_CODE_TO_EXECUTE_START\n{result_node.code}\nAGENT_STEP{current_step_number}_REFLECTED_CODE_TO_EXECUTE_END",
                        extra={"verbose": True},
                    )
                    exec_start_time = time.time()
                    exec_result = exec_callback(result_node.code, reset_session=True)
                    exec_duration = (
                        time.time() - exec_start_time
                    )  # Update exec_duration
                    logger.info(
                        f"AGENT_STEP{current_step_number}: Reflected code execution finished in {exec_duration:.2f}s. Success: {exec_result.term_out}",
                        extra={"verbose": True},
                    )
                    logger.debug(
                        f"AGENT_STEP{current_step_number}_REFLECTED_EXEC_RESULT_STDOUT_START\n{exec_result.term_out}\nAGENT_STEP{current_step_number}_REFLECTED_EXEC_RESULT_STDOUT_END",
                        extra={"verbose": True},
                    )
                    logger.debug(
                        f"AGENT_STEP{current_step_number}_REFLECTED_EXEC_RESULT_STDERR_START\n{exec_result.term_out}\nAGENT_STEP{current_step_number}_REFLECTED_EXEC_RESULT_STDERR_END",
                        extra={"verbose": True},
                    )

                    logger.info(
                        f"AGENT_STEP{current_step_number}: Parsing execution results for reflected code of node {result_node.id}.",
                        extra={"verbose": True},
                    )
                    result_node = self.parse_exec_result(
                        node=result_node, exec_result=exec_result
                    )  # Re-parse
                else:
                    logger.info(
                        f"AGENT_STEP{current_step_number}: Self-reflection did not result in applicable code changes for node {result_node.id}.",
                        extra={"verbose": True},
                    )
            except Exception as e:  # Catch errors specifically from reflection process
                logger.error(
                    f"AGENT_STEP{current_step_number}: Error during self-reflection stage for node {result_node.id}: {e}",
                    exc_info=True,
                    extra={"verbose": True},
                )

        # Update effectiveness flags
        if (
            self._prev_buggy and not result_node.is_buggy
        ):  # If it *was* buggy (before reflection if any) and now is *not*
            result_node.effective_debug_step = True  # This might mean the main debug/improve worked, or reflection worked
            result_node.effective_reflections = (
                reflection_applied  # True only if reflection was applied and fixed it
            )
        else:
            result_node.effective_debug_step = False
            result_node.effective_reflections = False
        self._prev_buggy = result_node.is_buggy  # Final update for next step

        # --- Logging to W&B and Journal ---
        logger.info(
            f"AGENT_STEP{current_step_number}: Preparing step log data for W&B.",
            extra={"verbose": True},
        )
        step_log_data = {
            f"exec/exec_time_s": exec_duration,
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name": self.competition_name,
            "exec/exception_type": (
                result_node.exc_type if result_node.exc_type else "None"
            ),  # Ensure it's a string
            f"code/estimated_quality": int(self._code_quality),
            # f"eval/reflection_usage": 1 if reflection_applied and not result_node.is_buggy else 0, # Old name
            f"eval/reflection_applied_successfully": (
                1 if reflection_applied and not result_node.is_buggy else 0
            ),  # More specific name
            # f"eval/effective_debug_step": 1 if result_node.effective_debug_step else 0, # Old name
            f"eval/effective_fix_this_step": (
                1 if result_node.effective_debug_step else 0
            ),  # Renamed for clarity
            # f"eval/effective_reflections": 1 if result_node.effective_reflections else 0, # This was probably redundant with reflection_applied_successfully
        }

        agent_validation_metrics_defined = False  # Flag to track if it's defined
        if (
            not result_node.is_buggy
            and result_node.metric
            and result_node.metric.value is not None
        ):
            step_log_data[f"eval/validation_metric"] = result_node.metric.value
            agent_validation_metrics_defined = True  # Set flag

            if self.competition_benchmarks:  # Check if benchmarks are available
                # Use .get with a default that ensures comparison is false if key missing
                # (e.g. comparing to float('inf') for '>' will be false)
                above_median_val = (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks.get("median_threshold", float("inf"))
                    else 0
                )
                gold_medal_val = (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks.get("gold_threshold", float("inf"))
                    else 0
                )
                silver_medal_val = (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks.get("silver_threshold", float("inf"))
                    else 0
                )
                bronze_medal_val = (
                    1
                    if result_node.metric.value
                    > self.competition_benchmarks.get("bronze_threshold", float("inf"))
                    else 0
                )

                self._above_median_flags.append(above_median_val)
                self._gold_medal_flags.append(gold_medal_val)
                self._silver_medal_flags.append(silver_medal_val)
                self._bronze_medal_flags.append(bronze_medal_val)

                # --- Bar charts for threshold flags ---
                if wandb and self.wandb_run:  # Check if wandb is available
                    # Above Median
                    above_true = sum(self._above_median_flags)
                    above_false = len(self._above_median_flags) - above_true
                    above_table = wandb.Table(
                        data=[
                            ["Above Median", above_true],
                            ["Below Median", above_false],
                        ],
                        columns=["label", "count"],
                    )
                    step_log_data["plots/above_median_bar"] = wandb.plot.bar(
                        above_table, "label", "count", title="Above Median Steps"
                    )

                    # Gold Medal
                    gold_true = sum(self._gold_medal_flags)
                    gold_false = len(self._gold_medal_flags) - gold_true
                    gold_table = wandb.Table(
                        data=[["Gold Medal", gold_true], ["No Gold Medal", gold_false]],
                        columns=["label", "count"],
                    )
                    step_log_data["plots/gold_medal_bar"] = wandb.plot.bar(
                        gold_table, "label", "count", title="Gold Medal Steps"
                    )

                    # Silver Medal
                    silver_true = sum(self._silver_medal_flags)
                    silver_false = len(self._silver_medal_flags) - silver_true
                    silver_table = wandb.Table(
                        data=[
                            ["Silver Medal", silver_true],
                            ["No Silver Medal", silver_false],
                        ],
                        columns=["label", "count"],
                    )
                    step_log_data["plots/silver_medal_bar"] = wandb.plot.bar(
                        silver_table, "label", "count", title="Silver Medal Steps"
                    )

                    # Bronze Medal
                    bronze_true = sum(self._bronze_medal_flags)
                    bronze_false = len(self._bronze_medal_flags) - bronze_true
                    bronze_table = wandb.Table(
                        data=[
                            ["Bronze Medal", bronze_true],
                            ["No Bronze Medal", bronze_false],
                        ],
                        columns=["label", "count"],
                    )
                    step_log_data["plots/bronze_medal_bar"] = wandb.plot.bar(
                        bronze_table, "label", "count", title="Bronze Medal Steps"
                    )
            else:
                logger.warning(
                    f"AGENT_STEP{current_step_number}: Competition benchmarks not available, skipping medal plots.",
                    extra={"verbose": True},
                )
        else:
            step_log_data[f"eval/validation_metric"] = float(
                "nan"
            )  # W&B handles NaN well

        # Final check for submission file existence
        submission_path = submission_dir / "submission.csv"
        submission_exists = submission_path.exists()
        if not result_node.is_buggy and not submission_exists:  # This logic is good
            logger.warning(
                f"AGENT_STEP{current_step_number}: Node {result_node.id} was not buggy but submission.csv MISSING. Marking as buggy.",
                extra={"verbose": True},
            )
            result_node.is_buggy = True  # Mark as buggy
            result_node.metric = (
                WorstMetricValue()
            )  # Reset metric because it's effectively a failure
            # If it was previously considered not buggy and had a metric, that metric is now invalid.
            # We should also ensure that `step_log_data["eval/validation_metric"]` reflects this.
            step_log_data[f"eval/validation_metric"] = float("nan")
            # And remove it from metric_hist if it was added based on the now-invalid assumption
            if (
                agent_validation_metrics_defined
                and self._metric_hist
                and self._metric_hist[-1]
                == result_node.metric.original_value_before_reset_to_worst
            ):  # hypothetical attribute
                # This logic is tricky: if it was added to _metric_hist based on a value that's now invalid
                # because submission.csv is missing, we might want to remove it.
                # For simplicity now, we'll let it be logged as NaN for this step.
                # The key is that is_buggy is now true.
                pass

        step_log_data[f"eval/submission_produced"] = 1 if submission_exists else 0

        # --- Histogram of validation metric
        # This should only append if the node is NOT buggy AND has a valid metric.
        # The previous block already sets step_log_data["eval/validation_metric"] to NaN if buggy.
        if (
            not result_node.is_buggy
            and result_node.metric
            and result_node.metric.value is not None
        ):
            self._metric_hist.append(result_node.metric.value)

        if wandb and self.wandb_run:
            if len(self._metric_hist) >= 1:  # Plot even with one point
                try:  # Add try-except for W&B table creation
                    metric_table_data = [
                        [v] for v in self._metric_hist if isinstance(v, (int, float))
                    ]  # Ensure data is plottable
                    if metric_table_data:
                        tbl = wandb.Table(data=metric_table_data, columns=["val"])
                        step_log_data["plots/val_metric_scatter"] = wandb.plot.scatter(
                            tbl,
                            "val",
                            "val",
                            title="Validation Metric Values (Non-Buggy Steps)",
                        )
                    else:
                        logger.warning(
                            f"AGENT_STEP{current_step_number}: No valid metric data to plot for val_metric_scatter.",
                            extra={"verbose": True},
                        )
                except Exception as e:
                    logger.error(
                        f"AGENT_STEP{current_step_number}: Error creating W&B scatter plot for metrics: {e}",
                        exc_info=True,
                        extra={"verbose": True},
                    )

            # Keep a rolling list of 0/1 flags for every step
            self._bug_flags.append(1 if result_node.is_buggy else 0)
            bug_count = sum(self._bug_flags)
            clean_count = len(self._bug_flags) - bug_count
            try:
                bug_table = wandb.Table(
                    data=[["Buggy", bug_count], ["Clean", clean_count]],
                    columns=["label", "count"],
                )
                step_log_data["plots/bug_vs_clean"] = wandb.plot.bar(
                    bug_table, "label", "count", title="Buggy vs clean steps"
                )
            except Exception as e:
                logger.error(
                    f"AGENT_STEP{current_step_number}: Error creating W&B bar plot for bug_vs_clean: {e}",
                    exc_info=True,
                    extra={"verbose": True},
                )

            # --- Bar chart: Submission produced vs missing
            self._sub_flags.append(
                1 if submission_exists else 0
            )  # submission_exists is defined above
            with_sub = sum(self._sub_flags)
            without_sub = len(self._sub_flags) - with_sub
            try:
                sub_table = wandb.Table(
                    data=[["Has submission", with_sub], ["No submission", without_sub]],
                    columns=["label", "count"],
                )
                step_log_data["plots/submission_presence"] = wandb.plot.bar(
                    sub_table, "label", "count", title="Submission produced vs missing"
                )
            except Exception as e:
                logger.error(
                    f"AGENT_STEP{current_step_number}: Error creating W&B bar plot for submission_presence: {e}",
                    exc_info=True,
                    extra={"verbose": True},
                )

        # --- Send log data to W&B ---
        if self.wandb_run:
            # t_wandb_start = time.time() # last variable seems unused
            logger.info(
                f"AGENT_STEP{current_step_number}: Logging data to W&B. Keys: {list(step_log_data.keys())}",
                extra={"verbose": True},
            )
            try:
                self.wandb_run.log(step_log_data, step=current_step_number)
            except Exception as e:
                logger.error(
                    f"AGENT_STEP{current_step_number}: Error logging to W&B: {e}",
                    exc_info=True,
                    extra={"verbose": True},
                )
            # last = time.time() # Unused
        else:
            logger.info(
                f"AGENT_STEP{current_step_number}: W&B run not available, skipping W&B log.",
                extra={"verbose": True},
            )
        # --- End Send log data ---

        # Storing node.code_quality happens in parse_exec_result now.

        result_node.stage = node_stage
        result_node.exec_time = exec_duration

        self.journal.append(result_node)
        logger.info(
            f"AGENT_STEP{current_step_number}: Appended node {result_node.id} to journal. Journal size: {len(self.journal.nodes)}",
            extra={"verbose": True},
        )

        best_node = self.journal.get_best_node()
        if best_node and best_node.id == result_node.id:
            logger.info(
                f"AGENT_STEP{current_step_number}: Node {result_node.id} is new best node (Metric: {best_node.metric.value if best_node.metric else 'N/A':.4f}). Caching solution.",
                extra={"verbose": True},
            )
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_submission_dir = self.cfg.workspace_dir / "best_submission"
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            best_submission_dir.mkdir(exist_ok=True, parents=True)
            if submission_exists:
                shutil.copy(
                    submission_path, best_submission_dir / "submission.csv"
                )  # Add filename
            best_code_path = best_solution_dir / "solution.py"
            with open(best_code_path, "w") as f:
                f.write(result_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f:
                f.write(str(result_node.id))
        elif best_node:
            logger.info(
                f"AGENT_STEP{current_step_number}: Current best node is {best_node.id} (Metric: {best_node.metric.value if best_node.metric else 'N/A':.4f})",
                extra={"verbose": True},
            )
        else:
            logger.info(
                f"AGENT_STEP{current_step_number}: No best node identified yet.",
                extra={"verbose": True},
            )

        # log_step is for console, keep it as is or make it more verbose if needed
        log_step(
            step=current_step_number,
            total=self.acfg.steps,
            stage=node_stage,
            is_buggy=result_node.is_buggy,
            exec_time=exec_duration,
            metric=(
                result_node.metric.value
                if result_node.metric and result_node.metric.value is not None
                else None
            ),
        )

        t_step_end = time.time()
        logger.info(
            f"AGENT_STEP_END: {current_step_number}, Duration: {t_step_end - t_step_start:.2f}s",
            extra={"verbose": True},
        )

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        log_prefix = f"PARSE_EXEC_RESULT_STEP{self.current_step}_NODE{node.id}"
        logger.info(f"{log_prefix}: Parsing execution result.", extra={"verbose": True})

        node.absorb_exec_result(exec_result)
        logger.debug(
            f"{log_prefix}_ABSORBED_EXEC_RESULT_NODE_EXC_TYPE: {node.exc_type}",
            extra={"verbose": True},
        )
        logger.debug(
            f"{log_prefix}_ABSORBED_EXEC_RESULT_NODE_TERM_OUT_START\n{node.term_out}\n{log_prefix}_ABSORBED_EXEC_RESULT_NODE_TERM_OUT_END",
            extra={"verbose": True},
        )

        introduction = "You are a Kaggle grandmaster ... evaluate the output ... empirical findings."  # Truncated
        if self.acfg.obfuscate:
            introduction = "You are an expert machine learning engineer ... evaluate the output ... empirical findings."  # Truncated

        feedback_system_prompt = {  # Renamed from prompt to be specific
            "Introduction": introduction,
            "Task Description": self.task_desc,
            "Code Executed": wrap_code(node.code),
            "Execution Output Log": wrap_code(node.term_out, lang=""),
        }
        # No user_message for function calling with system prompt

        max_retries = 3
        review_response_dict: Optional[dict] = None  # Explicitly dict or None

        for attempt in range(max_retries):
            logger.info(
                f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Querying feedback LLM.",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_SYSTEM_PROMPT_START\n{feedback_system_prompt}\n{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_SYSTEM_PROMPT_END",
                extra={"verbose": True},
            )
            logger.debug(
                f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_FUNC_SPEC_START\n{review_func_spec.to_dict()}\n{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_FUNC_SPEC_END",
                extra={"verbose": True},
            )

            try:
                # This query should return a dict if func_spec is used correctly by the backend
                raw_response = query(
                    system_message=feedback_system_prompt,
                    user_message=None,  # User message is None for this type of call
                    func_spec=review_func_spec,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    # excute = False, # 'excute' param not in query, assume func_spec implies it
                    convert_system_to_user=self.acfg.convert_system_to_user,
                    current_step=self.current_step,  # Pass current_step
                )
                logger.info(
                    f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Received response.",
                    extra={"verbose": True},
                )
                logger.debug(
                    f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_RAW_RESPONSE_START\n{raw_response}\n{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_RAW_RESPONSE_END",
                    extra={"verbose": True},
                )

                if not isinstance(raw_response, dict):
                    logger.error(
                        f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Response is not a dict as expected for function call. Type: {type(raw_response)}",
                        extra={"verbose": True},
                    )
                    # Try to parse if it's a string that looks like JSON (common failure mode)
                    if isinstance(raw_response, str):
                        try:
                            import json

                            raw_response = json.loads(raw_response)
                            if not isinstance(
                                raw_response, dict
                            ):  # Still not a dict after parsing
                                raise ValueError("Parsed JSON is not a dict")
                            logger.info(
                                f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Successfully parsed string response to dict.",
                                extra={"verbose": True},
                            )
                        except Exception as json_e:
                            logger.error(
                                f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Failed to parse string response as JSON: {json_e}",
                                extra={"verbose": True},
                            )
                            raw_response = None  # Force retry or failure
                    else:
                        raw_response = None  # Force retry or failure

                review_response_dict = (
                    cast(dict, raw_response) if isinstance(raw_response, dict) else None
                )

                if review_response_dict and all(
                    k in review_response_dict
                    for k in [
                        "is_bug",
                        "has_csv_submission",
                        "summary",
                        "metric",
                        "lower_is_better",
                        "code_quality",
                    ]
                ):
                    logger.info(
                        f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Successfully received and validated feedback response.",
                        extra={"verbose": True},
                    )
                    break
                else:
                    logger.warning(
                        f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Feedback LLM response missing required keys or is None. Response: {review_response_dict}",
                        extra={"verbose": True},
                    )
                    review_response_dict = None  # Force retry
            except Exception as e:
                logger.error(
                    f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Error querying feedback LLM: {e}",
                    exc_info=True,
                    extra={"verbose": True},
                )

            if attempt == max_retries - 1 and review_response_dict is None:
                logger.error(
                    f"{log_prefix}: Feedback LLM query failed after {max_retries} retries. Using default error values.",
                    extra={"verbose": True},
                )
                review_response_dict = {
                    "is_bug": True,
                    "has_csv_submission": False,
                    "summary": "Failed to get feedback from LLM after multiple retries.",
                    "metric": None,
                    "lower_is_better": True,
                    "code_quality": 0,
                }
                break  # Exit loop with default error values

        # Ensure review_response_dict is not None here (it should have default if all retries failed)
        if review_response_dict is None:  # Should not happen due to above logic
            review_response_dict = {
                "is_bug": True,
                "has_csv_submission": False,
                "summary": "CRITICAL: review_response_dict became None unexpectedly.",
                "metric": None,
                "lower_is_better": True,
                "code_quality": 0,
            }

        metric_value = review_response_dict.get("metric")
        if not isinstance(metric_value, (float, int)):
            if metric_value is not None:  # Log if it was something else
                logger.warning(
                    f"{log_prefix}: Metric value from LLM is not a float/int: '{metric_value}' (type: {type(metric_value)}). Setting to None.",
                    extra={"verbose": True},
                )
            metric_value = None

        self._code_quality = review_response_dict.get("code_quality", 0)
        if not isinstance(self._code_quality, (int, float)):  # Ensure it's a number
            logger.warning(
                f"{log_prefix}: Code quality from LLM is not an int/float: '{self._code_quality}'. Setting to 0.",
                extra={"verbose": True},
            )
            self._code_quality = 0
        node.code_quality = int(self._code_quality)  # Store on node as well

        submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
        has_csv_submission_actual = submission_path.exists()
        has_csv_submission_reported = review_response_dict.get(
            "has_csv_submission", False
        )

        node.analysis = review_response_dict.get(
            "summary", "Feedback LLM summary missing."
        )
        logger.debug(
            f"{log_prefix}_LLM_ANALYSIS_SUMMARY_START\n{node.analysis}\n{log_prefix}_LLM_ANALYSIS_SUMMARY_END",
            extra={"verbose": True},
        )

        is_bug_llm = review_response_dict.get("is_bug", True)
        exc_type_exists = node.exc_type is not None
        metric_missing = metric_value is None
        csv_missing_reported = not has_csv_submission_reported
        csv_missing_actual = not has_csv_submission_actual

        node.is_buggy = (
            is_bug_llm
            or exc_type_exists
            or metric_missing
            or csv_missing_reported
            or csv_missing_actual
        )
        logger.info(
            f"{log_prefix}: Final buggy status: {node.is_buggy}",
            extra={"verbose": True},
        )

        if node.is_buggy:
            bug_reasons = []
            if is_bug_llm:
                bug_reasons.append(f"LLM_judged_buggy (Summary: {node.analysis})")
            if exc_type_exists:
                bug_reasons.append(f"Exception_occurred ({node.exc_type})")
            if metric_missing:
                bug_reasons.append("Metric_missing_or_invalid_from_LLM")
            if csv_missing_reported:
                bug_reasons.append("LLM_reported_CSV_missing")
            if csv_missing_actual:
                bug_reasons.append("Actual_CSV_file_missing")
            logger.info(
                f"{log_prefix}: Reasons for buggy status: {'; '.join(bug_reasons)}",
                extra={"verbose": True},
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(
                f"{log_prefix}: Node determined not buggy. Metric value: {metric_value}, Lower_is_better: {not review_response_dict.get('lower_is_better', True)}",
                extra={"verbose": True},
            )
            node.metric = MetricValue(
                metric_value,
                maximize=not review_response_dict.get("lower_is_better", True),
            )

        return node
