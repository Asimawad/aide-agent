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
from .utils.response import extract_code, extract_text_up_to_code, wrap_code,trim_long_string, format_code, extract_plan, extract_summary
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
            "3. Implement the solution proposed in the plan.",
            "4. Calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
            "5. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",
            "6. The script must run without errors. Focus on correctness first.",
            "7. The code should be clean and easy to understand. It should be well-documented and well-structured.",
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
        "Response format": ("Your response for the summary should be a detailed and high quality bullet points of the bugs in the previous solution, summarizing all the information and problems(5-7 sentences), "
                "Your response for the plan should be a detailed and high quality bullet points of the steps of your proposed solution in natural language (7-10 sentences), "
                "There should be no additional headings or Code in your response. Just natural language text (summary) under ## Bugs Summary/Analysis: and natural language text (plan) under ## Plan: "
                "explicitly,structure your answer exactly like this: " ) + fmt
        }

    @property
    def code_prompt_resp_fmt(self):
        fmt = (
                    "\n\n---\n"
                    "1) CODE (one fenced Python block):\n"
                    "```python\n"
                    "<your python code here>\n"
                    "```"
                )
        return {
            "Response format": (
                "Your response should be a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just the markdown code block. "
                "explicitly,structure your answer exactly like this: ") + fmt
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
                "explicitly,structure your answer exactly like this: ") + fmt
        }
    def plan_query(self, prompt, retries=3) -> tuple[str]:
        """Generate a step by step natural language plan that will be fed to the coder model."""
        system_prompt = {
            "SYSTEM":"You are a Kaggle Grandmaster and a team leader. you can plan high detailed and quality machine learning engineering solutions,",
            "user_instructions": {
               "Possible Questions you will face": "You will be asked to come up with a step by step plan to solve the kaggle competetion",
               "How to answer the user": "Whenever you answer, always: 1. Write a \"## Task Summary:\" section in plain text consisting of 5-7 sentences distilling the task for you team members that are responsible for implementing the solution. 2. Write a \"## Plan:\" section in plain text consisting of detailed and high quality bullet points that will be used by the team members to implement the solution (7-10 bullet points). ",
                "Critical Instructions":"Do not give/write code solutions, coding is not your job, just consice summary and detailed plan"
                }
        }

        completion_text = None

        for _ in range(retries):
            completion_text  = query(
                system_message=system_prompt,
                user_message=prompt,
                model=self.acfg.code.planner_model,
                temperature=self.acfg.code.temp,
                current_step=self.current_step,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )


            plan    = extract_plan(completion_text)
            summary = extract_summary(completion_text)

            if plan and summary:
                # merge all code blocks into a single string
                return summary, plan, ""

            logger.info("Plan + summary extraction failed, retrying...")
        logger.info("Final plan + summary extraction attempt failed, giving up...")
        return "", completion_text, ""  # type: ignore
    # Inside aide-ds/aide/agent.py, within the Agent class
    def code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Follow a predefined plan and implement the code that solves the kaggle competetion."""
        system_prompt = {
            "SYSTEM":"You are a Kaggle Grandmaster and great at implementing machine learning engineering code. Precisely follow the plan to implement the code that solves the kaggle competetion.",
            "user_instructions": {
               "What you will face": "You will be given a plan to implement the code that solves the kaggle competetion. Precisely follow the plan to implement the code.",
               "How to answer the user": "Whenever you answer, always: answer in one section called \"CODE:\" containing exactly one fenced Python block: ```python implementing the plan"
            }
        }

        completion_text = None
        for _ in range(retries):

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

            if code:
                # merge all code blocks into a single string
                return nl_text, code , ""

            logger.info("code extraction failed, retrying...")
        logger.info("Final code extraction attempt failed, giving up...")
        return "", completion_text, ""  # type: ignore

    def _draft(self, parent_node=None) -> Node: 
        console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Drafting") # Keep if you like console output here
        # logger.info(f"Agent step {self.current_step}: Drafting new solution (parent: {parent_node})")

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
        plan_introduction = (
            f"given the following task description for a machine learning competition named {self.competition_name}, develop a complete and detailed plan to solve it."
        )
        code_introduction = (
            f"given the following task description about a machine learning competition named {self.competition_name}, and the plan to solve it, develop a complete code to solve it."
        )
        prompt_user_message: Any = {
            "Introduction": plan_introduction,
            "Overall Task Description": self.task_desc, # This is the markdown/text from the competition
            "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(),
            "Instructions": {},
        }

# Fallback if no template is found - revert to original _draft prompting style
        prompt_user_message["Instructions"] |= self.plan_prompt_resp_fmt # Original response format
        prompt_user_message["Instructions"] |= { # Original sketch guidelines
            "Solution plan guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization. as we are using this as a first draft for future improvements",
                "the summary should be 5-7 sentences that describe the task in a nutshell, so that the team members can understand the task and the plan",
                "Take the Memory section into consideration when proposing the design.",
                "The solution plan should be detailed and high quality bullet points that are easy to follow.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        # prompt_user_message["Instructions"] |= self._prompt_impl_guideline # Original implementation guidelines

        prompt_user_message["Instructions"] |= self._prompt_environment # Common environment prompt

        if self.acfg.data_preview:
            prompt_user_message["Data Overview"] = self.data_preview

        agent_summary_for_step, agent_plan_for_step, _ = self.plan_query(prompt_user_message)

        
        code_prompt_user_message: Any = {
            "Introduction": code_introduction,
            "Overall Task Description": agent_summary_for_step, # This is the markdown/text from the competition
            "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(),
            "Instructions": {},
        }


        code_prompt_user_message["Instructions"] |= self._prompt_environment # Common environment prompt

        if self.acfg.data_preview:
            code_prompt_user_message["Data Overview"] = self.data_preview



        _, generated_code , _ = self.code_query(code_prompt_user_message)
        
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
            summary=agent_summary_for_step, # This field seems not heavily used, but kept for consistency
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
        planner_introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first summarize the task, and outline your proposed improvement in natural language based on the provided previous solution. "
            "then you should outline a high quality and detailed step by step plan in natural language for how the solution can be improved "
        )
        
        code_introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution and a high quality plan for improvement below and should implement the improvement in order to further increase the (test time) performance. "
                "for this you should write the code that implement this improvement plan in Python based on the provided previous solution and following the given plan. "
        )

        plan_prompt_user_message: Any = {
            "Introduction": planner_introduction,
            "Overall Task Description": self.task_desc, # This is the markdown/text from the competition
            "Memory (Summary of Previous Attempts on this Task)": self.journal.generate_summary(),
        }
        plan_prompt_user_message["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }
        plan_prompt_user_message["Instructions"] |= self.plan_prompt_resp_fmt
        plan_prompt_user_message["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "you should provide a summary of the task description and the previous solution and then outline a high quality and detailed step by step plan in natural language for how the solution can be improved ",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
            ],
        }


        agent_summary_for_step, agent_plan_for_step, _ = self.plan_query(plan_prompt_user_message)

        prompt: Any = {
            "Introduction": code_introduction,
            "Task description summary and previous solution": agent_summary_for_step,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }
        prompt["Improvement plan"] = {
            "Plan": agent_plan_for_step,
        }
        prompt["Instructions"] |= self.code_prompt_resp_fmt
        prompt["Instructions"] |= {
            "code improvement guideline": [
                "You should precisely follow the plan for improvement and implement the code that implements the improvement.",
                "the final code should be a single code block and should be formatted using the code block format. and it should be complete and self contained.",
                "the code should be well documented and should be easy to understand.",
                "you should strictly follow the plan for improvement and implement the code that implements the improvement.",
                "Take the Memory section into consideration during the implementation to avoid bugs.",
                "The code should be optimized for performance and should be efficient.",
                "The code should be well formatted and should be easy to read.",  
                "code should be between ```python fences"  
                "only write code, do not write any other text"        
                  ],
        }
        prompt["Instructions"] |=    {
            "1. Write a complete, single-file Python script. ",
            "2. starting with imports, and load necessary data from the './input/' directory, the same way the previous solution did.",
            "3. Implement the improvement proposed in the plan.",
            "4. remember to calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
            "5. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. the same way the previous solution did.",
            "6. The code should be clean and easy to understand. It should be well-documented and well-structured.",
        }



        _, generated_code , _ = self.code_query(prompt)
        new_node = Node(plan=agent_plan_for_step, code=generated_code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        console.rule(f"[cyan]Stage : Debugging")
        logger.info(f"Agent step {self.current_step}: Generating code (parent type: {parent_node.stage_name})", extra={"verbose": True})
        plan_introduction = (
            "You are a Kaggle grandmaster AND A TEAM LEADER. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be a summary of the problems/bugs in the previous solution in natural language bullet points."
            "followed by a detailed plan for fixing the bugs in natural language bullet points.(7-10 bullet points)"
        )

        plan_prompt: Any = {
            "Introduction": plan_introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        plan_prompt["Instructions"] |= self.debug_prompt_resp_fmt
  
        # plan_prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            plan_prompt["Data Overview"] = self.data_preview




        agent_summary_for_step, agent_plan_for_step, _ = self.plan_query(plan_prompt)


        code_introduction = (
            "You are a Kaggle grandmaster AND A TEAM MEMBER. "
            "Your team's previous solution had a bug and/or did not produce a submission.csv, "
            "you will be given the previous solution and the plan for fixing the bugs. "
            "you should implement the bugfix/solution in Python based on the provided previous solution and following the given plan. "
        )

        code_prompt: Any = {
            "Introduction": code_introduction,
            "Task description": agent_summary_for_step,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }


        code_prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "precisely follow the plan for fixing the bugs and implement the code that implements the improvement.",
                "the final code should be a single code block and should be formatted using the code block format. and it should be complete and self contained.",
            ],
        }
        code_prompt["Instructions"] |= self.code_prompt_resp_fmt

        _, generated_code , _ = self.code_query(code_prompt)
        new_node = Node(plan=agent_plan_for_step, code=generated_code, parent=parent_node)
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
