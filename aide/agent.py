import shutil
import logging
import random
import json
import time
from rich.syntax import Syntax # Keep for logging if used
from rich.console import Console # Keep for console output
from typing import Any, Callable, cast, Optional, Dict # Added Dict
from .backend import query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview # data_preview.generate
from .utils.config import Config
from .utils.pretty_logging import log_step # logger from pretty_logging might conflict, be careful

from .utils.wandb_logger import WandbLogger
from .utils.prompt_utils import (
    review_func_spec,
    get_agent_draft_user_prompt,
    get_agent_improve_user_prompt,
    get_agent_debug_user_prompt,
    CHAINED_CODER_USER_PROMPT_CONSTRUCTORS, # New
    CHAINED_CODER_SYSTEM_PROMPT_GETTERS,
    get_agent_system_prompt,
    get_agent_draft_system_prompt,
    get_planner_agent_draft_plan_user_prompt,
    get_planner_agent_draft_code_user_prompt,
    get_planner_agent_improve_plan_user_prompt,
    get_planner_agent_improve_code_user_prompt,
    get_planner_agent_debug_plan_user_prompt,
    get_planner_agent_debug_code_user_prompt,
    get_planner_agent_plan_system_prompt,
    get_planner_agent_code_system_prompt,
    wrap_code as prompt_utils_wrap_code # Alias if local wrap_code is different
)

from .utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code, # This is the local wrap_code, ensure it's the one you intend for direct calls
    trim_long_string,
    format_code,
    extract_summary_and_plan
)
from .utils.self_reflection import (
    perform_two_step_reflection,
)
from .utils.metric import MetricValue, WorstMetricValue # Moved here for clarity


try:
    import wandb
except ImportError:
    wandb = None


logger = logging.getLogger("aide") # Assuming "aide" is the main logger from pretty_logging
console = Console()


def format_time(time_in_sec: int): # Should be float for more precision
    time_in_sec = int(time_in_sec) # Cast to int if original signature is intended
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


ExecCallbackType = Callable[[str, bool], ExecutionResult]


class Agent: # This is now the base class
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_logger: Optional['WandbLogger'] = None, # Type hint with quotes for forward declaration
        competition_benchmarks=None,
    ):
        if isinstance(task_desc, dict):
            from .backend import compile_prompt_to_md
            self.task_desc = compile_prompt_to_md(task_desc)
        else:
            self.task_desc = task_desc

        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.wandb_logger = wandb_logger # Store the WandbLogger instance
        self.competition_benchmarks = competition_benchmarks
        self.competition_name = self.cfg.competition_name
        
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        self._prev_buggy: bool = False
        self._code_quality: float = 0.0 # For feedback LLM
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
        # console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Search Policy")

        log_prefix_base = f"{self.__class__.__name__.upper()}_SEARCH_POLICY_STEP{self.current_step}"
        search_cfg = self.acfg.search

        search_cfg = self.acfg.search

        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info(f"{log_prefix_base}: Selected: Draft new node (drafts: {len(self.journal.draft_nodes)} < {search_cfg.num_drafts}).", extra={"verbose": True})
            return None

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                n for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"{log_prefix_base}: Selected: Debug node {node_to_debug.id} (debug_prob triggered, depth {node_to_debug.debug_depth}).", extra={"verbose": True})
                return node_to_debug
            else:
                logger.info(f"{log_prefix_base}: Attempted debug (debug_prob triggered), but no debuggable nodes found.", extra={"verbose": True})

        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info(f"{log_prefix_base}: Selected: Draft new node (no good nodes to improve).", extra={"verbose": True})
            return None

        greedy_node = self.journal.get_best_node()
        if greedy_node:
            if greedy_node.is_buggy:
                 logger.info(f"{log_prefix_base}: Selected: Debug greedy node {greedy_node.id} (it was marked buggy).", extra={"verbose": True})
                 return greedy_node
            metric_display = f"{greedy_node.metric.value:.3f}" if greedy_node.metric and greedy_node.metric.value is not None else 'N/A'
            logger.info(f"{log_prefix_base}: Selected: Improve greedy node {greedy_node.id} (metric: {metric_display}).", extra={"verbose": True})
            return greedy_node
        # Corrected line:
        metric_display = f"{greedy_node.metric.value:.3f}" if greedy_node.metric and greedy_node.metric.value is not None else 'N/A'
        logger.info(f"{log_prefix_base}: Selected: Improve greedy node {greedy_node.id} (metric: {metric_display}).", extra={"verbose": True})
        return greedy_node

    # REMOVE: _prompt_environment, _prompt_impl_guideline, _prompt_resp_fmt
    # These are now handled by functions in prompt_utils.py

    def plan_and_code_query(self, user_prompt_dict: Dict[str, Any], excute: bool, system_prompt_dict=None, retries: int = 3) -> tuple[str, str, str]: 
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        if system_prompt_dict is None:
            system_prompt_dict = get_agent_system_prompt()
        log_prefix = f"AGENT_PLAN_CODE_QUERY_STEP->{self.current_step}" 
        completion_text = None
        for attempt in range(retries):
            logger.info(f"Sending request. attempt {attempt+1}/{retries}", extra={"verbose": True})
            try:
                completion_text = query(
                    system_message=system_prompt_dict,
                    user_message=user_prompt_dict,
                    model=self.acfg.code.model,
                    temperature=self.acfg.code.temp,
                    max_tokens=self.acfg.code.max_new_tokens,
                    current_step=self.current_step,
                    inference_engine=self.cfg.inference_engine,
                    num_responses=self.acfg.code.num_return_sequences,
                    convert_system_to_user=self.acfg.convert_system_to_user,
                )
            except Exception as e:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Query failed: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: return "", f"LLM Query Error: {e}", "LLM_QUERY_ERROR"
                time.sleep(2)
                continue
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)
            if code and nl_text:
                logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Successfully extracted plan and code.", extra={"verbose": True})
                return nl_text, code, "execution_summary_placeholder"
            logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Plan or code extraction failed. Raw text: '{trim_long_string(completion_text)}'", extra={"verbose": True})
        logger.error(f"{log_prefix}: All {retries} attempts for plan+code extraction failed.", extra={"verbose": True})
        return "", completion_text or "No LLM response received", "EXTRACTION_FAILED"


    def _draft(self, parent_node=None) -> Node:
        log_prefix_base = f"{self.__class__.__name__}_DRAFT_STEP:{self.current_step}" # Generic prefix
        logger.info(f"{log_prefix_base}: Starting drafting. Parent: {parent_node.id if parent_node else 'None'}", extra={"verbose": True})
        draft_sys_prompt=get_agent_draft_system_prompt()
        journal_summary=self.journal.generate_summary(include_code=False)
        logger.info(f"{log_prefix_base}: Journal summary: {journal_summary}", extra={"verbose": True})
        prompt_user_message = get_agent_draft_user_prompt( # Agent uses its specific prompt structure
            task_desc=self.task_desc,
            journal_summary=journal_summary,
            competition_name=self.competition_name,
            obfuscate=self.acfg.obfuscate,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        agent_plan_for_step, generated_code, exec_summary = (
            self.plan_and_code_query(user_prompt_dict=prompt_user_message, excute=False,system_prompt_dict = draft_sys_prompt, retries=self.acfg.get('query_retries', 1))
        )
        
        if not agent_plan_for_step: agent_plan_for_step = "PLAN_GENERATION_FAILED"
        if not generated_code: generated_code = "# CODE_GENERATION_FAILED"
        logger.debug(f"{log_prefix_base}_DRAFT_PLAN_START\n{agent_plan_for_step}\n{log_prefix_base}_DRAFT_PLAN_END", extra={"verbose": True})
        logger.debug(f"{log_prefix_base}_DRAFT_CODE_RAW_START\n{generated_code}\n{log_prefix_base}_DRAFT_CODE_RAW_END", extra={"verbose": True})
        new_node = Node(plan=agent_plan_for_step, code=generated_code, summary=exec_summary)
        if parent_node: new_node.parent = parent_node
        logger.info(f"{log_prefix_base}: Drafted new node {new_node.id}.", extra={"verbose": True})
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        log_prefix_base = f"{self.__class__.__name__.upper()}_IMPROVE_STEP{self.current_step}"
        logger.info(f"{log_prefix_base}: Starting improvement for node {parent_node.id}.", extra={"verbose": True})
        prompt_user_message = get_agent_improve_user_prompt( # Agent uses its specific prompt
            task_desc=self.task_desc,
            journal_summary=self.journal.generate_summary(include_code=False),
            competition_name=self.competition_name,
            parent_node_code=parent_node.code,
        )
        plan, code, _ = self.plan_and_code_query(prompt_user_message, excute=False, retries=self.acfg.get('query_retries', 1))

        if not plan: plan = "IMPROVEMENT_PLAN_FAILED"
        if not code: code = parent_node.code
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"{log_prefix_base}: Improvement plan for node {parent_node.id}: {trim_long_string(plan)}", extra={"verbose": True})
        logger.info(f"{log_prefix_base}: Improved node {parent_node.id} to new node {new_node.id}.", extra={"verbose": True})
        return new_node

    
    def _debug(self, parent_node: Node) -> Node:
        log_prefix_base = f"{self.__class__.__name__}_DEBUG_STEP{self.current_step}"
        logger.info(f"{log_prefix_base}: Starting debugging for node {parent_node.id}.", extra={"verbose": True})
        logger.info(f"Buggy code: {parent_node.code}", extra={"verbose": True})
        prompt_user_message = get_agent_debug_user_prompt( # Agent uses its specific prompt
            task_desc=self.task_desc,
            competition_name=self.competition_name,
            parent_node_code=parent_node.code,
            parent_node_feedback=parent_node.analysis,
            parent_node_term_out=parent_node.term_out,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        plan, code, _ = self.plan_and_code_query(prompt_user_message, excute=False, retries=self.acfg.get('query_retries', 1))

        if not plan: plan = "DEBUG_PLAN_FAILED"
        if not code: code = parent_node.code
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"{log_prefix_base}: Debugged node {parent_node.id} to create new node {new_node.id}", extra={"verbose": True})
        logger.debug(f"{log_prefix_base}_DEBUG_PLAN_START\n{plan}\n{log_prefix_base}_DEBUG_PLAN_END", extra={"verbose": True})
        logger.debug(f"{log_prefix_base}_DEBUG_CODE_START\n{wrap_code(code)}\n{log_prefix_base}_DEBUG_CODE_END", extra={"verbose": True})
        return new_node

    def reflect(self, node: Node) -> tuple[str, str]:
        log_prefix_base = f"{self.__class__.__name__.upper()}_REFLECT_STEP{self.current_step}_NODE{node.id}"
        # ... (rest of reflect implementation from Agent class) ...
        logger.info(f"{log_prefix_base}: Initiating self-reflection.", extra={"verbose": True})
        try:
            reflection_plan, revised_code = perform_two_step_reflection(
                code=node.code, analysis=node.analysis, term_out=node.term_out,
                task_desc=self.task_desc, model_name=self.acfg.code.model,
                temperature=self.acfg.code.temp, convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query, wrap_code_func=prompt_utils_wrap_code, extract_code_func=extract_code,
                current_step=self.current_step
            )
        except Exception as e:
            logger.error(f"{log_prefix_base}: Error during self-reflection call: {e}", exc_info=True, extra={"verbose": True})
            return f"REFLECTION_ERROR: {e}", node.code
        if revised_code and revised_code.strip() and revised_code != node.code: logger.info(f"{log_prefix_base}: Self-reflection resulted in code changes.", extra={"verbose": True})
        elif reflection_plan == "No specific errors found requiring changes.": logger.info(f"{log_prefix_base}: Self-reflection found no errors requiring changes.", extra={"verbose": True})
        else: logger.warning(f"{log_prefix_base}: Self-reflection finished, but revised code is same as original or empty. Plan: {trim_long_string(reflection_plan)}", extra={"verbose": True})
        return reflection_plan, revised_code



    def double_reflect(self, code: str) -> tuple[str, str]:
        """
        Performs a two-step self-reflection using the external utility function.
        This version doesn't have `analysis` or `term_out` from a node.
        Returns: Tuple: (reflection_plan, revised_code)
        """
        log_prefix_base = f"AGENT_DOUBLE_REFLECT_STEP{self.current_step}" # No node ID here
        logger.info(f"{log_prefix_base}: Initiating self-reflection (double_reflect variant).", extra={"verbose": True})

        try:
            reflection_plan, revised_code = perform_two_step_reflection(
                code=code, # Original code for reflection
                analysis="No specific prior analysis available for this reflection.", # Generic analysis
                term_out="No specific terminal output available for this reflection.", # Generic term_out
                task_desc=self.task_desc,
                model_name=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query,
                wrap_code_func=prompt_utils_wrap_code,
                extract_code_func=extract_code,
                current_step=self.current_step
            )
        except Exception as e:
            logger.error(f"{log_prefix_base}: Error during double_reflect call: {e}", exc_info=True, extra={"verbose": True})
            return f"DOUBLE_REFLECTION_ERROR: {e}", code


        if revised_code and revised_code.strip() and revised_code != code:
            logger.info(f"{log_prefix_base}: Self-reflection (double_reflect) resulted in code changes.", extra={"verbose": True})
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info(f"{log_prefix_base}: Self-reflection (double_reflect) found no errors requiring changes.", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix_base}: Self-reflection (double_reflect) finished, but revised code is same as original or empty. Plan: {trim_long_string(reflection_plan)}", extra={"verbose": True})

        logger.debug(f"{log_prefix_base}_REFLECTION_PLAN_START\n{reflection_plan}\n{log_prefix_base}_REFLECTION_PLAN_END", extra={"verbose": True})
        # logger.debug(f"{log_prefix_base}_REVISED_CODE_BY_DOUBLE_REFLECTION_START\n{wrap_code(revised_code)}\n{log_prefix_base}_REVISED_CODE_BY_DOUBLE_REFLECTION_END", extra={"verbose": True})
        return reflection_plan, revised_code

    def update_data_preview(self):
        log_prefix = f"{self.__class__.__name__.upper()}_DATA_PREVIEW_STEP{self.current_step}"

        logger.info(f"{log_prefix}: Updating data preview.", extra={"verbose": True})
        try:
            self.data_preview = data_preview.generate(self.cfg.workspace_dir / "input")
            logger.info(f"{log_prefix}: Data preview updated.", extra={"verbose": True})
        except Exception as e:
            logger.error(f"{log_prefix}: Failed to update data preview: {e}", exc_info=True, extra={"verbose": True})
            self.data_preview = "Error generating data preview."

    def process_step(self,exec_callback: ExecCallbackType,result_node: Node,node_stage: str, current_step_number: int, use_reflection: bool = True):
        
        logger.info(f"Executing code for step {current_step_number}.", extra={"verbose": True})
        exec_start_time = time.time()
        exec_result = exec_callback(result_node.code, reset_session=True)
        exec_duration = time.time() - exec_start_time
        logger.info(f"Code execution for step {current_step_number} finished in {exec_duration:.2f}s.", extra={"verbose": True})
        result_node = self.parse_exec_result(node=result_node, exec_result=exec_result)
        buggy_status_before_reflection = result_node.is_buggy
        if use_reflection and self.acfg.ITS_Strategy == "self-reflection" and result_node.is_buggy:
            _, reflection_code = self.reflect(node=result_node)
            if reflection_code and reflection_code.strip() and reflection_code != result_node.code:
                result_node.code = reflection_code; reflection_applied = True
                exec_start_time_reflect = time.time()
                exec_result_reflect = exec_callback(result_node.code, reset_session=True)
                exec_duration = time.time() - exec_start_time_reflect
                result_node = self.parse_exec_result(node=result_node, exec_result=exec_result_reflect)
        if buggy_status_before_reflection and not result_node.is_buggy:
            result_node.effective_debug_step = True; result_node.effective_reflections = reflection_applied
        else:
            result_node.effective_debug_step = False; result_node.effective_reflections = False
        self._prev_buggy = result_node.is_buggy
        if result_node.is_buggy:

            console.print(f"[bold red]---------[/bold red]\n") # Console output
            console.print(f"[bold red]stage: {node_stage}[/bold red]") # Console output
            console.print(f"[bold red]Result: Buggy[/bold red]") # Console output
            console.print(f"[bold red]Feedback: {result_node.analysis}[/bold red]") # Console output
        else: 
            console.print(f"[bold green]---------[/bold green]\n") # Console output
            console.print(f"[bold green]stage: {node_stage}[/bold green]")
            console.print(f"[bold green]Result: Not Buggy[/bold green]") # Console output
            console.print(f"[bold green]Feedback: {result_node.analysis}[/bold green]") # Console output
        return result_node, exec_duration
    def step(self, exec_callback: ExecCallbackType, current_step_number: int):
        log_prefix_main = f"{self.__class__.__name__.upper()}_STEP{current_step_number}"
        logger.info(f"{log_prefix_main}_START: Total Steps Configured: {self.acfg.steps}", extra={"verbose": True})
        t_step_start = time.time()
        submission_dir = self.cfg.workspace_dir / "submission"
        shutil.rmtree(submission_dir, ignore_errors=True); submission_dir.mkdir(exist_ok=True)
        self.current_step = current_step_number
        if not self.journal.nodes or self.data_preview is None: self.update_data_preview()
        parent_node = self.search_policy()
        result_node: Node; draft_flag = False; node_stage = "unknown"
        if parent_node is None:
            draft_flag = True; node_stage = "draft"; result_node = self._draft(parent_node)
        elif parent_node.is_buggy:
            node_stage = "debug"; result_node = self._debug(parent_node)
        else:
            node_stage = "improve"; result_node = self._improve(parent_node)

        # Process step
        reflection_applied = False
        result_node, exec_duration = self.process_step(exec_callback=exec_callback, result_node=result_node, node_stage=node_stage, current_step_number=current_step_number, use_reflection=draft_flag)
        print(reflection_applied)
        step_log_data = { # Prepare data for WandbLogger
            f"exec/exec_time_s": exec_duration,
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name": self.competition_name,
            "exec/exception_type": result_node.exc_type if result_node.exc_type else "None",
            f"code/estimated_quality": int(self._code_quality),
            f"eval/reflection_applied_successfully": 1 if reflection_applied and not result_node.is_buggy else 0,
            f"eval/effective_fix_this_step": 1 if result_node.effective_debug_step else 0,
            f"eval/validation_metric": result_node.metric.value if not result_node.is_buggy and result_node.metric else float('nan'),
            f"eval/submission_produced": 1 if (submission_dir / "submission.csv").exists() and not result_node.is_buggy else 0,
        }
        if self.wandb_logger: self.wandb_logger.log_step_data(step_log_data, current_step_number)

        result_node.stage = node_stage; result_node.exec_time = exec_duration
        self.journal.append(result_node)
        logger.info(f"{log_prefix_main}: Appended node {result_node.id} to journal. Journal size: {len(self.journal.nodes)}", extra={"verbose": True})

        # Cache best solution
        best_node = self.journal.get_best_node()
        if best_node and best_node.id == result_node.id :
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            if (submission_dir / "submission.csv").exists(): shutil.copy(submission_dir / "submission.csv", best_solution_dir / "submission.csv")
            with open(best_solution_dir / "solution.py", "w") as f: f.write(result_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f: f.write(str(result_node.id))

        log_step(step=current_step_number, total=self.acfg.steps, stage=node_stage,
                 is_buggy=result_node.is_buggy, exec_time=exec_duration,
                 metric=(result_node.metric.value if result_node.metric and result_node.metric.value is not None else None))
        t_step_end = time.time()


    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        log_prefix = ""
        # ... (implementation from Agent class) ...
        # This method is complex and has its own LLM call for feedback.
        # It should be inheritable directly if Agent's version is suitable.
        logger.info(f"{log_prefix}: Parsing execution result.", extra={"verbose": True})
        node.absorb_exec_result(exec_result)
        introduction = ("You are a Kaggle grandmaster ... evaluate the output ... empirical findings.")
        if self.acfg.obfuscate: introduction = ("You are an expert machine learning engineer ... evaluate the output ... empirical findings.")
        feedback_system_prompt = {
            "Introduction": introduction, "Task Description": self.task_desc,
            "Code Executed": prompt_utils_wrap_code(node.code),
            "Execution Output Log": prompt_utils_wrap_code(node.term_out, lang=""),
        }
        max_retries = self.acfg.feedback.get("retries", 3)
        review_response_dict: Optional[Dict[str, Any]] = None
        for attempt in range(max_retries):
            try:
                raw_response = query(system_message=feedback_system_prompt, user_message=None,
                                     func_spec=review_func_spec, model=self.acfg.feedback.model,
                                     temperature=self.acfg.feedback.temp,
                                     convert_system_to_user=self.acfg.convert_system_to_user,
                                     current_step=self.current_step)
                if not isinstance(raw_response, dict):
                    if isinstance(raw_response, str):
                        try: parsed_raw_response = json.loads(raw_response)
                        except Exception: parsed_raw_response = None
                        if isinstance(parsed_raw_response, dict): raw_response = parsed_raw_response
                        else: raw_response = None
                    else: raw_response = None
                review_response_dict = cast(Dict[str, Any], raw_response) if isinstance(raw_response, dict) else None
                if review_response_dict and all(k in review_response_dict for k in review_func_spec.json_schema["required"]): break
                else: review_response_dict = None
            except Exception as e: logger.error(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Error: {e}", exc_info=True, extra={"verbose": True})
            if attempt == max_retries - 1 and review_response_dict is None:
                review_response_dict = {"is_bug": True, "has_csv_submission": False, "summary": "LLM feedback failed.", "metric": None, "lower_is_better": True, "code_quality": 0}; break
        if review_response_dict is None: review_response_dict = {"is_bug": True, "has_csv_submission": False, "summary": "CRITICAL: review_response_dict None.", "metric": None, "lower_is_better": True, "code_quality": 0}
        metric_value = review_response_dict.get("metric"); self._code_quality = review_response_dict.get("code_quality", 0)
        if not isinstance(metric_value, (float, int)): metric_value = None
        if not isinstance(self._code_quality, (int, float)): self._code_quality = 0
        node.code_quality = int(self._code_quality)
        has_csv_submission_actual = (self.cfg.workspace_dir / "submission" / "submission.csv").exists()
        has_csv_submission_reported = review_response_dict.get("has_csv_submission", False)
        node.analysis = review_response_dict.get("summary", "Feedback LLM summary missing.")
        with open("review_response_dict.json", "w") as f: json.dump(review_response_dict, f)
        node.is_buggy = (review_response_dict.get("is_bug", True) or node.exc_type is not None or metric_value is None or not has_csv_submission_reported or not has_csv_submission_actual)
        if node.is_buggy:
            node.metric = WorstMetricValue()
        else: 
            node.metric = MetricValue(metric_value, maximize=not review_response_dict.get("lower_is_better", True))
        return node

#############################################################################
# PlannerAgent Implementation
#############################################################################
class PlannerAgent(Agent): # Inherit from Agent
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_run=None, # Replaced by wandb_logger
        wandb_logger: Optional['WandbLogger'] = None,
        competition_benchmarks=None,
    ):
        super().__init__(task_desc, cfg, journal, wandb_logger, competition_benchmarks)

        # Example: self.planner_specific_attribute = "some_value"
        # _code_quality is already in base Agent __init__


    # Override _query_llm_with_retries as it's specific to PlannerAgent's two-model approach
    def _query_llm_with_retries(
        self,
        query_type: str,
        system_prompt: Dict[str, Any],
        user_prompt: Dict[str, Any],
        model: str,
        temperature: float,
        planner_flag: bool,
        convert_system_to_user: bool,
        retries: int = 3,
    ) -> Any:

        completion_text = None
        log_prefix = f"PLANNER_AGENT_LLM_QUERY_{query_type.upper()}_STEP{self.current_step}"
        for attempt in range(retries):
            logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Sending request. Model: {model}, Temp: {temperature}, PlannerFlag: {planner_flag}", extra={"verbose": True})
            try:
                completion_text = query(
                    system_message=system_prompt, user_message=user_prompt,
                    model=model, temperature=temperature, planner=planner_flag,
                    current_step=self.current_step, convert_system_to_user=convert_system_to_user,
                    max_tokens=self.acfg.code.max_new_tokens
                )
                logger.info(f"{log_prefix}_ATTEMPT{attempt+1}: Received response.", extra={"verbose": True})
                return completion_text
            except Exception as e:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Error during LLM query: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: logger.error(f"{log_prefix}: All {retries} retries failed.", extra={"verbose": True}); return None
                time.sleep(2)
        return None

    def plan_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        system_prompt = get_planner_agent_plan_system_prompt()
        log_prefix = f"PlannerAgent_Plan_QUERY_STEP: {self.current_step}"
        completion_text = self._query_llm_with_retries(query_type="PLANNER_PLAN", system_prompt=system_prompt, user_prompt=user_prompt_dict,
                                                       model=self.acfg.code.planner_model, temperature=self.acfg.code.temp,
                                                       planner_flag=True, convert_system_to_user=self.acfg.convert_system_to_user, retries=retries)
        if completion_text is None: return "", "", ""
        summary, plan = extract_summary_and_plan(completion_text)
        if not (plan and summary): plan = plan or str(completion_text); summary = summary or "SUMMARY_EXTRACTION_FAILED"
        logger.info(f"{log_prefix}: Extracted summary and plan: {summary} \n ------ \n {plan} \n ------ \n END", extra={"verbose": True})
        return summary, plan, ""


    def code_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        system_prompt = get_planner_agent_code_system_prompt()
        log_prefix = f"CoderAgent_Code_QUERY_STEP: {self.current_step}"
        completion_text = self._query_llm_with_retries(query_type="PLANNER_CODER", system_prompt=system_prompt, user_prompt=user_prompt_dict,
                                                       model=self.acfg.code.model, temperature=self.acfg.code.temp,
                                                       planner_flag=False, convert_system_to_user=self.acfg.convert_system_to_user, retries=retries)
        if completion_text is None: return "", "", ""
        code = extract_code(completion_text)
        if not code:
            code = str(completion_text)
            return "", code, ""

        code = extract_code(completion_text)

        if code:
            logger.info(f"{log_prefix}: Successfully extracted code.", extra={"verbose": True})
            logger.debug(f"{log_prefix} \n EXTRACTED_CODE_START\n ----------- \n {code}\n ----------- \n EXTRACTED_CODE_END", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix}: Code extraction failed. Raw text: '{trim_long_string(str(completion_text))}'", extra={"verbose": True})
            code = str(completion_text) 

        return "", code, "" 

    def _code_segment_query(self, 
                                user_prompt_dict: Dict[str, Any], 
                                system_prompt_dict: Dict[str, Any], # Specific system prompt for the segment
                                retries: int = 1
                            ) -> str: # Returns only the code snippet string
            log_prefix = f"PLANNER_AGENT_CODE_SEGMENT_QUERY_STEP{self.current_step}"
            
            completion_text = self._query_llm_with_retries(
                query_type="CODE_SEGMENT_GENERATION",
                system_prompt=system_prompt_dict, 
                user_prompt=user_prompt_dict,
                model=self.acfg.code.model, # Coder model
                temperature=self.acfg.code.temp,
                planner_flag=False,
                convert_system_to_user=self.acfg.convert_system_to_user, 
                retries=retries
            )

            if completion_text is None:
                logger.error(f"{log_prefix}: LLM query returned None.")
                return "#LLM_QUERY_RETURNED_NONE_FOR_SEGMENT"

            code_snippet = extract_code(completion_text)
            
            if not code_snippet or not code_snippet.strip():
                logger.warning(f"{log_prefix}: Code extraction failed. Using raw completion text. Raw: {trim_long_string(completion_text)}")
                if "```python" not in completion_text and "import " not in completion_text and "def " not in completion_text:
                    code_snippet = f"#CODE_EXTRACTION_FAILED_OR_NOT_CODE_FOR_SEGMENT: {trim_long_string(completion_text)}"
                else:
                    code_snippet = completion_text
            
            return code_snippet.strip() if code_snippet else ""


    def _generate_code_segment(self,
                               segment_name: str,
                               task_summary: str,
                               master_plan_text: str,
                               code_accumulator: str) -> str:
        """Generates code for a single segment using the chained Coder."""
        log_prefix_segment = f"Code Chaine step {self.current_step }"
        logger.info(f"{log_prefix_segment}: Generating code.")

        system_prompt_getter = CHAINED_CODER_SYSTEM_PROMPT_GETTERS.get(segment_name)
        user_prompt_constructor = CHAINED_CODER_USER_PROMPT_CONSTRUCTORS.get(segment_name)

        if not system_prompt_getter or not user_prompt_constructor:
            logger.error(f"{log_prefix_segment}: No prompt definition found for segment '{segment_name}'.")
            return f"# ERROR: No prompt definition for segment: {segment_name}\n"

        segment_system_prompt = system_prompt_getter()
        segment_user_prompt = user_prompt_constructor(
            task_summary=task_summary,
            master_plan_text=master_plan_text,
            current_code_so_far=code_accumulator, # Pass the code built so far
            competition_name=self.competition_name,
            data_preview_content=self.data_preview
        )
        
        # Assuming self.code_query calls the Coder LLM (e.g., o4-mini)
        code_snippet = self._code_segment_query( # Call the new specialized method
            user_prompt_dict=segment_user_prompt,
            system_prompt_dict=segment_system_prompt,
            retries=self.acfg.get('coder_segment_retries', 1) 
        )
        if not code_snippet or code_snippet.strip() == "#CODE_FAILED" or not code_snippet.strip():
            logger.error(f"{log_prefix_segment}: Code generation failed or produced empty code.")
            return f"# FAILED TO GENERATE CODE FOR SEGMENT: {segment_name}\n"
        
        logger.info(f"{log_prefix_segment}: Successfully generated code snippet for {segment_name.replace(' ', '_')}.")
        logger.debug(f"{log_prefix_segment} {segment_name.replace(' ', '_')} Snippet: \n{code_snippet.strip()}\n{log_prefix_segment}_SNIPPET_END")
        return code_snippet.strip()


    def _draft_generate_code_chained(self, task_summary: str, master_plan_text: str) -> str:
        log_prefix_chain = f"PLANNER_AGENT_CHAINED_DRAFT_STEP{self.current_step}"
        logger.info(f"{log_prefix_chain}: Starting chained code generation for draft.")
        
        # Initial boilerplate for the script
        code_accumulator = f"# Script generated by AIDE PlannerAgent (Chained Coder) - Step {self.current_step}\n"
        code_accumulator += f"# Competition: {self.competition_name}\n"
        code_accumulator += f"# Task Summary: {task_summary.splitlines()[0]}...\n" # First line of summary
        code_accumulator += "# --- Master Plan ---\n"
        for i, plan_step_line in enumerate(master_plan_text.splitlines()):
             if plan_step_line.strip() and not plan_step_line.strip().startswith("##"): # Add non-empty lines, skip markdown headers
                code_accumulator += f"# {plan_step_line.strip()}\n"
        code_accumulator += "# --- End Master Plan ---\n\n"

        # Define the order of segments
        # Make sure segment_names match keys in CHAINED_CODER_USER_PROMPT_CONSTRUCTORS and SYSTEM_PROMPT_GETTERS
        segments_order = [
            "Setup & Imports",
            "Data Loading",
            "Data Preprocessing", # This now includes Dataset/Transforms/Split/Loaders
            "Modeling",
            "Training & Validation", # This now includes loss/optimizer and the loop
            "Prediction & Submission"
        ]

        for segment_name in segments_order:
            code_snippet = self._generate_code_segment(
                segment_name, task_summary, master_plan_text, code_accumulator
            )
            code_accumulator += code_snippet + "\n\n" # Add two newlines for separation
            if f"# FAILED TO GENERATE CODE FOR SEGMENT: {segment_name}" in code_snippet:
                logger.warning(f"{log_prefix_chain}: Halting chain due to failure in segment: {segment_name}")
                break # Optional: decide if you want to continue or halt on segment failure

        logger.info(f"{log_prefix_chain}: Chained code generation process complete.")
        return code_accumulator.strip()


    # Modify the existing _draft method to use this chained approach
    def _draft(self, parent_node=None) -> Node:
        log_prefix = f"PLANNER_AGENT_DRAFT_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting drafting process. Parent: {parent_node.id if parent_node else 'None'}")
        
        # 1. Generate Master Plan using the Planner model
        logger.info(f"{log_prefix}: Calling Planner for Task Summary and Master Plan.")
        plan_user_prompt = get_planner_agent_draft_plan_user_prompt(
            task_desc=self.task_desc, 
            journal_summary=self.journal.generate_summary(include_code=False), # Memory
            competition_name=self.competition_name, 
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        # self.plan_query uses self.acfg.code.planner_model
        task_summary, master_plan_text, _ = self.plan_query(
            plan_user_prompt, 
            retries=self.acfg.get('planner_retries', 3)
        )
        
        if not master_plan_text or master_plan_text.strip() == "": 
            logger.error(f"{log_prefix}: Master plan generation failed by Planner. Aborting draft.")
            final_plan_text = "MASTER_PLAN_FAILED_BY_PLANNER"
            generated_code = "# MASTER_PLAN_FAILED_BY_PLANNER - No code generated."
            final_summary = task_summary or "PLANNER_FAILED_TO_PRODUCE_SUMMARY_AND_PLAN"
        else:
            logger.info(f"{log_prefix}: Master Plan received from Planner. Proceeding to chained code generation.")
            logger.debug(f"{log_prefix}_MASTER_PLAN_START\n{master_plan_text}\n{log_prefix}_MASTER_PLAN_END")
            final_plan_text = master_plan_text # Store the full plan text for the node
            
            # 2. Generate Code via Chaining using the Coder model
            generated_code = self._draft_generate_code_chained(task_summary, master_plan_text)
            final_summary = task_summary # Use the summary from the planner

            if not generated_code or generated_code.strip().startswith("# FAILED TO GENERATE CODE FOR SEGMENT:") or generated_code.strip() == "# SEGMENT 1 (Data Loading & Initial Setup) FAILED TO GENERATE":
                 logger.error(f"{log_prefix}: Chained code generation resulted in failure or predominantly error messages.")
                 # Keep generated_code as is, it will contain error placeholders

        new_node = Node(plan=final_plan_text, code=generated_code, summary=final_summary, task_summary=final_summary, parent=parent_node)
        logger.info(f"{log_prefix}: Drafted new node {new_node.id} using Planner-ChainedCoder.")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        log_prefix = f"PLANNER_AGENT_IMPROVE_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting improvement for node {parent_node.id}.", extra={"verbose": True})
        plan_user_prompt = get_planner_agent_improve_plan_user_prompt(
            task_desc=self.task_desc, parent_node_code=parent_node.code,
            competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview)
        task_summary, improvement_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not improvement_plan: return Node(plan="IMPROVE_PLAN_FAILED", code=parent_node.code, summary=task_summary or "IMPROVE_PLAN_FAILED", parent=parent_node)
        code_user_prompt = get_planner_agent_improve_code_user_prompt(
            task_summary_from_planner=task_summary, improvement_plan_from_planner=improvement_plan,
            parent_node_code=parent_node.code, journal_summary=self.journal.generate_summary(include_code=False),
            competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview)
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not generated_code: generated_code = parent_node.code
        new_node = Node(plan=improvement_plan, code=generated_code, summary=task_summary, task_summary=task_summary, parent=parent_node)
        logger.info(f"{log_prefix}: Improved node {parent_node.id} to new node {new_node.id}.", extra={"verbose": True})
        return new_node


    def _debug(self, parent_node: Node) -> Node:
        log_prefix = f"PLANNER_AGENT_DEBUG_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting debugging for node {parent_node.id}.", extra={"verbose": True})
        plan_user_prompt = get_planner_agent_debug_plan_user_prompt(
            task_desc=self.task_desc, parent_node_code=parent_node.code,
            parent_node_term_out=parent_node.term_out,
            acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        bug_summary, fix_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not fix_plan: return Node(plan="DEBUG_PLAN_FAILED", code=parent_node.code, summary=bug_summary or "DEBUG_PLAN_FAILED", parent=parent_node)
        code_user_prompt = get_planner_agent_debug_code_user_prompt(
            bug_summary_from_planner=bug_summary, fix_plan_from_planner=fix_plan,
            parent_node_code=parent_node.code, parent_node_feedback=parent_node.analysis, parent_node_term_out=parent_node.term_out,
            competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview)
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not generated_code: generated_code = parent_node.code
        new_node = Node(plan=fix_plan, code=generated_code, summary=bug_summary, task_summary=bug_summary, parent=parent_node)
        logger.info(f"{log_prefix}: Debugged node {parent_node.id} to new node {new_node.id}.", extra={"verbose": True})
        return new_node


    def step(self, exec_callback: ExecCallbackType, current_step_number: int):


        log_prefix_main = f"PLANNER_AGENT_STEP{current_step_number}"
        t_step_start = time.time()
        submission_history = self.cfg.workspace_dir / "submission_history"
        submission_history.mkdir(exist_ok=True)
        submission_dir = self.cfg.workspace_dir / "submission"
        submission_csv = submission_dir / "submission.csv"
        if submission_csv.exists():
            backup_csv = submission_history / f"submission_step_{current_step_number}.csv"
            shutil.copy(submission_csv, backup_csv)
        logger.info(f"{log_prefix_main}: Clearing submission directory: {submission_dir}", extra={"verbose": True})
        shutil.rmtree(submission_dir, ignore_errors=True)
        submission_dir.mkdir(exist_ok=True)
        self.current_step = current_step_number

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        result_node: Node
        draft_flag = False
        node_stage = "unknown"

        if parent_node is None:
            draft_flag = True
            node_stage = "draft"
            logger.info(f"{log_prefix_main}: Stage selected: DRAFTING.", extra={"verbose": True})
            result_node = self._draft(parent_node)
        elif parent_node.is_buggy:
            node_stage = "debug"
            logger.info(f"{log_prefix_main}: Stage selected: DEBUGGING node {parent_node.id}.", extra={"verbose": True})
            result_node = self._debug(parent_node)
        else:
            node_stage = "improve"
            logger.info(f"{log_prefix_main}: Stage selected: IMPROVING node {parent_node.id}.", extra={"verbose": True})
            result_node = self._improve(parent_node)


        # Process step
        reflection_applied = False
        result_node, exec_duration = self.process_step(exec_callback=exec_callback, result_node=result_node, node_stage=node_stage, current_step_number=current_step_number, use_reflection=draft_flag)
        print(reflection_applied)


        logger.info(f"{log_prefix_main}: Preparing step log data for W&B.", extra={"verbose": True})

        step_log_data = {
            f"exec/exec_time_s": exec_duration,
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name": self.competition_name,
            "exec/exception_type": result_node.exc_type if result_node.exc_type else "None",
            f"code/estimated_quality": int(self._code_quality), # Assuming self._code_quality is set in parse_exec_result
            f"eval/reflection_applied_successfully": 1 if reflection_applied and not result_node.is_buggy else 0,
            f"eval/effective_fix_this_step": 1 if result_node.effective_debug_step else 0,
            f"eval/validation_metric": result_node.metric.value if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None else float('nan'),
            f"eval/submission_produced": 1 if (submission_dir / "submission.csv").exists() and not result_node.is_buggy else 0,
        }
        
        agent_validation_metrics_defined = False
        if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None:
            step_log_data[f"eval/validation_metric"] = result_node.metric.value
            agent_validation_metrics_defined = True
            if self.competition_benchmarks and wandb and self.wandb_logger:
                for threshold_name, key_suffix in [
                    ("median_threshold", "above_median"), ("gold_threshold", "gold_medal"),
                    ("silver_threshold", "silver_medal"), ("bronze_threshold", "bronze_medal")]:
                    flag_attr = f"_{key_suffix}_flags"
                    if not hasattr(self, flag_attr): setattr(self, flag_attr, [])
                    threshold_value = self.competition_benchmarks.get(threshold_name, float('inf'))
                    is_met = 1 if result_node.metric.value > threshold_value else 0
                    getattr(self, flag_attr).append(is_met)
                    true_count = sum(getattr(self, flag_attr))
                    false_count = len(getattr(self, flag_attr)) - true_count
                    table = wandb.Table(
                        data=[[key_suffix.replace('_', ' ').title(), true_count], [f"Not {key_suffix.replace('_', ' ').title()}", false_count]],
                        columns=["label", "count"])
                    step_log_data[f"plots/{key_suffix}_bar"] = wandb.plot.bar(table, "label", "count", title=f"{key_suffix.replace('_', ' ').title()} Steps")
        else:
            step_log_data[f"eval/validation_metric"] = float("nan")

        submission_path = submission_dir / "submission.csv"
        submission_exists = submission_path.exists()
        if not result_node.is_buggy and not submission_exists:
            logger.warning(f"{log_prefix_main}: Node {result_node.id} not buggy BUT submission.csv MISSING. Marking as buggy.", extra={"verbose": True})
            result_node.is_buggy = True; result_node.metric = WorstMetricValue()
            step_log_data[f"eval/validation_metric"] = float("nan"); step_log_data[f"eval/is_buggy"] = 1
            if agent_validation_metrics_defined and self._metric_hist and hasattr(result_node.metric, 'original_value_before_reset_to_worst') and \
               result_node.metric.original_value_before_reset_to_worst is not None and self._metric_hist[-1] == result_node.metric.original_value_before_reset_to_worst:
                self._metric_hist.pop()
        step_log_data[f"eval/submission_produced"] = 1 if submission_exists else 0

        if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None:
            self._metric_hist.append(result_node.metric.value)
        if wandb and self.wandb_logger:
            if len(self._metric_hist) >= 1:
                metric_table_data = [[v] for v in self._metric_hist if isinstance(v, (int, float))]
                if metric_table_data:
                    tbl = wandb.Table(data=metric_table_data, columns=["val"])
                    step_log_data["plots/val_metric_scatter"] = wandb.plot.scatter(tbl, "val", "val", title="Validation Metric Values")
            self._bug_flags.append(1 if result_node.is_buggy else 0)
            bug_count = sum(self._bug_flags); clean_count = len(self._bug_flags) - bug_count
            bug_table = wandb.Table(data=[["Buggy", bug_count], ["Clean", clean_count]], columns=["label", "count"])
            step_log_data["plots/bug_vs_clean"] = wandb.plot.bar(bug_table, "label", "count", title="Buggy vs Clean Steps")
            self._sub_flags.append(1 if submission_exists else 0)
            with_sub = sum(self._sub_flags); without_sub = len(self._sub_flags) - with_sub
            sub_table = wandb.Table(data=[["Has submission", with_sub], ["No submission", without_sub]], columns=["label", "count"])
            step_log_data["plots/submission_presence"] = wandb.plot.bar(sub_table, "label", "count", title="Submission Produced vs Missing")
        if self.wandb_logger:
           try:
               self.wandb_logger.log_step_data(
                    step_log_data, 
                    current_step_number,
                    result_node=result_node, # Pass the result_node
                    competition_benchmarks=self.competition_benchmarks # Pass benchmarks
                )
           except Exception as e_wandb: logger.error(f"{log_prefix_main}: Error logging to W&B: {e_wandb}", exc_info=True, extra={"verbose": True})

        result_node.stage = node_stage
        result_node.exec_time = exec_duration
        self.journal.append(result_node)
        logger.info(f"{log_prefix_main}: Appended node {result_node.id} to journal. Journal size: {len(self.journal.nodes)}", extra={"verbose": True})

        best_node = self.journal.get_best_node()
        if best_node and best_node.id == result_node.id:
            logger.info(f"{log_prefix_main}: Node {result_node.id} is new best (Metric: {best_node.metric.value if best_node.metric else 'N/A':.4f}). Caching.", extra={"verbose": True})
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            if submission_exists: shutil.copy(submission_path, best_solution_dir / "submission.csv")
            with open(best_solution_dir / "solution.py", "w") as f: f.write(result_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f: f.write(str(result_node.id))
        elif best_node:
            logger.info(f"{log_prefix_main}: Current best node is {best_node.id} (Metric: {best_node.metric.value if best_node.metric else 'N/A':.4f})", extra={"verbose": True})

        log_step(
            step=current_step_number, total=self.acfg.steps, stage=node_stage,
            is_buggy=result_node.is_buggy, exec_time=exec_duration,
            metric=(result_node.metric.value if result_node.metric and result_node.metric.value is not None else None)
        )