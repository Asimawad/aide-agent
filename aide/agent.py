# aide/agent.py
import shutil
import logging
import random
import json
import time
from pathlib import Path # Ensure Path is imported
from rich.console import Console # Keep for console output
from typing import Any, Callable, cast, Optional, Dict ,List,Tuple,Union # Added Dict
from .backend import query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview # data_preview.generate
from .utils.config import Config
from .utils.pretty_logging import log_step # logger from pretty_logging might conflict, be careful
from .backend.utils import ContextLengthExceededError # Add this import at the top of agent.py
from .utils.wandb_logger import WandbLogger
from .utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code, 
    trim_long_string,
    format_code,
    extract_plan, 
    extract_reflection_summary_and_revised_code,
    extract_summary_and_plan,
)
from .utils.self_reflection import (
    perform_two_step_reflection,
)
from .utils.metric import MetricValue, WorstMetricValue # Moved here for clarity

from .utils.prompt_base import (
    get_agent_draft_user_prompt,
    get_agent_improve_user_prompt,
    review_func_spec,
    get_agent_debug_user_prompt,
    CHAINED_CODER_USER_PROMPT_CONSTRUCTORS, # New
    CHAINED_CODER_SYSTEM_PROMPT_GETTERS,
    get_segment_reflection_system_prompt,
    get_segment_reflection_user_prompt,
    get_agent_system_prompt,
    get_agent_draft_system_prompt,
    get_agent_improve_system_prompt,
    get_agent_debug_system_prompt,
    get_planner_agent_draft_plan_user_prompt,
    get_planner_agent_draft_code_user_prompt,
    get_planner_agent_improve_plan_user_prompt,
    get_planner_agent_improve_code_user_prompt,
    get_planner_agent_debug_plan_user_prompt,
    get_planner_agent_debug_code_user_prompt,
    get_planner_agent_plan_system_prompt,
    get_planner_agent_code_system_prompt,
    wrap_code as prompt_utils_wrap_code, # Alias if local wrap_code is different
    get_chunked_reflection_system_prompt,
    get_chunked_reflection_user_prompt
)


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

class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_logger: Optional[WandbLogger] = None, 
        competition_benchmarks: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(task_desc, dict):
            from .backend import compile_prompt_to_md
            self.task_desc = compile_prompt_to_md(task_desc)
        else:
            self.task_desc = task_desc

        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.wandb_logger = wandb_logger 
        self.competition_benchmarks = competition_benchmarks
        self.competition_name = self.cfg.competition_name
        
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        self._prev_buggy: bool = False # Tracks buggy status *before* reflection for current step logic
        self._code_quality: float = 0.0 # Set by parse_exec_result
        self.reflection_applied = False

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        # console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Search Policy")

        log_prefix_base = f"Search_Policy-Step: {self.current_step}"
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

    def plan_and_code_query(self, user_prompt_dict: Dict[str, Any], excute: bool, system_prompt_dict=None, retries: int = 3) -> tuple[str, str, str]: 
        if system_prompt_dict is None: system_prompt_dict = get_agent_system_prompt()
        log_prefix = f"Step: {self.current_step}" 
        completion_text = None
        for attempt in range(retries):

            try:
                completion_text = query(
                    system_message=system_prompt_dict, user_message=user_prompt_dict,
                    model=self.acfg.code.model, temperature=self.acfg.code.temp,
                    max_tokens=self.acfg.code.max_new_tokens, current_step=self.current_step,
                    inference_engine=self.cfg.inference_engine,
                    num_responses=self.acfg.code.num_return_sequences,
                    convert_system_to_user=self.acfg.convert_system_to_user)
            
            
            except ContextLengthExceededError as cle:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Context length exceeded: {cle}. Failing this operation.", extra={"verbose": True})
                return "", f"LLM Query Error: Context Length Exceeded - {str(cle)}", "CONTEXT_LENGTH_EXCEEDED"
            
            except Exception as e: 
                if "ContextLengthExceededError" in str(type(e)) or "context length" in str(e).lower(): # Heuristic check
                    logger.error(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Context length exceeded: {e}. Failing this operation.", extra={"verbose": True})
                    return "", f"LLM Query Error: Context Length Exceeded - {str(e)}", "CONTEXT_LENGTH_EXCEEDED"
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Query failed: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: return "", f"LLM Query Error: {e}", "LLM_QUERY_ERROR"
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5)) # Make delay configurable
                continue
            
            if completion_text == "Exceeded context length limit":
                return "", completion_text or "No LLM response received", "EXTRACTION_FAILED"
            code = extract_code(completion_text)            
            nl_text = extract_text_up_to_code(completion_text)
            if code and nl_text:
                logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Successfully extracted plan and code.", extra={"verbose": True})
                return nl_text, code, "execution_summary_placeholder"
            logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Plan or code extraction failed. Raw text: '{trim_long_string(completion_text)}'", extra={"verbose": True})
        logger.error(f"{log_prefix}: All {retries} attempts for plan+code extraction failed.", extra={"verbose": True})
        return "", completion_text or "No LLM response received", "EXTRACTION_FAILED"


    def _draft(self, parent_node=None) -> Node:
        log_prefix_base = f"{self.__class__.__name__}_DRAFT_STEP:{self.current_step}" 
        logger.info(f"{log_prefix_base}: Starting drafting. Parent: {parent_node.id if parent_node else 'None'}", extra={"verbose": True})
        draft_sys_prompt=get_agent_draft_system_prompt()
        journal_summary=self.journal.generate_summary(include_code=False)
        logger.info(f"{log_prefix_base}: Journal summary: {journal_summary}", extra={"verbose": True})

        prompt_user_message = get_agent_draft_user_prompt( 
            task_desc=self.task_desc, journal_summary=journal_summary,
            competition_name=self.competition_name, obfuscate=self.acfg.obfuscate,
            acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        agent_plan_for_step, generated_code, exec_summary = (
            self.plan_and_code_query(user_prompt_dict=prompt_user_message, excute=False,system_prompt_dict = draft_sys_prompt, retries=self.acfg.get('query_retries', 1)))
        if not agent_plan_for_step: agent_plan_for_step = "PLAN_GENERATION_FAILED"
        if not generated_code: generated_code = "# CODE_GENERATION_FAILED"
        logger.debug(f"{log_prefix_base}_DRAFT_PLAN_START\n{agent_plan_for_step}\n{log_prefix_base}_DRAFT_PLAN_END", extra={"verbose": True})
        logger.debug(f"{log_prefix_base}_DRAFT_CODE_RAW_START\n{generated_code}\n{log_prefix_base}_DRAFT_CODE_RAW_END", extra={"verbose": True})
        new_node = Node(plan=agent_plan_for_step, code=generated_code, summary=exec_summary)
        if parent_node: new_node.parent = parent_node
        logger.info(f"{log_prefix_base}: Drafted new node {new_node.id}.", extra={"verbose": True})
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        log_prefix_base = f"{self.__class__.__name__}_IMPROVE_STEP{self.current_step}"
        logger.info(f"{log_prefix_base}: Starting improvement for node {parent_node.id}.", extra={"verbose": True})
        improve_sys_prompt = get_agent_improve_system_prompt() # From prompt_utils
        prompt_user_message = get_agent_improve_user_prompt(
            task_desc=self.task_desc, journal_summary=self.journal.generate_summary(include_code=False),
            competition_name=self.competition_name, parent_node_code=parent_node.code)
        plan, code, _ = self.plan_and_code_query(prompt_user_message, excute=False, system_prompt_dict=improve_sys_prompt, retries=self.acfg.get('query_retries', 1))
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
        debug_sys_prompt = get_agent_debug_system_prompt() # Use the new debug system prompt
        prompt_user_message = get_agent_debug_user_prompt(
            task_desc=self.task_desc, competition_name=self.competition_name,
            parent_node_code=parent_node.code, parent_node_term_out=parent_node.term_out,
            parent_node_feedback=parent_node.analysis, 
            acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        plan, code, _ = self.plan_and_code_query(prompt_user_message, excute=False, system_prompt_dict=debug_sys_prompt, retries=self.acfg.get('query_retries', 1))

        if not plan: plan = "DEBUG_PLAN_FAILED"
        if not code: code = parent_node.code
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"{log_prefix_base}: Debugged node {parent_node.id} to create new node {new_node.id}", extra={"verbose": True})
        logger.debug(f"{log_prefix_base}_DEBUG_PLAN_START\n{plan}\n{log_prefix_base}_DEBUG_PLAN_END", extra={"verbose": True})
        logger.debug(f"{log_prefix_base}_DEBUG_CODE_START\n{wrap_code(code)}\n{log_prefix_base}_DEBUG_CODE_END", extra={"verbose": True})
        return new_node

    def reflect(self, node: Node) -> tuple[str, str]:
        log_prefix_base = f"{self.__class__.__name__.upper()}_REFLECT_STEP{self.current_step}_NODE{node.id}"
        logger.info(f"{log_prefix_base}: Initiating self-reflection.", extra={"verbose": True})
        try:
            reflection_plan, revised_code = perform_two_step_reflection(
                code=node.code, analysis=node.analysis, term_out=node.term_out,
                task_desc=self.task_desc, model_name=self.cfg.agent.code.planner_model, 
                temperature=self.acfg.code.temp, convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query, wrap_code_func=prompt_utils_wrap_code, extract_code_func=extract_code,
                current_step=self.current_step )
        except Exception as e:
            logger.error(f"{log_prefix_base}: Error during self-reflection call: {e}", exc_info=True, extra={"verbose": True})
            return f"REFLECTION_ERROR: {e}", node.code
        if revised_code and revised_code.strip() and revised_code != node.code: logger.info(f"{log_prefix_base}: Self-reflection resulted in code changes.", extra={"verbose": True})
        elif "No specific errors found requiring changes." in reflection_plan : logger.info(f"{log_prefix_base}: Self-reflection found no errors requiring changes.", extra={"verbose": True})
        else: logger.warning(f"{log_prefix_base}: Self-reflection finished, but revised code is same as original or empty. Plan: {trim_long_string(reflection_plan)}", extra={"verbose": True})
        return reflection_plan, revised_code
    
    def update_data_preview(self):
        log_prefix = f"Data_Preview-Step: {self.current_step}"

        logger.info(f"{log_prefix}: Updating data preview.", extra={"verbose": True})
        try:
            self.data_preview = data_preview.generate(self.cfg.workspace_dir / "input")

            if self.current_step == 1:
                logger.info(f"{log_prefix}: Data preview: {self.data_preview}", extra={"verbose": True})
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
                result_node.code = reflection_code; self.reflection_applied = True
                exec_start_time_reflect = time.time()
                exec_result_reflect = exec_callback(result_node.code, reset_session=True)
                exec_duration = time.time() - exec_start_time_reflect
                result_node = self.parse_exec_result(node=result_node, exec_result=exec_result_reflect)
                logger.info(f"Reflection applied: {self.reflection_applied} and result_node.is_buggy: {result_node.is_buggy}", extra={"verbose": True})


            if buggy_status_before_reflection and not result_node.is_buggy:
                result_node.effective_debug_step = True; result_node.effective_reflections = self.reflection_applied
                console.print(f"[bold green]Effective debug step: {result_node.effective_debug_step} and effective reflections: {result_node.effective_reflections}[/bold green]")
            else:
                result_node.effective_debug_step = False; result_node.effective_reflections = False
                console.print(f"[bold red]Effective debug step: {result_node.effective_debug_step} and effective reflections: {result_node.effective_reflections}[/bold red]")
        self._prev_buggy = result_node.is_buggy
        if result_node.is_buggy:

            console.print(f"[bold red]---------[/bold red]\n") # Console output
            console.print(f"[bold red]stage: {node_stage}[/bold red]") # Console output
            console.print(f"[bold red]Result: Buggy[/bold red]") # Console output
            console.print(f"[bold red]Feedback: {result_node.analysis}[/bold red]") # Console output
            # log them to the verbose file
            logger.debug(f"stage: {node_stage}", extra={"verbose": True})
            logger.debug(f"Result: Buggy", extra={"verbose": True})
            logger.debug(f"Feedback: {result_node.analysis}", extra={"verbose": True})
        else: 
            console.print(f"[bold green]---------[/bold green]\n") # Console output
            console.print(f"[bold green]stage: {node_stage}[/bold green]")
            console.print(f"[bold green]Result: Not Buggy[/bold green]") # Console output
            console.print(f"[bold green]Feedback: {result_node.analysis}[/bold green]") # Console output
            logger.debug(f"stage: {node_stage}", extra={"verbose": True})
            logger.debug(f"Result: Not Buggy", extra={"verbose": True})
            logger.debug(f"Feedback: {result_node.analysis}", extra={"verbose": True})
        return result_node, exec_duration


    def step(self, exec_callback: ExecCallbackType, current_step_number: int):
        log_prefix_main = f"{self.__class__.__name__.upper()}_STEP{current_step_number}"
        logger.info(f"{log_prefix_main}_START: Total Steps Configured: {self.acfg.steps}", extra={"verbose": True})
        t_step_start = time.time()
        
        # Define submission_dir for this step
        submission_dir_this_step = self.cfg.workspace_dir / "submission"
        
        # Backup and clear submission directory
        submission_history_dir_for_run = Path(self.cfg.log_dir) / "submission_history" # Centralized history
        submission_history_dir_for_run.mkdir(parents=True, exist_ok=True)
        current_submission_csv = submission_dir_this_step / "submission.csv"
        if current_submission_csv.exists(): # If a submission from PREVIOUS step exists
            try:
                backup_name = f"step_{current_step_number-1}_submission.csv" if current_step_number > 1 else "initial_submission.csv"
                shutil.copy2(current_submission_csv, submission_history_dir_for_run / backup_name)
                logger.info(f"{log_prefix_main}: Backed up previous submission to {backup_name}", extra={"verbose": True})
            except Exception as e_backup:
                logger.error(f"{log_prefix_main}: Error backing up submission: {e_backup}", extra={"verbose": True})

        shutil.rmtree(submission_dir_this_step, ignore_errors=True)
        submission_dir_this_step.mkdir(exist_ok=True)
        
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
        result_node, exec_duration = self.process_step(exec_callback=exec_callback, result_node=result_node, node_stage=node_stage, current_step_number=current_step_number, use_reflection=draft_flag)
        # Final check for submission file existence AFTER all potential executions for this step
        submission_path_final = submission_dir_this_step / "submission.csv"
        submission_exists_final = submission_path_final.exists()

        if not result_node.is_buggy and not submission_exists_final:
            logger.warning(f"Node {result_node.id} was NOT buggy BUT final submission.csv MISSING. Marking as buggy.", extra={"verbose": True})
            result_node.is_buggy = True 
            original_metric_val = result_node.metric.value if result_node.metric else None
            result_node.metric = WorstMetricValue()
            if original_metric_val is not None and result_node.metric is not None:
                 result_node.metric.original_value_before_reset_to_worst = original_metric_val
                    


        # Base data for logger, more complex plots will be derived by logger from result_node
        base_step_log_data = {
            f"exec/exec_time_s": exec_duration, # Total time for the step
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name": self.competition_name,
            "exec/exception_type": result_node.exc_type if result_node.exc_type else "None",
            f"code/estimated_quality": int(result_node.code_quality), # Use node's quality
            f"eval/reflection_applied_and_successful": 1 if self.reflection_applied and not result_node.is_buggy else 0,
            f"eval/effective_fix_this_step": 1 if result_node.effective_debug_step else 0, 
            f"eval/effective_reflection_fix_this_step": 1 if result_node.effective_reflections else 0,
            # eval/validation_metric and eval/submission_produced will be set/overridden by WandbLogger
        }
        
        if self.wandb_logger and self.wandb_logger.wandb_run:
            self.wandb_logger.log_step_data(
                base_step_log_data=base_step_log_data, 
                result_node=result_node, # Pass the finalized node
                current_step_number=current_step_number,
                current_submission_dir=submission_dir_this_step # Pass current submission dir
            )

        result_node.stage = node_stage
        result_node.exec_time = exec_duration # Store total exec time on node
        self.journal.append(result_node)
        
        best_node = self.journal.get_best_node()
        if best_node and best_node.id == result_node.id :
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_submission_dir = self.cfg.workspace_dir / "best_submission" 
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            best_submission_dir.mkdir(exist_ok=True, parents=True)

            if submission_exists_final: 
                 shutil.copy2(submission_path_final, best_submission_dir / "submission.csv")
                 logger.info(f"{log_prefix_main}: Cached best submission.csv to {best_submission_dir}")
            
            with open(best_solution_dir / "solution.py", "w") as f: f.write(result_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f: f.write(str(result_node.id))
            logger.info(f"{log_prefix_main}: Cached best solution code for node {result_node.id}")

        log_step(step=current_step_number, total=self.acfg.steps, stage=node_stage,
                 is_buggy=result_node.is_buggy, exec_time=exec_duration,
                 metric=(result_node.metric.value if result_node.metric and result_node.metric.value is not None else None))
        t_step_end = time.time()
        logger.info(f"{log_prefix_main}_END: Duration: {t_step_end - t_step_start:.2f}s", extra={"verbose": True})

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        log_prefix = f"{self.__class__.__name__.upper()}_PARSE_EXEC_STEP{self.current_step}_NODE{node.id}"
        logger.info(f"{log_prefix}: Parsing execution result.", extra={"verbose": True})
        node.absorb_exec_result(exec_result)
        introduction = ("You are a Kaggle grandmaster ... evaluate the output ... empirical findings.")
        if self.acfg.obfuscate: introduction = ("You are an expert machine learning engineer ... evaluate the output ... empirical findings.")
        
        feedback_system_prompt = {
            "Introduction": introduction, "Task Description": self.task_desc,
            "Code Executed": prompt_utils_wrap_code(node.code),
            "Execution Output Log": prompt_utils_wrap_code(node.term_out, lang=""),}
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
                else: 
                    logger.warning(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Response missing required keys or not a dict. Response: {review_response_dict}")
                    review_response_dict = None 
            except Exception as e: logger.error(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Error: {e}", exc_info=True, extra={"verbose": True})
            if attempt == max_retries - 1 and review_response_dict is None:
                review_response_dict = {"is_bug": True, "has_csv_submission": False, "summary": "LLM feedback failed after retries.", "metric": None, "lower_is_better": True, "code_quality": 0}; break
        if review_response_dict is None: review_response_dict = {"is_bug": True, "has_csv_submission": False, "summary": "CRITICAL: review_response_dict was None after loop.", "metric": None, "lower_is_better": True, "code_quality": 0}
        
        metric_value = review_response_dict.get("metric")
        # self._code_quality is set here, which is used by the logger
        self._code_quality = review_response_dict.get("code_quality", 0) 
        if not isinstance(metric_value, (float, int)): metric_value = None
        if not isinstance(self._code_quality, (int, float)): self._code_quality = 0 
        node.code_quality = int(self._code_quality) 

        submission_dir_for_check = self.cfg.workspace_dir / "submission" # Use current submission dir
        has_csv_submission_actual = (submission_dir_for_check / "submission.csv").exists()
        has_csv_submission_reported_by_llm = review_response_dict.get("has_csv_submission", False)
        
        node.analysis = review_response_dict.get("summary", "Feedback LLM summary missing.")
        
        node.is_buggy = (
            review_response_dict.get("is_bug", True) 
            or node.exc_type is not None
            or metric_value is None 
            or not has_csv_submission_reported_by_llm 
            or not has_csv_submission_actual 
        )
        
        bug_reasons = []
        if review_response_dict.get("is_bug", True): bug_reasons.append("LLM judged buggy")
        if node.exc_type is not None: bug_reasons.append(f"Exception ({node.exc_type})")
        if metric_value is None: bug_reasons.append("Metric missing/invalid")
        if not has_csv_submission_reported_by_llm: bug_reasons.append("LLM reported no CSV")
        if not has_csv_submission_actual: bug_reasons.append("Actual CSV not found")
        
        if node.is_buggy:
            logger.info(f"{log_prefix}:\n\n determined as BUGGY. \n\nReasons: {'; '.join(bug_reasons) if bug_reasons else 'None explicitly stated'}", extra={"verbose":True})
            node.metric = WorstMetricValue()
            if metric_value is not None and node.metric is not None: 
                 node.metric.original_value_before_reset_to_worst = metric_value
        else: 
            logger.info(f"{log_prefix}:\n\n determined as NOT BUGGY. \n\nReasons: {'; '.join(bug_reasons) if bug_reasons else 'None explicitly stated'}", extra={"verbose":True})
            node.metric = MetricValue(metric_value, maximize=not review_response_dict.get("lower_is_better", True))
        
        return node

class SelfDebugAgent(Agent): # Inherit from Agent
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


    def _execute_and_evaluate_node(self, 
                                   node_to_process: Node, 
                                   exec_callback: ExecCallbackType, 
                                   log_prefix_label: str = "Initial") -> Tuple[Node, float]: # Returns (processed_node, exec_duration)
        """
        Helper to execute a node's code, parse the result, and update the node.
        Returns the processed node and execution duration.
        """
        logger.info(f"{log_prefix_label} Execution: Executing code for Node {node_to_process.id} (Step {self.current_step}).", extra={"verbose": True})
        
        exec_start_time = time.time()
        # Ensure that the exec_callback is available on self, or pass it in.
        # Assuming self.exec_callback exists if you are calling this from within another method of the class that has it.
        # If exec_callback is passed to step(), then it should be passed here too.
        exec_result = exec_callback(node_to_process.code, reset_session=True) 
        exec_duration = time.time() - exec_start_time
        logger.info(f"{log_prefix_label} Execution: Code execution finished in {exec_duration:.2f}s for Node {node_to_process.id}.", extra={"verbose": True})

        # parse_exec_result updates the node_to_process in-place and returns it
        processed_node = self.parse_exec_result(node=node_to_process, exec_result=exec_result)
        # Note: self.parse_exec_result also handles setting node.is_buggy, node.metric, node.analysis etc.
        
        logger.info(f"{log_prefix_label} Evaluation: Node {processed_node.id} - Buggy: {processed_node.is_buggy}, Metric: {processed_node.metric.value if processed_node.metric else 'N/A'}", extra={"verbose":True})
        return processed_node, exec_duration


    def step(self, exec_callback: ExecCallbackType, current_step_number: int):
        log_prefix_main_step = f"{self.__class__.__name__.upper()}_AIDE_STEP_{current_step_number}" # Changed from self.current_step for clarity
        logger.info(f"{log_prefix_main_step}_START: Total AIDE Steps Configured: {self.acfg.steps}")
        t_step_start = time.time()
        
        # --- Submission Directory Handling (as before) ---
        submission_dir_this_step = self.cfg.workspace_dir / "submission"
        submission_history_dir_for_run = Path(self.cfg.log_dir) / "submission_history"
        submission_history_dir_for_run.mkdir(parents=True, exist_ok=True)
        current_submission_csv = submission_dir_this_step / "submission.csv"
        if current_submission_csv.exists():
            try:
                backup_name = f"step_{current_step_number-1}_submission.csv" if current_step_number > 1 else "initial_submission.csv"
                shutil.copy2(current_submission_csv, submission_history_dir_for_run / backup_name)
            except Exception as e_backup:
                logger.error(f"{log_prefix_main_step}: Error backing up submission: {e_backup}")
        shutil.rmtree(submission_dir_this_step, ignore_errors=True)
        submission_dir_this_step.mkdir(exist_ok=True)
        
        self.current_step = current_step_number # Ensure self.current_step is updated for logging within helpers
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
        
        # --- Initial Node Generation (Draft, Improve, Debug) ---
        parent_node_from_search = self.search_policy() # Renamed for clarity
        
        # This 'result_node' is the primary node for this AIDE step.
        # It starts as a candidate from _draft, _improve, or _debug.
        # If iterative debugging happens, this same 'result_node' object will be modified in place.
        result_node: Node 
        initial_node_stage: str = "unknown"
        apply_self_reflection_after_initial_op: bool = False

        if parent_node_from_search is None:
            initial_node_stage = "draft"
            logger.info(f"{log_prefix_main_step}: Search policy selected DRAFT.")
            result_node = self._draft(parent_node=None) # _draft returns a NEW Node
            apply_self_reflection_after_initial_op = True # Reflection often useful for first drafts
        elif parent_node_from_search.is_buggy:
            initial_node_stage = "debug"
            logger.info(f"{log_prefix_main_step}: Search policy selected DEBUG for Node {parent_node_from_search.id}.")
            result_node = self._debug(parent_node=parent_node_from_search) # _debug returns a NEW Node
        else:
            initial_node_stage = "improve"
            logger.info(f"{log_prefix_main_step}: Search policy selected IMPROVE for Node {parent_node_from_search.id}.")
            result_node = self._improve(parent_node=parent_node_from_search) # _improve returns a NEW Node
        
        result_node.stage = initial_node_stage # Set the initial stage

        # --- Initial Execution and Evaluation of the Generated/Improved/Debugged Node ---
        # The process_step logic is now partially in _execute_and_evaluate_node
        # and the reflection/S* iterative debug logic is below.
        
        logger.info(f"{log_prefix_main_step}: Performing initial execution for Node {result_node.id} (Stage: {initial_node_stage}).")
        result_node, exec_duration = self._execute_and_evaluate_node(
            result_node, exec_callback, f"InitialOp_{initial_node_stage.upper()}"
        )
        # result_node is now updated with exec results, is_buggy, metric, analysis.
        # We will append it to journal *after* potential reflection and iterative debugging.

        # --- Self-Reflection (Optional, based on strategy) ---
        # Your existing self-reflection logic:
        buggy_status_before_reflection = result_node.is_buggy
        self.reflection_applied = False # Reset for this step
        
        # Check if reflection should be applied for this ITS_Strategy and node state
        # E.g., only for "self-reflection" strategy and if buggy and draft_flag was true
        should_apply_reflection = (
            apply_self_reflection_after_initial_op and # True if initial op was _draft
            self.acfg.ITS_Strategy == "self-reflection" and 
            result_node.is_buggy
        )

        if should_apply_reflection:
            logger.info(f"{log_prefix_main_step}: Node {result_node.id} is buggy after initial {initial_node_stage}. Applying self-reflection.")
            reflection_plan, reflection_code = self.reflect(node=result_node) # self.reflect is from base Agent
            
            if reflection_code and reflection_code.strip() and reflection_code != result_node.code:
                result_node.code = reflection_code
                result_node.plan += f"\n\n--- Self-Reflection Applied ---\n{reflection_plan}" # Append reflection plan
                self.reflection_applied = True
                
                logger.info(f"{log_prefix_main_step}: Re-executing Node {result_node.id} after self-reflection.")
                result_node, reflection_exec_duration = self._execute_and_evaluate_node(
                    result_node, exec_callback, "PostReflection"
                )
                exec_duration += reflection_exec_duration # Add to total exec time for the step
            else:
                logger.info(f"{log_prefix_main_step}: Self-reflection did not result in code changes for Node {result_node.id}.")

        # Update effective_debug_step based on reflection outcome
        if buggy_status_before_reflection and not result_node.is_buggy and self.reflection_applied:
            result_node.effective_debug_step = True # Count reflection as an effective debug
            result_node.effective_reflections = True 
            # console.print(f"[bold green]Self-reflection fixed the bug for Node {result_node.id}![/bold green]")
        elif self.reflection_applied: # Reflection applied but still buggy or was not buggy before
            result_node.effective_reflections = False # Or some other flag for "reflection attempted"


        # === S* Style Iterative Debugging Integration START ===
        # Get max_rounds from config, default to 0 if not present or not TOTAgent context
        # Ensure s_star_iterative_debug is a section in your agent config, e.g., agent.s_star_iterative_debug.max_rounds
        max_s_star_rounds = 0
        if hasattr(self.acfg, 's_star_iterative_debug') and hasattr(self.acfg.s_star_iterative_debug, 'max_rounds'):
            max_s_star_rounds = self.acfg.s_star_iterative_debug.max_rounds
        else: # Fallback if not defined, or log a warning
            # logger.debug(f"{log_prefix_main_step}: s_star_iterative_debug.max_rounds not in config or not TOTAgent. Skipping S* iterative debug.")
            pass


        if result_node.is_buggy and max_s_star_rounds > 0:
            logger.info(f"{log_prefix_main_step}: Node {result_node.id} is buggy. Starting S*-style iterative debugging (max {max_s_star_rounds} rounds).")
            
            # Create a temporary node for iterative debugging lineage. 
            # The 'result_node' will be the one that finally gets added to the journal for this AIDE step.
            # Each debug attempt generates a *new candidate* based on the *current state* of result_node.
            
            for debug_attempt in range(max_s_star_rounds):
                log_prefix_iter_debug = f"{log_prefix_main_step}_IterDebugAttempt_{debug_attempt + 1}"
                logger.info(f"{log_prefix_iter_debug}: For Node {result_node.id}.")

                # self._debug expects a parent_node whose attributes (code, term_out, analysis) are used.
                # So, result_node (which is currently buggy and has the latest error info) acts as this "parent_node" for the _debug call.
                # _debug will create a *new* Node object with the proposed fix.
                debug_candidate_node = self._debug(parent_node=result_node) # Pass current result_node as parent for debug context
                
                # Execute the debug_candidate_node's code
                logger.info(f"{log_prefix_iter_debug}: Executing debug candidate (New Node ID: {debug_candidate_node.id}).")
                processed_debug_candidate_node, iter_debug_exec_duration = self._execute_and_evaluate_node(
                    debug_candidate_node, exec_callback, f"IterDebug_{debug_attempt+1}"
                )
                exec_duration += iter_debug_exec_duration

                # Now, decide if this debug attempt was successful *for the main result_node*
                if not processed_debug_candidate_node.is_buggy:
                    logger.info(f"{log_prefix_iter_debug}: SUCCEEDED. Updating main result_node (ID: {result_node.id}) with successful debug from candidate (ID: {processed_debug_candidate_node.id}).")
                    # Transfer successful state to the original result_node for this AIDE step
                    result_node.code = processed_debug_candidate_node.code
                    result_node.plan = processed_debug_candidate_node.plan # Debug might change the plan
                    result_node.summary = processed_debug_candidate_node.summary
                    result_node.absorb_exec_result(processed_debug_candidate_node.exec_result_obj_for_journal) # Make sure this method exists and sets all exec fields
                    result_node.analysis = processed_debug_candidate_node.analysis
                    result_node.metric = processed_debug_candidate_node.metric
                    result_node.code_quality = processed_debug_candidate_node.code_quality
                    result_node.is_buggy = False # Explicitly mark as not buggy
                    result_node.effective_debug_step = True # Mark this AIDE step as an effective debug
                    result_node.s_star_debug_rounds_applied = debug_attempt + 1 # Custom field
                    # The parentage of result_node is already set from initial D/I/D. 
                    # The debug_candidate_node's lineage shows it came from result_node, which is good for tracing in journal.
                    self.journal.append(processed_debug_candidate_node) # Add the successful debug attempt to journal
                    break # Exit iterative debug loop
                else:
                    logger.info(f"{log_prefix_iter_debug}: FAILED. Error: {processed_debug_candidate_node.exc_type}. Main result_node (ID: {result_node.id}) remains buggy.")
                    # The result_node's state (code, term_out, analysis) needs to be updated with this failed attempt's info
                    # so the *next* _debug call in the loop gets the latest error.
                    result_node.code = processed_debug_candidate_node.code # Take the latest attempt's code
                    result_node.plan = processed_debug_candidate_node.plan
                    result_node.summary = processed_debug_candidate_node.summary
                    result_node.absorb_exec_result(processed_debug_candidate_node.exec_result_obj_for_journal)
                    result_node.analysis = processed_debug_candidate_node.analysis
                    result_node.metric = WorstMetricValue() # Still buggy
                    result_node.is_buggy = True
                    self.journal.append(processed_debug_candidate_node) # Add the FAILED debug attempt to journal too

            if result_node.is_buggy: # If still buggy after all S* rounds
                logger.info(f"{log_prefix_main_step}: Node {result_node.id} remains buggy after {max_s_star_rounds} S*-style iterative debug rounds.")
        # === S* Style Iterative Debugging Integration END ===


        # --- Final Journaling and Logging for the AIDE step ---

        # If iterative debug fixed it, stage might still be "draft" but effective_debug_step is True.
        result_node.stage = initial_node_stage # Keep the original stage of the operation for this AIDE step

        self.journal.append(result_node) # Append the final state of result_node for this AIDE step

        # Display feedback on console
        if result_node.is_buggy:
            console.print(f"[bold red]Step {current_step_number}: {initial_node_stage.upper()} -> BUGGY. Analysis: {result_node.analysis}[/]")
        else:
            metric_str = f"{result_node.metric.value:.4f}" if result_node.metric and result_node.metric.value is not None else "N/A"
            console.print(f"[bold green]Step {current_step_number}: {initial_node_stage.upper()} -> SUCCESS. Metric: {metric_str}. Analysis: {result_node.analysis}[/]")

        # W&B Logging (as before)
        submission_path_final = submission_dir_this_step / "submission.csv"
        base_step_log_data = {
            "exec/exec_time_s": exec_duration, # Total time for the AIDE step
            "eval/is_buggy": 1 if result_node.is_buggy else 0,
            "progress/current_step": current_step_number,
            "progress/competition_name": self.competition_name,
            "exec/exception_type": result_node.exc_type if result_node.exc_type else "None",
            "code/estimated_quality": int(result_node.code_quality),
            "eval/reflection_applied_this_step": 1 if self.reflection_applied else 0, # Changed key for clarity
            "eval/reflection_successful_this_step": 1 if self.reflection_applied and not result_node.is_buggy and buggy_status_before_reflection else 0,
            "eval/s_star_debug_rounds_if_applied": getattr(result_node, 's_star_debug_rounds_applied', 0),
            "eval/initial_op_was_effective_fix": 1 if not parent_node_from_search.is_buggy and initial_node_stage == "debug" and not result_node.is_buggy and not self.reflection_applied and not hasattr(result_node, 's_star_debug_rounds_applied') else 0,
        }
        if self.wandb_logger and self.wandb_logger.wandb_run:
            self.wandb_logger.log_step_data(base_step_log_data, result_node, current_step_number, submission_dir_this_step)
        
        # Caching best solution (as before)
        best_node_overall = self.journal.get_best_node() # This gets best from entire journal
        if best_node_overall and best_node_overall.id == result_node.id and not result_node.is_buggy: # Only cache if current step produced new best
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_submission_dir = self.cfg.workspace_dir / "best_submission" 
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            best_submission_dir.mkdir(exist_ok=True, parents=True)
            if submission_path_final.exists(): 
                 shutil.copy2(submission_path_final, best_submission_dir / "submission.csv")
            with open(best_solution_dir / "solution.py", "w") as f: f.write(result_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f: f.write(str(result_node.id))

        # Rich log_step (as before)
        log_step(step=current_step_number, total=self.acfg.steps, stage=initial_node_stage,
                 is_buggy=result_node.is_buggy, exec_time=exec_duration,
                 metric=(result_node.metric.value if result_node.metric and result_node.metric.value is not None else None))
        
        t_step_end = time.time()
        logger.info(f"{log_prefix_main_step}_END: Duration: {t_step_end - t_step_start:.2f}s", extra={"verbose": True})



class PlannerAgent(Agent):

    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_logger: Optional[WandbLogger] = None,
        competition_benchmarks: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(task_desc, cfg, journal, wandb_logger, competition_benchmarks)

    def _query_llm_with_retries( self, query_type: str, system_prompt: Dict[str, Any], user_prompt: Dict[str, Any], model: str, temperature: float, planner_flag: bool, convert_system_to_user: bool, retries: int = 3,) -> Any:
        completion_text = None; log_prefix = f"PLANNER_AGENT_LLM_QUERY_{query_type.upper()}_STEP{self.current_step}"
        for attempt in range(retries):
            logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Sending request. Model: {model}, Temp: {temperature}, PlannerFlag: {planner_flag}", extra={"verbose": True})
            try:
                completion_text = query(system_message=system_prompt, user_message=user_prompt, model=model, temperature=temperature, planner=planner_flag, current_step=self.current_step, convert_system_to_user=convert_system_to_user, max_tokens=self.acfg.code.max_new_tokens)
                logger.info(f"{log_prefix}_ATTEMPT{attempt+1}: Received response.", extra={"verbose": True}); return completion_text
            except Exception as e:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Error during LLM query: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: logger.error(f"{log_prefix}: All {retries} retries failed.", extra={"verbose": True}); return None
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
        return None
    
    def plan_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        system_prompt = get_planner_agent_plan_system_prompt(); log_prefix = f"PLANNER_AGENT_PLAN_QUERY_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Sending PLANNER_PLAN query to LLM.", extra={"verbose": True})
        logger.debug(f"{log_prefix}: System prompt: {system_prompt}", extra={"verbose": True})
        logger.debug(f"{log_prefix}: User prompt: {user_prompt_dict}", extra={"verbose": True})
        completion_text = self._query_llm_with_retries(query_type="PLANNER_PLAN", system_prompt=system_prompt, user_prompt=user_prompt_dict, model=self.acfg.code.planner_model, temperature=self.acfg.code.temp, planner_flag=True, convert_system_to_user=self.acfg.convert_system_to_user, retries=retries)
        if completion_text is None: return "", "", ""
        task_summary, plan = extract_summary_and_plan(completion_text,task=True); 
        if not (plan and task_summary): 
            plan = plan or str(completion_text) 
            task_summary = task_summary or "SUMMARY_EXTRACTION_FAILED_FROM_PLAN_QUERY" 
            logger.warning(f"{log_prefix}: Plan or summary extraction failed/partial. Raw: {trim_long_string(completion_text)}", extra={"verbose":True})
        logger.debug(f"{log_prefix}: Plan query completed. Task summary: {task_summary}\n\nPlan: {plan}", extra={"verbose": True})
        return task_summary, plan, ""

    def code_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        system_prompt = get_planner_agent_code_system_prompt(); log_prefix = f"PLANNER_AGENT_CODE_QUERY_STEP{self.current_step}"
        logger.debug(f"{log_prefix}: Sending PLANNER_CODE query to LLM.", extra={"verbose": True})
        logger.debug(f"{log_prefix}: System prompt: {system_prompt}", extra={"verbose": True})
        logger.debug(f"{log_prefix}: User prompt: {user_prompt_dict}", extra={"verbose": True})
        completion_text = self._query_llm_with_retries(query_type="PLANNER_CODE", system_prompt=system_prompt, user_prompt=user_prompt_dict, model=self.acfg.code.model, temperature=self.acfg.code.temp, planner_flag=False, convert_system_to_user=self.acfg.convert_system_to_user, retries=retries)
        if completion_text is None: return "", "", "" 
        code = extract_code(completion_text)
        if not code: code = str(completion_text) 
        logger.debug(f"{log_prefix}\n\nCode query completed. Code: {code}", extra={"verbose": True})
        return "", code, "" 

    def _draft(self, parent_node=None) -> Node:
        log_prefix = f"PLANNER_AGENT_DRAFT_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting drafting. Parent: {parent_node.id if parent_node else 'None'}", extra={"verbose": True})
        plan_user_prompt = get_planner_agent_draft_plan_user_prompt(task_desc=self.task_desc, journal_summary=self.journal.generate_summary(include_code=False), competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        task_summary, agent_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not agent_plan: agent_plan = "PLAN_FAILED_IN_DRAFT"
        if not task_summary: task_summary = "TASK_SUMMARY_FAILED_IN_DRAFT_PLAN_QUERY"
        code_user_prompt = get_planner_agent_draft_code_user_prompt(task_summary_from_planner=task_summary, plan_from_planner=agent_plan, journal_summary=self.journal.generate_summary(include_code=False), competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not generated_code: generated_code = "#CODE_FAILED_IN_DRAFT"
        new_node = Node(plan=agent_plan, code=generated_code, summary=task_summary, task_summary=task_summary, parent=parent_node)
        logger.debug(f"{log_prefix}: Drafted new node {new_node.id}.", extra={"verbose": True})
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        log_prefix = f"PLANNER_AGENT_IMPROVE_STEP{self.current_step}"
        logger.debug(f"{log_prefix}: Starting improvement for node {parent_node.id}.", extra={"verbose": True})
        plan_user_prompt = get_planner_agent_improve_plan_user_prompt(task_desc=self.task_desc, parent_node_code=parent_node.code, competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        task_summary, improvement_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not improvement_plan: improvement_plan = "IMPROVE_PLAN_FAILED"
        if not task_summary: task_summary = "TASK_SUMMARY_FAILED_IN_IMPROVE_PLAN_QUERY"
        code_user_prompt = get_planner_agent_improve_code_user_prompt(task_summary_from_planner=task_summary, improvement_plan_from_planner=improvement_plan, parent_node_code=parent_node.code, journal_summary=self.journal.generate_summary(include_code=False), competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not generated_code: generated_code = parent_node.code 
        new_node = Node(plan=improvement_plan, code=generated_code, summary=task_summary, task_summary=task_summary, parent=parent_node)
        logger.debug(f"{log_prefix}: Improved node {parent_node.id} to new node {new_node.id}.", extra={"verbose": True})
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        log_prefix = f"PLANNER_AGENT_DEBUG_STEP{self.current_step}"
        logger.debug(f"{log_prefix}: Starting debugging for node {parent_node.id}.", extra={"verbose": True})
        plan_user_prompt = get_planner_agent_debug_plan_user_prompt(task_desc=self.task_desc, parent_node_code=parent_node.code, parent_node_term_out=parent_node.term_out, acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        bug_summary, fix_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not fix_plan: fix_plan = "DEBUG_PLAN_FAILED"
        if not bug_summary: bug_summary = "BUG_SUMMARY_FAILED_IN_DEBUG_PLAN_QUERY"
        code_user_prompt = get_planner_agent_debug_code_user_prompt(bug_summary_from_planner=bug_summary, fix_plan_from_planner=fix_plan, parent_node_code=parent_node.code, parent_node_feedback=parent_node.analysis, parent_node_term_out=parent_node.term_out, competition_name=self.competition_name, acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))
        if not generated_code: generated_code = parent_node.code 
        new_node = Node(plan=fix_plan, code=generated_code, summary=bug_summary, task_summary=bug_summary, parent=parent_node)
        logger.debug(f"{log_prefix}: Debugged node {parent_node.id} to new node {new_node.id}.", extra={"verbose": True})
        return new_node
    
    def reflect(self, node: Node) -> tuple[str, str]:
        log_prefix = f"PLANNER_AGENT_REFLECT_STEP{self.current_step}_NODE{node.id}"
        logger.debug(f"{log_prefix}: Initiating self-reflection.", extra={"verbose": True})
        try:
            reflection_plan, revised_code = perform_two_step_reflection(
                code=node.code, analysis=node.analysis, term_out=node.term_out,
                task_desc=self.task_desc, model_name=self.cfg.agent.code.planner_model, 
                temperature=self.acfg.code.temp, convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query, wrap_code_func=prompt_utils_wrap_code, extract_code_func=extract_code,
                current_step=self.current_step )
        except Exception as e:
            logger.error(f"{log_prefix}: Error during self-reflection call: {e}", exc_info=True, extra={"verbose": True})
            return f"REFLECTION_ERROR: {e}", node.code
        if revised_code and revised_code.strip() and revised_code != node.code: logger.info(f"{log_prefix}: Self-reflection resulted in code changes.", extra={"verbose": True})
        elif "No specific errors found requiring changes." in reflection_plan: logger.info(f"{log_prefix}: Self-reflection found no errors requiring changes.", extra={"verbose": True})
        else: logger.warning(f"{log_prefix}: Self-reflection finished, but revised code is same as original or empty. Plan: {trim_long_string(reflection_plan)}", extra={"verbose": True})
        return reflection_plan, revised_code
#############################################################################
# CodeChainAgent Implementation
#############################################################################
class CodeChainAgent(Agent): # Inherit from Agent
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


    # Override _query_llm_with_retries as it's specific to CodeChainAgent's two-model approach
    def _query_llm_with_retries(
        self,
        query_type: str,
        system_prompt: Dict[str, Any],
        user_prompt: Dict[str, Any],
        model: str,
        temperature: float,
        convert_system_to_user: bool,
        planner_flag: bool=False,
        retries: int = 3,
        max_tokens: int = None,
    ) -> Any:
        completion_text = None
        log_prefix = f""
        for attempt in range(retries):
            logger.info(f"Generation Attempt {attempt+1}/{retries}: Sending request. Model: {model}, Temp: {temperature}, PlannerFlag: {planner_flag}", extra={"verbose": True})
            try:
                completion_text = query(
                    system_message=system_prompt, user_message=user_prompt,
                    model=model, temperature=temperature, planner=planner_flag,
                    current_step=self.current_step, convert_system_to_user=convert_system_to_user,
                    max_tokens=self.acfg.code.max_new_tokens,
                )
                logger.info(f"{log_prefix} Attempt {attempt+1}: Received response.", extra={"verbose": True})
                if query_type == "Segment-Generation":
                    code_snippet = extract_code(completion_text)
                    if not code_snippet or not code_snippet.strip():
                        logger.warning(f"{log_prefix} Attempt {attempt+1}: Retrying ...")
                        continue
                    else:
                        logger.info(f"{log_prefix} Attempt {attempt+1}: Successfully extracted code.", extra={"verbose": True})
                        logger.debug(f"{log_prefix} \n EXTRACTED_CODE_START\n ----------- \n {code_snippet}\n ----------- \n EXTRACTED_CODE_END", extra={"verbose": True})
                        return code_snippet.strip()

                if completion_text.startswith("Exceeded context length limit"):
                    if retries == 0:
                        try:
                            user_prompt.pop("Memory", None)
                        except Exception as e:
                            logger.error(f"{log_prefix} Attempt {attempt+1}: Error dropping memory: {e}", exc_info=True, extra={"verbose": True})
                    if retries == 1:
                        try:
                            user_prompt.pop("Memory", None)
                            user_prompt.pop("Environment and Packages", None)
                            user_prompt.pop("Data Overview", None)

                        except Exception as e:
                            logger.error(f"{log_prefix} Attempt {attempt+1}: Error dropping environment and packages: {e}", exc_info=True, extra={"verbose": True})
                    if retries == 2:
                        try:
                            user_prompt.pop("Memory", None)
                            user_prompt.pop("Instructions", None)
                        except Exception as e:
                            logger.error(f"{log_prefix} Attempt {attempt+1}: Error dropping data overview: {e}", exc_info=True, extra={"verbose": True})
                    retries += 1
                    continue
                return completion_text
            except ContextLengthExceededError as cle: # Catch specific error
                logger.error(f"{log_prefix} Attempt {attempt+1}: Context Length Exceeded: {cle}. Aborting retries for this call.", exc_info=False, extra={"verbose": True}) # exc_info=False as CLE is already logged well
                return None #
            except Exception as e:
                logger.error(f"{log_prefix} Attempt {attempt+1}: Error during LLM query: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: 
                    logger.error(f"{log_prefix}: All {retries} retries failed.", extra={"verbose": True})
                    return None 
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5)) # Make delay configurable
        return ""


    def plan_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        system_prompt = get_planner_agent_plan_system_prompt()
        log_prefix = f"Plan_Step: {self.current_step}"
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
                                retries: int = 3
                              ) -> str: # Returns only the code snippet string

            completion_text = self._query_llm_with_retries(
                query_type="Segment-Generation",
                system_prompt=system_prompt_dict, 
                user_prompt=user_prompt_dict,
                model=self.acfg.code.model, # Coder model
                temperature=self.acfg.code.temp,
                planner_flag=False,
                convert_system_to_user=self.acfg.convert_system_to_user, 
                retries=retries
            )

            if completion_text is None:
                logger.error(f"LLM query returned None.")
                return "#LLM_QUERY_RETURNED_NONE_FOR_SEGMENT"
  
            code_snippet = completion_text
            
            return code_snippet.strip() if code_snippet else ""

    def _generate_code_segment(self,
                               segment_name: str,
                               task_summary: str,
                               master_plan_text: str,
                               code_accumulator: str,
                               chain_reflection: bool=False,
                               ) -> str:
        """Generates code for a single segment using the chained Coder."""
        log_prefix_segment = f"Code Chain step {self.current_step }"
        logger.info(f"{log_prefix_segment}: Generating code. Segment: {segment_name}", extra={"verbose": True})


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
        
        code_snippet = self._code_segment_query( # Call the new specialized method
            user_prompt_dict=segment_user_prompt,
            system_prompt_dict=segment_system_prompt,
            retries=self.acfg.get('coder_segment_retries', 3) 
        )
        if not code_snippet or code_snippet.strip() == "#CODE_FAILED" or not code_snippet.strip():
            logger.error(f"{log_prefix_segment}: Code generation failed or produced empty code.")
            return f"# FAILED TO GENERATE CODE FOR SEGMENT: {segment_name}\n"
        
        logger.info(f"{log_prefix_segment}: Successfully generated code snippet for {segment_name.replace(' ', '_')}.", extra={"verbose": True})
        logger.debug(f"{segment_name.replace(' ', '_')} Snippet: \n{code_snippet.strip()}\n ")

        if chain_reflection:
            # Reflecting on the code snippet
            logger.info(f"{log_prefix_segment}: Initial snippet generated. Now reflecting.")

        # Perform self-reflection on the generated snippet
            reflection_summary, code_snippet = self._reflect_on_segment(
                task_summary=task_summary,
                master_plan_text=master_plan_text,
                segment_name=segment_name,
                code_before_segment=code_accumulator,
                initial_segment_snippet=code_snippet
            )
            


            logger.info(f"{log_prefix_segment}_Revised Snippet: {trim_long_string(code_snippet)}")
        return code_snippet.strip() 
    
    def _draft_generate_code_chained(self, task_summary: str, master_plan_text: str) -> str:
        log_prefix_chain = f"CodeChainAgent_Chained_Draft_Step: {self.current_step}"
        logger.info(f"Starting chained code generation for draft.")
        
        # Initial boilerplate for the script
        code_accumulator = f"# Script generated by AIDE CodeChainAgent (Chained Coder) - Step {self.current_step}\n"
        code_accumulator += f"# Competition: {self.competition_name}\n"
        code_accumulator += f"# Task Summary: {task_summary.splitlines()[0]}...\n" # First line of summary
        code_accumulator += "# --- Master Plan ---\n"
        for i, plan_step_line in enumerate(master_plan_text.splitlines()):
             if plan_step_line.strip() and not plan_step_line.strip().startswith("##"): # Add non-empty lines, skip markdown headers
                code_accumulator += f"# {plan_step_line.strip()}\n"
        code_accumulator += "# --- End Master Plan ---\n\n"


        segments_order = [
            "Setup & Imports",
            "Data Loading",
            "Data Preprocessing",
            "Modeling",
            "Training & Validation", 
            "Prediction & Submission"
        ]

        chunked_reflection = (self.acfg.ITS_Strategy == "codechain_v3")
        chunk_size = 2
        if chunked_reflection:
            return self._generate_chuncked_code(task_summary, master_plan_text, chunk_size, code_accumulator)
        else:
            chain_reflection = True if self.acfg.ITS_Strategy == "codechain_v2" else False 
            for segment_name in segments_order:
                code_snippet = self._generate_code_segment(
                    segment_name, task_summary, master_plan_text, code_accumulator, chain_reflection
                )
                code_accumulator += code_snippet + "\n\n" # Add two newlines for separation
                if f"# FAILED TO GENERATE CODE FOR SEGMENT: {segment_name}" in code_snippet:
                    logger.warning(f"{log_prefix_chain}: Halting chain due to failure in segment: {segment_name}")
                    break # Optional: decide if you want to continue or halt on segment failure

            logger.info(f"{log_prefix_chain}: Chained code generation process complete.")
            return code_accumulator.strip()



    def _generate_chuncked_code(self, task_summary: str, master_plan_text: str, chunk_size: int = 2, code_accumulator: str = "") -> str:
        log_prefix_chain = f"CodeChainAgent_Chained_Draft_Step: {self.current_step}"

        segments_order = [
            "Setup & Imports",
            "Data Loading",
            "Data Preprocessing",
            "Modeling",
            "Training & Validation", 
            "Prediction & Submission"
        ]
        i = 0
        while i < len(segments_order):
            chunk = segments_order[i : i + chunk_size]
            code_before = code_accumulator
            combined_chunk = ""

            # 1) generate each segment in this chunk
            for seg in chunk:
                snippet = self._generate_code_segment(
                    seg, task_summary, master_plan_text, code_accumulator
                )
                combined_chunk += snippet + "\n\n"
                code_accumulator += snippet + "\n\n"
                if f"# FAILED TO GENERATE CODE FOR SEGMENT: {seg}" in snippet:
                    logger.warning(f"{log_prefix_chain}: failure in {seg}, skipping reflection for this chunk.")
                    break

            # 2) if configured, reflect on the whole chunk of `chunk_size` segments
            if len(combined_chunk.strip()) > 0:
                # for now we just duplicate the same task_summary per segment

                _, revised_chunk = self._reflect_on_chunk(
                    task_summary,
                    master_plan_text,
                    chunk,
                    code_before,
                    combined_chunk
                )
                # splice out the old chunk and replace with revised
                code_accumulator = code_before + revised_chunk + "\n\n"

            i += chunk_size

        logger.info(f"{log_prefix_chain}: Chained code generation complete.")
        return code_accumulator.strip()

    def _reflect_on_segment(self,
                            task_summary: str,
                            master_plan_text: str,
                            segment_name: str,
                            code_before_segment: str,
                            initial_segment_snippet: str
                           ) -> tuple[str, str]: # Returns (reflection_summary, revised_snippet)
        log_prefix_reflect = f"CodeChainAgent_Reflect_Step: {self.current_step}_Segment_{segment_name.replace(' ', '_')}"
        logger.info(f"Reflecting on initial snippet for segment '{segment_name}'.")

        system_prompt = get_segment_reflection_system_prompt()
        user_prompt = get_segment_reflection_user_prompt(
            task_summary=task_summary,
            master_plan_text=master_plan_text,
            current_segment_name=segment_name,
            code_generated_before_this_segment=code_before_segment,
            initial_code_snippet_for_this_segment=initial_segment_snippet
        )

        # Use a feedback/reflection model, could be o3-mini or same as coder
        reflection_llm = self.acfg.code.model # Or another config for reflection model
        
        reflection_completion_text = self._query_llm_with_retries(
            query_type=f"CodeChainAgent_Reflect_Step: {self.current_step}_Segment_{segment_name.replace(' ', '_')}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=reflection_llm,
            temperature=self.acfg.code.temp, # Or a specific reflection temp

            convert_system_to_user=self.acfg.convert_system_to_user,
            retries=self.acfg.get('reflection_retries', 1),
            max_tokens=self.acfg.code.max_new_tokens
        )

        if reflection_completion_text is None:
            logger.warning(f"{log_prefix_reflect}: Reflection LLM query returned None. Using initial snippet.")
            return "Reflection failed: No LLM response.", initial_segment_snippet

        # You'll need a robust way to parse this output
        reflection_summary, revised_snippet = extract_reflection_summary_and_revised_code(reflection_completion_text)

        if not revised_snippet or not revised_snippet.strip():
            logger.warning(f"{log_prefix_reflect}: Reflection did not produce a revised code snippet, or it was empty. Using initial snippet. Summary: {reflection_summary}")
            return reflection_summary or "Reflection did not produce code.", initial_segment_snippet
        
        if revised_snippet.strip() == initial_segment_snippet.strip():
            logger.debug(f"{log_prefix_reflect}: Reflection confirmed initial snippet is good. Summary: {reflection_summary}")
        else:
            logger.debug(f"{log_prefix_reflect}: Reflection produced a revised snippet. Summary: {reflection_summary}")
            # logger.debug(f"{log_prefix_reflect}_REVISED_SNIPPET_START\n{revised_snippet}\n{log_prefix_reflect}_REVISED_SNIPPET_END")

        return reflection_summary, revised_snippet

    def _reflect_on_chunk(
            self,
            task_summary: str,
            master_plan_text: str,
            segment_names: List[str],
            code_before_chunk: str,
            chunk_code: str
        ) -> tuple[str, str]:
            """
            Reflect on a whole chunk of segments at once.
            Returns (reflection_summary, revised_chunk_code)
            """
            tag = "_".join(s.replace(" ", "_") for s in segment_names)
            log_prefix = f"CodeChainAgent_ChunkReflect_Step:{self.current_step}_Segments_{tag}"
            logger.info(f"{log_prefix}: Reflecting on chunk of segments {segment_names}")

            system_prompt = get_chunked_reflection_system_prompt()   # your placeholder
            user_prompt = get_chunked_reflection_user_prompt(
                task_summary=task_summary,
                master_plan=master_plan_text,
                segment_names=segment_names,
                code_before_chunk=code_before_chunk,
                initial_chunk_code=chunk_code

            )

            completion = self._query_llm_with_retries(
                query_type=f"Chunk-Reflection_{tag}",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                retries=self.acfg.get('reflection_retries', 1),
                max_tokens=self.acfg.code.max_new_tokens,
            )
            if not completion:
                logger.warning(f"{log_prefix}: No response; returning original chunk.")
                return "", chunk_code

            summary, revised = extract_reflection_summary_and_revised_code(completion)
            if not revised.strip() or revised.strip() == "# FAILED TO FIND 'Revised Code Snippet:' SECTION":
                logger.warning(f"{log_prefix}: Empty revised chunk; using original.")
                return summary, chunk_code

            logger.info(f"{log_prefix}: Chunk reflection produced revised code.")
            logger.debug(f"-----------------------------------------------------------------")
            logger.debug(f"{log_prefix}: Summary: {summary}", extra={"verbose": True})
            logger.debug(f"-----------------------------------------------------------------")
            logger.debug(f"{log_prefix}: Revised chunk: {revised}", extra={"verbose": True})
            logger.debug(f"-----------------------------------------------------------------")

            return summary, revised


    # Modify the existing _draft method to use this chained approach
    def _draft(self, parent_node=None) -> Node:
        log_prefix = f""
        logger.info(f"{log_prefix} Starting drafting process. Parent: {parent_node.id if parent_node else 'None'}")
        memory=self.journal.generate_summary(include_code=False) # Memory

        # 1. Generate Master Plan using the Planner model
        logger.info(f"{log_prefix} Calling Planner for Task Summary and Master Plan.")
        plan_user_prompt = get_planner_agent_draft_plan_user_prompt(
            task_desc=self.task_desc, 
            journal_summary=memory,
            competition_name=self.competition_name, 
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        logger.debug(f"Memory used for step {self.current_step}\n: {memory}", extra={"verbose": True})
        # self.plan_query uses self.acfg.code.planner_model
        task_summary, master_plan_text, _ = self.plan_query(
            plan_user_prompt, 
            retries=self.acfg.get('planner_retries', 3)
        )
        
        if not master_plan_text or master_plan_text.strip() == "": 
            logger.error(f"{log_prefix} Master plan generation failed by Planner. Aborting draft.")
            final_plan_text = "MASTER_PLAN_FAILED_BY_PLANNER"
            generated_code = "# MASTER_PLAN_FAILED_BY_PLANNER - No code generated."
            final_summary = task_summary or "PLANNER_FAILED_TO_PRODUCE_SUMMARY_AND_PLAN"
        else:
            logger.info(f"{log_prefix} Master Plan received from Planner. Proceeding to chained code generation.")

            final_plan_text = master_plan_text # Store the full plan text for the node
            
            # 2. Generate Code via Chaining using the Coder model
            generated_code = self._draft_generate_code_chained(task_summary, master_plan_text)
            final_summary = task_summary # Use the summary from the planner

            if not generated_code or generated_code.strip().startswith("# FAILED TO GENERATE CODE FOR SEGMENT:") or generated_code.strip() == "# SEGMENT 1 (Data Loading & Initial Setup) FAILED TO GENERATE":
                 logger.error(f"{log_prefix} Chained code generation resulted in failure or predominantly error messages.")
                 # Keep generated_code as is, it will contain error placeholders

        new_node = Node(plan=final_plan_text, code=generated_code, summary=final_summary, task_summary=final_summary, parent=parent_node)
        logger.info(f"{log_prefix} Drafted new node {new_node.id} using ChainedCoder.")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        log_prefix = f"CodeChainAgent_Improve_Step: {self.current_step}"
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
        log_prefix = f"CodeChainAgent_Debug_Step: {self.current_step}"
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

#############################################################################
# SelfConsistencyAgent Implementation
#############################################################################
class SelfConsistencyAgent(Agent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_logger: Optional[WandbLogger] = None,
        competition_benchmarks: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(task_desc, cfg, journal, wandb_logger, competition_benchmarks)
        logger.info(
            f"SelfConsistencyAgent initialized. N={self.acfg.self_consistency.num_responses}, "
            f"Strategy='{self.acfg.self_consistency.selection_strategy}'"
        )

    def plan_and_code_query(self,
                            user_prompt_dict: Dict[str, Any],
                            system_prompt_dict: Optional[Dict[str, Any]] = None,
                            retries: int = 3,
                            return_all_responses: bool = False, 
                            num_responses: int = 1,
                           ) -> Union[Tuple[str, str, str], List[Tuple[str, str, str]]]:
        if system_prompt_dict is None:
            system_prompt_dict = get_agent_system_prompt()
        
        log_prefix = f"AGENT_PNC_QUERY_Step:{self.current_step}"
        
        n_to_request_from_backend = num_responses
        
        default_single_logical_error: Tuple[str,str,str] = ("", "LLM Query Error: Unknown failure", "LLM_QUERY_FAILED_AGENT")

        for attempt in range(retries):
            logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Calling backend.query (requesting n={n_to_request_from_backend}).", extra={"verbose": True})
            
            raw_llm_outputs_list= query(
                system_message=system_prompt_dict,
                user_message=user_prompt_dict,
                model=self.acfg.code.model,
                temperature=0.95,
                max_tokens=self.acfg.code.max_new_tokens,
                current_step=self.current_step,
                inference_engine=self.cfg.inference_engine,
                num_responses=n_to_request_from_backend,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )
            if isinstance(raw_llm_outputs_list, str) and \
                ("Exceeded context length limit" in raw_llm_outputs_list or \
                "CONTEXT_LENGTH_EXCEEDED" in raw_llm_outputs_list): # Check common error strings
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Backend returned Context Length Exceeded string: {raw_llm_outputs_list}")

                raise ContextLengthExceededError(f"CLE from backend: {raw_llm_outputs_list}")

            if isinstance(raw_llm_outputs_list, list):

                for item_idx, item_content in enumerate(raw_llm_outputs_list):
                    if isinstance(item_content, str) and \
                        ("Exceeded context length limit" in item_content or \
                        "CONTEXT_LENGTH_EXCEEDED" in item_content):
                        logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Item {item_idx} in list from backend signals Context Length Exceeded: {item_content}")
                        raise ContextLengthExceededError(f"CLE in list item from backend: {item_content}")
            
            if num_responses == 1 : 
                if not isinstance(raw_llm_outputs_list, (str)) :
                    logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Expected single str/dict from backend (n=1), got {type(raw_llm_outputs_list)}. Content: {str(raw_llm_outputs_list)[:200]}")
                    if attempt == retries -1 : return None # Total failure
                    time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
                    continue # Retry 
 
            # 2. Validate the list structure from backend.query
            if not isinstance(raw_llm_outputs_list, list) or len(raw_llm_outputs_list) != n_to_request_from_backend:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Backend.query returned malformed response. Expected list of size {n_to_request_from_backend}, got {type(raw_llm_outputs_list)} of size {len(raw_llm_outputs_list) if isinstance(raw_llm_outputs_list, list) else 'N/A'}.")
                error_tuple_malformed: Tuple[str,str,str] = ("", "LLM Query Error: Malformed backend response structure", "LLM_MALFORMED_BACKEND_STRUCTURE")
                if attempt == retries - 1:
                    return [error_tuple_malformed] * n_to_request_from_backend if return_all_responses else error_tuple_malformed
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
                continue
            
            # 3. Process each item in the raw_llm_outputs_list
            processed_candidates: List[Tuple[str, str, str]] = []
            any_item_had_unrecoverable_error = False

            for idx, raw_text_item in enumerate(raw_llm_outputs_list):
                if not isinstance(raw_text_item, str):
                    logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}: Item {idx} is not a string ({type(raw_text_item)}). Marking as extraction error.")
                    processed_candidates.append(("", "Non-string item in response", "EXTRACTION_FAILED_TYPE"))
                    continue # Still add placeholder, but this item is bad

                if "ERROR: context length exceeded" in raw_text_item or "Exceeded context length limit" in raw_text_item:
                    logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Item {idx} content indicates Context Length Exceeded. Failing entire operation.")
                    any_item_had_unrecoverable_error = True
                    default_single_logical_error = ("", "LLM Query Error: Context Length Exceeded in item content", "CONTEXT_LENGTH_EXCEEDED_CONTENT")
                    break 

                if raw_text_item.startswith("ERROR:"): # Generic error message content from provider
                    logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}: Item {idx} is an error message: '{trim_long_string(raw_text_item)}'.", extra={"verbose": True})
                    processed_candidates.append(("", f"# {raw_text_item}", raw_text_item))
                    continue
                code = extract_code(raw_text_item)
                nl_text = extract_text_up_to_code(raw_text_item)

                if code and nl_text:
                    processed_candidates.append((nl_text, code, "plan_code_summary_placeholder")) # Changed summary
                    print(f"Candidate {idx+1}: for plan and code extraction is extracted successfully: ✅")

                else:
                    logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}: Plan or code extraction failed for item {idx}. Raw: '{trim_long_string(raw_text_item)}'", extra={"verbose": True})
                    processed_candidates.append((nl_text or "EXTRACTION_FAILED_PLAN", 
                                                 code or "# EXTRACTION_FAILED_CODE", 
                                                 "Extraction of plan/code failed for this candidate"))
            
            if any_item_had_unrecoverable_error:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Unrecoverable error (like CLE in item) encountered. Failing this attempt.")
                if return_all_responses:
                    return [default_single_logical_error] * n_to_request_from_backend
                else:
                    return default_single_logical_error

            # 4. Check if any valid extractions occurred
            has_at_least_one_good_extraction = any(
                not (cand_plan.startswith("EXTRACTION_FAILED") or cand_code.startswith("# EXTRACTION_FAILED")) 
                for cand_plan, cand_code, _ in processed_candidates
            )

            if not processed_candidates or not has_at_least_one_good_extraction:
                logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}: No valid plan/code could be extracted from any of the {len(raw_llm_outputs_list)} responses. Retrying if attempts left.")
                if attempt == retries - 1:
                    logger.error(f"{log_prefix}: All retries failed to yield any valid plan/code extraction.")
                    return [default_single_logical_error] * n_to_request_from_backend if return_all_responses else default_single_logical_error
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
                continue # Go to next attempt

            logger.info(f"{log_prefix}_ATTEMPT{attempt+1}: Successfully processed LLM outputs into {len(processed_candidates)} candidate tuples.", extra={"verbose":True})
            if return_all_responses:
                return processed_candidates # Return the list of (plan,code,summary) tuples
            else:
                for p_nl, p_code, p_sum in processed_candidates:
                    if not (p_nl.startswith("EXTRACTION_FAILED") or p_code.startswith("# EXTRACTION_FAILED")):
                        return p_nl, p_code, p_sum
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Logic error - has_at_least_one_good_extraction was true, but no good extraction found in loop.")
                return processed_candidates[0]

        logger.error(f"{log_prefix}: All {retries} query attempts failed.", extra={"verbose": True})
        return [default_single_logical_error] * n_to_request_from_backend if return_all_responses else default_single_logical_error

    def _get_master_plan(self) -> Tuple[str, str]: # Returns (task_summary, master_plan_text)
        """
        Generates a single, detailed master plan for the current task.
        This uses the same planning mechanism as PlannerAgent or CodeChainAgent's initial planning.
        """
        log_prefix = f"SC_AGENT_GET_MASTER_PLAN_Step:{self.current_step}"
        logger.info(f"{log_prefix}: Generating master plan.", extra={"verbose": True})


        plan_user_prompt_dict = get_planner_agent_draft_plan_user_prompt(
            task_desc=self.task_desc,
            journal_summary=self.journal.generate_summary(include_code=False), 
            competition_name=self.competition_name,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
  
        task_summary, master_plan_text, _ = self.plan_query(
            user_prompt_dict=plan_user_prompt_dict,
            retries=3, 
        )
  
        if not master_plan_text or master_plan_text.strip() == "":
            logger.error(f"{log_prefix}: Master plan generation failed or returned empty. Defaulting to error plan.")
            master_plan_text = "MASTER_PLAN_GENERATION_FAILED"
        if not task_summary or task_summary.strip() == "":
            task_summary = "TASK_SUMMARY_GENERATION_FAILED_DURING_PLANNING"
            
        logger.info(f"{log_prefix}: Master plan generated successfully.", extra={"verbose":True})
        logger.debug(f"{log_prefix}_TASK_SUMMARY_START\n{task_summary}\n{log_prefix}_TASK_SUMMARY_END", extra={"verbose":True})
        logger.debug(f"{log_prefix}_MASTER_PLAN_START\n{master_plan_text}\n{log_prefix}_MASTER_PLAN_END", extra={"verbose":True})
        
        return task_summary, master_plan_text

    def _get_N_code_candidates_for_plan(self, 
                                        task_summary: str, 
                                        master_plan_text: str
                                       ) -> List[Tuple[str, str, str]]: # Returns list of (plan_placeholder, code_candidate, summary_placeholder)
        """
        Generates N code candidates for the given master_plan_text and task_summary.
        """
        log_prefix = f"SC_AGENT_GET_N_CODES_Step:{self.current_step}"
        N = self.acfg.self_consistency.num_responses
        print(f"self.acfg.self_consistency: {self.acfg.self_consistency}")
        logger.info(f"{log_prefix}: Generating {N} code candidates for the master plan.", extra={"verbose": True})


        code_gen_user_prompt_dict = get_planner_agent_draft_code_user_prompt(
            task_summary_from_planner=task_summary,
            plan_from_planner=master_plan_text,
            journal_summary=self.journal.generate_summary(include_code=False), # Memory can still be useful
            competition_name=self.competition_name,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )

        code_candidate_tuples: List[Tuple[str, str, str]] = self.code_query(
            user_prompt_dict=code_gen_user_prompt_dict,
            retries=3,
            num_responses=N, # Ask for N responses
            temperature=0.99
        )
        if not code_candidate_tuples or not isinstance(code_candidate_tuples, list):
            logger.error(f"{log_prefix}: Failed to get code candidates or received unexpected type: {type(code_candidate_tuples)}. Returning empty list.")
            return [("", "#CODE_QUERY_FAILED_UNEXPECTED", "Code query failed")] * N # Pad with errors

        # Ensure we have N tuples, even if some are error placeholders from code_query
        while len(code_candidate_tuples) < N:
            logger.warning(f"{log_prefix}: code_query returned fewer than N items. Padding.")
            code_candidate_tuples.append(("", "#CODE_QUERY_MISSING_ITEM", "Code query missing item"))
            
        logger.info(f"{log_prefix}: Generated {len(code_candidate_tuples)} raw code candidate tuples.", extra={"verbose":True})
        for i, (_, code_cand, _) in enumerate(code_candidate_tuples):
             logger.debug(f"{log_prefix}_CANDIDATE_{i+1}_CODE_START\n{code_cand}\n{log_prefix}_CANDIDATE_{i+1}_CODE_END", extra={"verbose":True})
        
        return code_candidate_tuples

    def _evaluate_and_select_candidate(self, 
                                       task_summary: str, # From _get_master_plan
                                       master_plan_text: str, # From _get_master_plan
                                       code_candidate_tuples: List[Tuple[str, str, str]], # From _get_N_code_candidates
                                       parent_node_for_lineage: Optional[Node], # For _improve/_debug lineage
                                       operation_type: str # "draft", "improve", or "debug" for logging
                                      ) -> Node: # Returns the single, chosen, fully-evaluated Node
        """
        Evaluates N code candidates, selects the best one based on configured strategy,
        and returns that chosen candidate as a fully populated Node object.
        """
        log_prefix_eval_select = f"SC_AGENT_EVAL_SELECT_{operation_type.upper()}_Step:{self.current_step}"
        logger.info(f"{log_prefix_eval_select}: Evaluating {len(code_candidate_tuples)} code candidates using strategy '{self.acfg.self_consistency.selection_strategy}'.")

        if not code_candidate_tuples:
            logger.error(f"{log_prefix_eval_select}: No code candidates provided for evaluation.")
            # Create a definitive error node to return
            error_node = Node(
                plan=master_plan_text or "MASTER_PLAN_UNAVAILABLE",
                code="#NO_CODE_CANDIDATES_TO_EVALUATE",
                summary=task_summary or "TASK_SUMMARY_UNAVAILABLE",
                task_summary=task_summary or "TASK_SUMMARY_UNAVAILABLE"
            )
            if parent_node_for_lineage: error_node.parent = parent_node_for_lineage
            error_node.is_buggy = True
            error_node.metric = WorstMetricValue()
            error_node.analysis = "Self-consistency: No code candidates were generated or provided for evaluation."
            return error_node

        evaluated_temp_nodes: List[Node] = []

        for i, (_, code_str, original_summary_placeholder) in enumerate(code_candidate_tuples):
            candidate_log_prefix = f"{log_prefix_eval_select}_CANDIDATE_{i+1}"
            logger.info(f"{candidate_log_prefix}: Processing.", extra={"verbose": True})
            
            # Create a temporary Node for this candidate
            temp_node = Node(
                plan=master_plan_text, 
                code=code_str, 
                summary=original_summary_placeholder, # This will be updated by parse_exec_result
                task_summary=task_summary # Store the overall task summary
            )

            if parent_node_for_lineage:
                temp_node.parent = parent_node_for_lineage 

            # Check for placeholder error codes from _get_N_code_candidates_for_plan
            if code_str.startswith("#LLM_QUERY_RETURNED_NONE") or \
               code_str.startswith("#NON_STRING_RESPONSE_ITEM") or \
               code_str.startswith("#CODE_QUERY_FAILED_UNEXPECTED") or \
               code_str.startswith("#CODE_QUERY_MISSING_ITEM") or \
               "#CODE_EXTRACTION_FAILED" in code_str or \
               code_str.startswith("#ERROR:"):
                print(f"Candidate is an error placeholder. Marking as buggy without execution....")
                logger.debug(f"{candidate_log_prefix}: Candidate is an error placeholder. Marking as buggy without execution. Code: {code_str[:100]}...", extra={"verbose": True})
                temp_node.is_buggy = True
                temp_node.metric = WorstMetricValue()
                temp_node.analysis = f"Candidate generation/extraction failed: {code_str.splitlines()[0] if code_str else 'Unknown reason'}"
                temp_node.exec_time = 0.0
            elif not hasattr(self, 'exec_callback') or not callable(self.exec_callback):
                logger.error(f"{candidate_log_prefix}: exec_callback not available. Cannot execute candidate.")
                temp_node.is_buggy = True
                temp_node.metric = WorstMetricValue()
                temp_node.analysis = "Execution callback was missing, cannot evaluate candidate."
                temp_node.exec_time = 0.0
            else:

                logger.info(f"{candidate_log_prefix}: Executing code.", extra={"verbose":True})
                exec_result = self.exec_callback(temp_node.code, reset_session=True) # Always reset for each SC candidate
                

                temp_node = self.parse_exec_result(node=temp_node, exec_result=exec_result)
                logger.info(f"{candidate_log_prefix}: Execution & parsing complete. Buggy: {temp_node.is_buggy}, Metric: {temp_node.metric.value if temp_node.metric else 'N/A'}, Analysis: {trim_long_string(temp_node.analysis, 100)}", extra={"verbose":True})

            evaluated_temp_nodes.append(temp_node)


        chosen_evaluated_node: Optional[Node] = None
        selection_strategy = self.acfg.self_consistency.selection_strategy

        if selection_strategy == "interpreter_first_success":
            for node_candidate in evaluated_temp_nodes:
                if not node_candidate.is_buggy:
                    chosen_evaluated_node = node_candidate
                    logger.info(f"{log_prefix_eval_select}: Selected first non-buggy candidate (Temp Node ID for eval: {node_candidate.id}).", extra={"verbose":True})
                    break
        
        elif selection_strategy == "interpreter_best_metric":
            non_buggy_candidates = [n for n in evaluated_temp_nodes if not n.is_buggy]
            if non_buggy_candidates:
                non_buggy_candidates.sort(key=lambda n: n.metric, reverse=True) # sort so best is first ###### THIS NEED TO BE LOOKED AT AGAIN
                chosen_evaluated_node = non_buggy_candidates[0]
                logger.info(f"{log_prefix_eval_select}: Selected non-buggy candidate with best metric: {chosen_evaluated_node.metric.value if chosen_evaluated_node.metric else 'N/A'} (Temp Node ID for eval: {chosen_evaluated_node.id}).", extra={"verbose":True})

        # Fallback if no candidate chosen by strategy (e.g., all were buggy)
        if not chosen_evaluated_node:
            if evaluated_temp_nodes: # Should always be true if code_candidate_tuples was not empty
                logger.warning(f"{log_prefix_eval_select}: No candidate chosen by strategy '{selection_strategy}' (likely all buggy). Defaulting to the first evaluated candidate (Temp Node ID: {evaluated_temp_nodes[0].id}).", extra={"verbose":True})
                chosen_evaluated_node = evaluated_temp_nodes[0]
            else:
                # This case should have been caught by the initial check of code_candidate_tuples
                logger.error(f"{log_prefix_eval_select}: CRITICAL: No evaluated_temp_nodes to select from. This should not happen.")
                # Construct a definitive error node if chosen_evaluated_node is somehow still None
                chosen_evaluated_node = Node(
                    plan=master_plan_text or "MASTER_PLAN_UNAVAILABLE_FALLBACK",
                    code="#FALLBACK_ERROR_NO_CHOSEN_NODE",
                    summary=task_summary or "TASK_SUMMARY_UNAVAILABLE_FALLBACK",
                    task_summary=task_summary or "TASK_SUMMARY_UNAVAILABLE_FALLBACK"
                )
                if parent_node_for_lineage: chosen_evaluated_node.parent = parent_node_for_lineage
                chosen_evaluated_node.is_buggy = True
                chosen_evaluated_node.metric = WorstMetricValue()
                chosen_evaluated_node.analysis = "Self-consistency: Failed to select any candidate after evaluation."
        
        
        chosen_idx = -1
        try:
            chosen_idx = evaluated_temp_nodes.index(chosen_evaluated_node) + 1
        except ValueError:
            logger.error(f"{log_prefix_eval_select}: Chosen node not found in evaluated_temp_nodes list. This is unexpected.")

        logger.info(f"{log_prefix_eval_select}: Final chosen candidate is evaluated temp Node #{chosen_idx} (ID: {chosen_evaluated_node.id}). "
                    f"Buggy: {chosen_evaluated_node.is_buggy}, Metric: {chosen_evaluated_node.metric.value if chosen_evaluated_node.metric else 'N/A'}.",
                    extra={"verbose": True})
        
        return chosen_evaluated_node

    def _evaluate_and_select_plan_code_pairs(self,
                                             candidate_plan_code_summary_tuples: List[Tuple[str, str, str]],
                                             parent_node_for_lineage: Node,
                                             operation_type: str,
                                             selection_strategy: str = "interpreter_first_success"
                                            ) -> Node:
        """
        Evaluates N (plan, code, summary) candidate pairs, selects the best one,
        and returns that chosen candidate as a fully populated Node object.
        Used for _improve and _debug stages of SelfConsistencyAgent.
        """
        log_prefix_eval_select = f"SC_AGENT_EVAL_SELECT_PAIRS_{operation_type.upper()}_Step:{self.current_step}"
        logger.info(f"{log_prefix_eval_select}: Evaluating {len(candidate_plan_code_summary_tuples)} (plan,code) candidate pairs using strategy '{self.acfg.self_consistency.selection_strategy}'.")

        if not candidate_plan_code_summary_tuples:
            logger.error(f"{log_prefix_eval_select}: No (plan,code) candidate pairs provided for evaluation.")
            error_node = Node(
                plan=f"NO_CANDIDATE_PAIRS_FOR_{operation_type.upper()}",
                code="#NO_CODE_CANDIDATES_TO_EVALUATE",
                summary=f"Failed to generate candidates for {operation_type}",
                task_summary=parent_node_for_lineage.task_summary or "TASK_SUMMARY_UNAVAILABLE" # Get from parent
            )
            error_node.parent = parent_node_for_lineage
            error_node.is_buggy = True
            error_node.metric = WorstMetricValue()
            error_node.analysis = f"Self-consistency {operation_type} failed: No (plan,code) candidate pairs were generated."
            return error_node

        evaluated_temp_nodes: List[Node] = []

        for i, output_tuple in enumerate(candidate_plan_code_summary_tuples):
            plan_str, code_str, original_summary_placeholder = output_tuple 
            candidate_log_prefix = f"{log_prefix_eval_select}_CANDIDATE_PAIR_{i+1}"
            logger.info(f"{candidate_log_prefix}: Processing.", extra={"verbose": True})
            
            temp_node = Node(
                plan=plan_str,
                code=code_str, 
                summary=original_summary_placeholder,
                task_summary=parent_node_for_lineage.task_summary
            )
            temp_node.parent = parent_node_for_lineage # Set parent for lineage and context

            if plan_str.startswith("EXTRACTION_FAILED") or plan_str.startswith("LLM Query Error:") or \
               code_str.startswith("#EXTRACTION_FAILED") or code_str.startswith("#LLM Query Error:") or \
               code_str.startswith("#ERROR:") or code_str == "Exceeded context length limit":
                logger.warning(f"{candidate_log_prefix}: Candidate pair is an error placeholder. Marking as buggy. Plan: {plan_str[:100]}, Code: {code_str[:100]}...")
                temp_node.is_buggy = True
                temp_node.metric = WorstMetricValue()
                temp_node.analysis = f"Candidate pair generation/extraction failed: P: '{plan_str[:50]}' C: '{code_str[:50]}'"
                temp_node.exec_time = 0.0
            elif not hasattr(self, 'exec_callback') or not callable(self.exec_callback):
                logger.error(f"{candidate_log_prefix}: exec_callback not available. Cannot execute candidate.")
                temp_node.is_buggy = True; temp_node.metric = WorstMetricValue()
                temp_node.analysis = "Execution callback was missing."; temp_node.exec_time = 0.0
            else:
                logger.info(f"{candidate_log_prefix}: Executing code.", extra={"verbose":True})
                exec_result = self.exec_callback(temp_node.code, reset_session=True)
                temp_node = self.parse_exec_result(node=temp_node, exec_result=exec_result)
                logger.info(f"{candidate_log_prefix}: Eval complete. Buggy: {temp_node.is_buggy}, Metric: {temp_node.metric.value if temp_node.metric else 'N/A'}", extra={"verbose":True})

            evaluated_temp_nodes.append(temp_node)

        chosen_evaluated_node: Optional[Node] = None
        selection_strategy = self.acfg.self_consistency.selection_strategy

        if selection_strategy == "interpreter_first_success":
            for node_candidate in evaluated_temp_nodes:
                if not node_candidate.is_buggy:
                    chosen_evaluated_node = node_candidate
                    logger.info(f"{log_prefix_eval_select}: Selected first non-buggy candidate pair (Temp Node ID: {node_candidate.id}).", extra={"verbose":True})
                    break
        
        elif selection_strategy == "interpreter_best_metric":
            non_buggy_candidates = [n for n in evaluated_temp_nodes if not n.is_buggy]
            if non_buggy_candidates:
                non_buggy_candidates.sort(key=lambda n: n.metric, reverse=True)
                chosen_evaluated_node = non_buggy_candidates[0]
                logger.info(f"{log_prefix_eval_select}: Selected non-buggy pair (best metric): {chosen_evaluated_node.metric.value if chosen_evaluated_node.metric else 'N/A'} (Temp Node ID: {chosen_evaluated_node.id}).", extra={"verbose":True})

        if not chosen_evaluated_node:
            if evaluated_temp_nodes:
                logger.warning(f"{log_prefix_eval_select}: No pair chosen by strategy '{selection_strategy}'. Defaulting to first evaluated pair (Temp Node ID: {evaluated_temp_nodes[0].id}).", extra={"verbose":True})
                chosen_evaluated_node = evaluated_temp_nodes[0]
            else:
                logger.error(f"{log_prefix_eval_select}: CRITICAL: No evaluated_temp_nodes for pairs. Creating error node.")
                chosen_evaluated_node = Node(
                    plan=f"FALLBACK_ERROR_NO_CHOSEN_PAIR_{operation_type.upper()}",
                    code="#FALLBACK_ERROR_NO_CHOSEN_PAIR",
                    summary=f"Fallback error for {operation_type}",
                    task_summary=parent_node_for_lineage.task_summary
                )
                chosen_evaluated_node.parent = parent_node_for_lineage
                chosen_evaluated_node.is_buggy = True; chosen_evaluated_node.metric = WorstMetricValue()
                chosen_evaluated_node.analysis = "Self-consistency: Failed to select any (plan,code) pair after evaluation."
        
        logger.info(f"{log_prefix_eval_select}: Final chosen candidate pair is evaluated temp Node (ID: {chosen_evaluated_node.id}). "
                    f"Buggy: {chosen_evaluated_node.is_buggy}, Metric: {chosen_evaluated_node.metric.value if chosen_evaluated_node.metric else 'N/A'}.",
                    extra={"verbose": True})
        
        return chosen_evaluated_node

    def _draft(self, parent_node: Optional[Node] = None) -> Node: 
        """
        Generates N code candidates for a single master plan, evaluates them, 
        selects the best, and returns it as a new Node.
        """
        log_prefix_draft = f"SC_AGENT_DRAFT_Step:{self.current_step}"
        logger.info(f"{log_prefix_draft}: Starting self-consistency draft operation (N={self.acfg.self_consistency.num_responses}).")


        task_summary, master_plan_text = self._get_master_plan()

        if master_plan_text == "MASTER_PLAN_GENERATION_FAILED_SELF_CONSISTENCY":
            logger.error(f"{log_prefix_draft}: Master plan generation failed. Cannot proceed with drafting code candidates.")
            # Create and return a definitive error node
            error_node = Node(
                plan=master_plan_text, # Contains the failure message
                code="#MASTER_PLAN_FAILED_NO_CODE_GENERATED",
                summary=task_summary, # Contains its own failure message
                task_summary=task_summary
            )
            # parent_node is None for draft
            error_node.is_buggy = True
            error_node.metric = WorstMetricValue()
            error_node.analysis = "Self-consistency draft failed: Master plan could not be generated."
            return error_node

        print(f"finished getting master plan, now getting code candidates\n\n")
        # 2. Get N Code Candidates for this Master Plan
        code_candidate_tuples = self._get_N_code_candidates_for_plan(
            task_summary=task_summary,
            master_plan_text=master_plan_text
        )
        print(f"finished getting code candidates, now evaluating and selecting the best code candidate\n\n")
        print(f"number of code candidates: {len(code_candidate_tuples)}")
        # Check if candidate generation itself failed critically (e.g., all N attempts returned errors)
        if not code_candidate_tuples:
             logger.error(f"{log_prefix_draft}: Failed to generate any viable code candidates for the master plan.")
             error_node = Node(
                plan=master_plan_text,
                code="#NO_VIABLE_CODE_CANDIDATES_GENERATED",
                summary=task_summary,
                task_summary=task_summary
             )
             error_node.is_buggy = True
             error_node.metric = WorstMetricValue()
             error_node.analysis = "Self-consistency draft failed: No viable code candidates were generated for the master plan."
             return error_node

        # 3. Evaluate and Select the Best Code Candidate
        chosen_evaluated_temp_node = self._evaluate_and_select_candidate(
            task_summary=task_summary,
            master_plan_text=master_plan_text,
            code_candidate_tuples=code_candidate_tuples,
            parent_node_for_lineage=None, # No parent for a draft node being created
            operation_type="draft"
        )

        # 4. Create the final Node for the journal from the chosen_evaluated_temp_node.
        final_journal_node = Node(
            plan=master_plan_text, # The common plan for all candidates
            code=chosen_evaluated_temp_node.code,
            summary=chosen_evaluated_temp_node.summary, # This was initially a placeholder, then updated by parse_exec_result
            task_summary=chosen_evaluated_temp_node.task_summary, # Should be the initial task_summary
            
            # Copy execution and evaluation results from the chosen temporary node
            _term_out=chosen_evaluated_temp_node._term_out, # Note: _term_out is the list
            exec_time=chosen_evaluated_temp_node.exec_time,
            exc_type=chosen_evaluated_temp_node.exc_type,
            exc_info=chosen_evaluated_temp_node.exc_info,
            exc_stack=chosen_evaluated_temp_node.exc_stack,
            analysis=chosen_evaluated_temp_node.analysis,
            metric=chosen_evaluated_temp_node.metric,
            code_quality=chosen_evaluated_temp_node.code_quality,
            is_buggy=chosen_evaluated_temp_node.is_buggy
        )

        logger.info(f"{log_prefix_draft}: Drafted new node {final_journal_node.id} via self-consistency. "
                    f"Chosen from {len(code_candidate_tuples)} candidates. "
                    f"Buggy: {final_journal_node.is_buggy}, Metric: {final_journal_node.metric.value if final_journal_node.metric else 'N/A'}",
                    extra={"verbose":True})
        logger.debug(f"{log_prefix_draft}_FINAL_CHOSEN_CODE_START\n{final_journal_node.code}\n{log_prefix_draft}_FINAL_CHOSEN_CODE_END", extra={"verbose":True})

        return final_journal_node

    def _improve(self, parent_node: Node) -> Node:
        log_prefix_improve = f"SC_AGENT_IMPROVE_Step:{self.current_step}"
        logger.info(f"{log_prefix_improve}: Starting self-consistency improve for node {parent_node.id} (N={self.acfg.self_consistency.num_responses}).", extra={"verbose": True})

        # 1. Prepare the prompt for generating N improvement (plan,code) pairs
        improve_sys_prompt = get_agent_improve_system_prompt() # This is AGENT_improve_SYSTEM_PROMPT_DICT
        
        improve_user_prompt_dict = get_agent_improve_user_prompt(
            task_desc=self.task_desc,
            journal_summary=self.journal.generate_summary(include_code=False), # Memory
            competition_name=self.competition_name,
            parent_node_code=parent_node.code 
        )

        # 2. Call Agent.plan_and_code_query to get N (plan,code,summary) tuples
        candidate_plan_code_summary_tuples: List[Tuple[str, str, str]] = self.plan_and_code_query(
            user_prompt_dict=improve_user_prompt_dict,
            system_prompt_dict=improve_sys_prompt,
            retries=3,
            num_responses=self.acfg.self_consistency.num_responses,  # Tell it to ask backend for N
            return_all_responses=True
        )

        if not candidate_plan_code_summary_tuples or \
           (isinstance(candidate_plan_code_summary_tuples, list) and \
            all(p.startswith("LLM Query Error:") or p.startswith("EXTRACTION_FAILED") for p,_,_ in candidate_plan_code_summary_tuples if p)): # Check if all are errors
            logger.error(f"{log_prefix_improve}: Failed to generate any improvement candidate pairs.")
            error_node = Node(
                plan="IMPROVEMENT_CANDIDATE_GENERATION_FAILED",
                code=parent_node.code, # Fallback to parent code
                summary="Failed to generate improvement candidates via self-consistency.",
                task_summary=parent_node.task_summary
            )
            error_node.parent = parent_node
            error_node.is_buggy = True; error_node.metric = WorstMetricValue()
            error_node.analysis = "Self-consistency improve failed: No improvement candidates generated."
            return error_node
        logger.info(f"{log_prefix_improve}: Finished generating improvement candidate pairs. Number of candidates: {len(candidate_plan_code_summary_tuples)}", extra={"verbose": True})
        print(f"finished generating improvement candidate pairs, now evaluating and selecting the best one\n\n")
   
        # 3. Evaluate these N pairs and select the best one
        chosen_evaluated_temp_node = self._evaluate_and_select_plan_code_pairs(
            candidate_plan_code_summary_tuples=candidate_plan_code_summary_tuples,
            parent_node_for_lineage=parent_node,
            operation_type="improve",
            selection_strategy="interpreter_best_metric"
        )

        # 4. Create the final Node for the journal
        final_journal_node = Node(
            plan=chosen_evaluated_temp_node.plan, # Plan from the chosen candidate
            code=chosen_evaluated_temp_node.code, # Code from the chosen candidate
            summary=chosen_evaluated_temp_node.summary,
            task_summary=chosen_evaluated_temp_node.task_summary,
            parent=parent_node, # Set parent for the journal tree
            
            _term_out=chosen_evaluated_temp_node._term_out,
            exec_time=chosen_evaluated_temp_node.exec_time,
            exc_type=chosen_evaluated_temp_node.exc_type,
            exc_info=chosen_evaluated_temp_node.exc_info,
            exc_stack=chosen_evaluated_temp_node.exc_stack,
            analysis=chosen_evaluated_temp_node.analysis,
            metric=chosen_evaluated_temp_node.metric,
            code_quality=chosen_evaluated_temp_node.code_quality,
            is_buggy=chosen_evaluated_temp_node.is_buggy
        )

        logger.info(f"{log_prefix_improve}: Improvement node {final_journal_node.id} created via SC. "
                    f"Buggy: {final_journal_node.is_buggy}, Metric: {final_journal_node.metric.value if final_journal_node.metric else 'N/A'}",
                    extra={"verbose":True})
        return final_journal_node

    def _debug(self, parent_node: Node) -> Node:
        log_prefix_debug = f"SC_AGENT_DEBUG_Step:{self.current_step}"
        
        logger.info(f"{log_prefix_debug}: Starting self-consistency debug for node {parent_node.id} (N={self.acfg.self_consistency.num_responses}).", extra={"verbose": True})

        debug_sys_prompt = get_agent_debug_system_prompt() 

        debug_user_prompt_dict = get_agent_debug_user_prompt(
            task_desc=self.task_desc,
            competition_name=self.competition_name,
            parent_node_code=parent_node.code,
            parent_node_term_out=parent_node.term_out,
            parent_node_feedback=parent_node.analysis, 
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        
        candidate_plan_code_summary_tuples: List[Tuple[str, str, str]] = self.plan_and_code_query(
            user_prompt_dict=debug_user_prompt_dict,
            system_prompt_dict=debug_sys_prompt,
            retries=3,
            return_all_responses=True,
            num_responses=self.acfg.self_consistency.num_responses
        )

        if not candidate_plan_code_summary_tuples or \
           (isinstance(candidate_plan_code_summary_tuples, list) and \
            all(p.startswith("LLM Query Error:") or p.startswith("EXTRACTION_FAILED") for p,_,_ in candidate_plan_code_summary_tuples if p)):
            logger.error(f"{log_prefix_debug}: Failed to generate any debug candidate pairs.")
            error_node = Node(
                plan="DEBUG_CANDIDATE_GENERATION_FAILED",
                code=parent_node.code, # Fallback to parent's buggy code
                summary="Failed to generate debug candidates via self-consistency.",
                task_summary=parent_node.task_summary
            )
            error_node.parent = parent_node
            error_node.is_buggy = True; error_node.metric = WorstMetricValue() # Remains buggy
            error_node.analysis = "Self-consistency debug failed: No debug candidates generated."
            return error_node

        chosen_evaluated_temp_node = self._evaluate_and_select_plan_code_pairs(
            candidate_plan_code_summary_tuples=candidate_plan_code_summary_tuples,
            parent_node_for_lineage=parent_node,
            operation_type="debug"
        )

        # 4. Create the final Node for the journal
        final_journal_node = Node(
            plan=chosen_evaluated_temp_node.plan,
            code=chosen_evaluated_temp_node.code,
            summary=chosen_evaluated_temp_node.summary,
            task_summary=chosen_evaluated_temp_node.task_summary,
            parent=parent_node,
            
            _term_out=chosen_evaluated_temp_node._term_out,
            exec_time=chosen_evaluated_temp_node.exec_time,
            exc_type=chosen_evaluated_temp_node.exc_type,
            exc_info=chosen_evaluated_temp_node.exc_info,
            exc_stack=chosen_evaluated_temp_node.exc_stack,
            analysis=chosen_evaluated_temp_node.analysis,
            metric=chosen_evaluated_temp_node.metric,
            code_quality=chosen_evaluated_temp_node.code_quality,
            is_buggy=chosen_evaluated_temp_node.is_buggy
        )

        logger.info(f"{log_prefix_debug}: Debug node {final_journal_node.id} created via SC. "
                    f"Buggy: {final_journal_node.is_buggy}, Metric: {final_journal_node.metric.value if final_journal_node.metric else 'N/A'}",
                    extra={"verbose":True})
        return final_journal_node

#############################################################################
# Tot Implementation
#############################################################################