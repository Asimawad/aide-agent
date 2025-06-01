# aide/agent.py
import shutil
import logging
import random
import re
import json
import time
from pathlib import Path 
from rich.console import Console 
from typing import Any, Callable, cast, Optional, Dict ,List, Tuple
from .backend import query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview 
from .utils.config import Config
from .utils.pretty_logging import log_step 
from .backend.utils import ContextLengthExceededError 
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
from .utils.metric import MetricValue, WorstMetricValue 

from .utils.prompt_utils import (
    get_agent_draft_user_prompt,
    get_agent_improve_user_prompt,
    review_func_spec,
    get_agent_debug_user_prompt,
    CHAINED_CODER_USER_PROMPT_CONSTRUCTORS, 
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
    wrap_code as prompt_utils_wrap_code, 
    AGENT_debug_SYSTEM_PROMPT_DICT, 
    AGENT_improve_SYSTEM_PROMPT_DICT, 
    get_chunked_reflection_system_prompt,
    get_chunked_reflection_user_prompt,
    get_tot_generate_initial_master_plans_user_prompt,
    get_tot_evaluate_master_plan_textual_user_prompt,
    get_tot_evaluator_system_prompt,
    get_tot_planner_system_prompt,
)


try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger("aide")
console = Console()

def format_time(time_in_sec: int): 
    time_in_sec = int(time_in_sec) 
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

############################################################################3
# Tree of Thoughts Agent Implementation (TOT)   
############################################################################3
class TOTAgent(Agent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_logger: Optional[WandbLogger] = None,
        competition_benchmarks: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(task_desc, cfg, journal, wandb_logger, competition_benchmarks)

    def parse_tot_response(self, eval_response: str) -> Tuple[str, str]:
        """
        Parses the response from the LLM and returns a tuple of the plan and the code.
        """
        score_match = re.search(r"Score:\s*(\d+)", eval_response, re.IGNORECASE)
        if score_match:
            try:
                return float(score_match.group(1))
            except ValueError:
                logger.warning(f"Could not parse score from evaluation: {eval_response}")
                return 0.0 
        logger.warning(f"Could not find score in evaluation: {eval_response}")
        return 0.0



    def _parse_multiple_master_plans(self, llm_response_text: str, num_expected: int) -> List[str]:
        plans = []
        raw_plans = re.split(r"Master Plan \d+:", llm_response_text, flags=re.IGNORECASE)
        for plan_text in raw_plans:
            plan_text = plan_text.strip()
            if plan_text:
                plans.append(plan_text)
        
        if not plans and llm_response_text.strip():
            logger.warning("Could not find 'Master Plan X:' delimiters, treating entire response as a single plan or multiple undelimited plans.")
            potential_plans = [p.strip() for p in llm_response_text.strip().split('\n\n') if p.strip()]
            if len(potential_plans) >= num_expected and len(potential_plans) <= num_expected + 2 : # Heuristic
                logger.info(f"Treating response as {len(potential_plans)} plans separated by double newlines.")
                return potential_plans
            elif potential_plans:
                logger.info("Treating entire response as a single plan due to lack of clear delimiters for multiple.")
                return [llm_response_text.strip()] 
            
        return plans[:num_expected] 
    

    def _generate_master_plan_with_tot(self, aide_input_context: Dict) -> str: 
        cfg_planning_tot = self.cfg.agent.tot.planning
        log_prefix = f"ToTAgent_MasterPlanToT_Step_{self.current_step}"

        if not cfg_planning_tot.enabled:
            logger.info(f"{log_prefix}: ToT for planning is disabled. Falling back to standard planner.")
            plan_user_prompt = get_planner_agent_draft_plan_user_prompt( 
                task_desc=aide_input_context.get("task_desc"),
                journal_summary=aide_input_context.get("journal_summary"),
                competition_name=aide_input_context.get("competition_name"),
                acfg_data_preview=self.acfg.data_preview, 
                data_preview_content=aide_input_context.get("data_preview_content")
            )
            _summary, single_plan, _err = self.plan_query(plan_user_prompt) 
            return single_plan if single_plan else "FALLBACK_PLAN_GENERATION_FAILED"
        
        logger.info(f"{log_prefix}: Starting ToT for Master Plan generation. Config: {cfg_planning_tot}")
       
        logger.info(f"{log_prefix}: Generating {cfg_planning_tot.n_generate_sample} initial Master Plan thoughts.")
        user_prompt_gen = get_tot_generate_initial_master_plans_user_prompt(
            aide_input_context, 
            cfg_planning_tot.n_generate_sample
        )
        raw_generated_plans_text = self._query_llm_with_retries( 
            query_type="TOT_PLAN_GEN",
            system_prompt=get_tot_planner_system_prompt(), 
            user_prompt=user_prompt_gen,
            model=self.acfg.code.planner_model, 
            temperature=self.acfg.code.temp, 
            planner_flag=True, 
            convert_system_to_user=self.acfg.convert_system_to_user,
        )

        candidate_plan_strings = self._parse_multiple_master_plans(raw_generated_plans_text, cfg_planning_tot.n_generate_sample)
        
        if not candidate_plan_strings:
            logger.error(f"{log_prefix}: Failed to generate or parse any Master Plan thoughts.")
            return "MASTER_PLAN_TOT_GENERATION_FAILED"

        logger.info(f"{log_prefix}: Generated {len(candidate_plan_strings)} plan thoughts.")
        for i, p_text in enumerate(candidate_plan_strings):
            logger.debug(f"{log_prefix}_GeneratedPlan_{i+1}_START\n{p_text}\n{log_prefix}_GeneratedPlan_{i+1}_END", extra={"verbose": True})


        evaluated_plans = [] 
        logger.info(f"{log_prefix}: Evaluating {len(candidate_plan_strings)} plan thoughts.")
        for i, plan_text in enumerate(candidate_plan_strings):
            user_prompt_eval = get_tot_evaluate_master_plan_textual_user_prompt(
                aide_input_context, 
                plan_text
            )
            evaluator_model_name = self.cfg.agent.tot.planning.get("evaluator_model_name") or self.cfg.agent.feedback.model 
            
            logger.info(f"{log_prefix}: Evaluating Plan {i+1} with model {evaluator_model_name}")
            logger.debug(f"{log_prefix}_PlanToEvaluate_{i+1}_START\n{plan_text}\n{log_prefix}_PlanToEvaluate_{i+1}_END", extra={"verbose": True})

            textual_eval_response = self._query_llm_with_retries(
                query_type="TOT_PLAN_EVAL",
                system_prompt=get_tot_evaluator_system_prompt(), 
                user_prompt=user_prompt_eval,
                model=evaluator_model_name, 
                temperature=self.cfg.agent.feedback.temp, 
                planner_flag=False, 
                convert_system_to_user=self.acfg.convert_system_to_user,
            )
            
            score = self._parse_textual_plan_evaluation(textual_eval_response)
            evaluated_plans.append({"plan_text": plan_text, "score": score, "raw_eval_response": textual_eval_response})
            logger.info(f"{log_prefix}: Plan {i+1} received score: {score}. Raw eval: {textual_eval_response[:100]}...")
            logger.debug(f"{log_prefix}_Plan_{i+1}_Score:{score}_RawEval:\n{textual_eval_response}\n", extra={"verbose": True})



        if not evaluated_plans:
            logger.error(f"{log_prefix}: No plans were successfully evaluated.")
            return "MASTER_PLAN_TOT_EVALUATION_FAILED"

        evaluated_plans.sort(key=lambda x: x["score"], reverse=True)
        
        best_n_plans_data = evaluated_plans[:cfg_planning_tot.n_select_sample]
        
        if not best_n_plans_data:
            logger.error(f"{log_prefix}: No plans selected after ToT process.")
            return "MASTER_PLAN_TOT_SELECTION_FAILED"

        chosen_plan_data = best_n_plans_data[0]
        logger.info(f"{log_prefix}: Selected Master Plan with score {chosen_plan_data['score']}.")
        logger.debug(f"{log_prefix}_CHOSEN_MASTER_PLAN_START\n{chosen_plan_data['plan_text']}\n{log_prefix}_CHOSEN_MASTER_PLAN_END", extra={"verbose":True})
        logger.debug(f"{log_prefix}_CHOSEN_MASTER_PLAN_RAW_EVAL_START\n{chosen_plan_data['raw_eval_response']}\n{log_prefix}_CHOSEN_MASTER_PLAN_RAW_EVAL_END", extra={"verbose":True})

        return chosen_plan_data["plan_text"]



    def _draft(self):
        aide_input_context = self._prepare_aide_input_for_tot(parent_node_being_expanded=None)
        selected_master_plan_text = self._generate_master_plan_with_tot(aide_input_context)
        
        draft_node = Node(plan=selected_master_plan_text, 
                        code=f"# Master Plan Selected by ToT:\n# {selected_master_plan_text.replace('\n', '\n# ')}\nprint('Plan selected, code generation TODO.')",
                        summary="Master Plan generated via ToT planning phase.")
        return draft_node

#############################################################################
# CodeChainAgent Implementation
#############################################################################
class CodeChainAgent(Agent): 
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_logger: Optional['WandbLogger'] = None,
        competition_benchmarks=None,
    ):
        super().__init__(task_desc, cfg, journal, wandb_logger, competition_benchmarks)


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
            except ContextLengthExceededError as cle: 
                logger.error(f"{log_prefix} Attempt {attempt+1}: Context Length Exceeded: {cle}. Aborting retries for this call.", exc_info=False, extra={"verbose": True}) 
                return None #
            except Exception as e:
                logger.error(f"{log_prefix} Attempt {attempt+1}: Error during LLM query: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: 
                    logger.error(f"{log_prefix}: All {retries} retries failed.", extra={"verbose": True})
                    return None 
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5)) 
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
                            system_prompt_dict: Dict[str, Any], 
                            retries: int = 3
                            ) -> str: 

            completion_text = self._query_llm_with_retries(
                query_type="Segment-Generation",
                system_prompt=system_prompt_dict, 
                user_prompt=user_prompt_dict,
                model=self.acfg.code.model, 
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
            current_code_so_far=code_accumulator, 
            competition_name=self.competition_name,
            data_preview_content=self.data_preview
        )
        
        code_snippet = self._code_segment_query( 
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
            logger.info(f"{log_prefix_segment}: Initial snippet generated. Now reflecting.")

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
        
        code_accumulator = f"# Script generated by AIDE CodeChainAgent (Chained Coder) - Step {self.current_step}\n"
        code_accumulator += f"# Competition: {self.competition_name}\n"
        code_accumulator += f"# Task Summary: {task_summary.splitlines()[0]}...\n" 
        code_accumulator += "# --- Master Plan ---\n"
        for i, plan_step_line in enumerate(master_plan_text.splitlines()):
            if plan_step_line.strip() and not plan_step_line.strip().startswith("##"): 
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
                code_accumulator += code_snippet + "\n\n" 
                if f"# FAILED TO GENERATE CODE FOR SEGMENT: {segment_name}" in code_snippet:
                    logger.warning(f"{log_prefix_chain}: Halting chain due to failure in segment: {segment_name}")
                    break 

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
                _, revised_chunk = self._reflect_on_chunk(
                    task_summary,
                    master_plan_text,
                    chunk,
                    code_before,
                    combined_chunk
                )
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

        reflection_llm = self.acfg.code.model 
        
        reflection_completion_text = self._query_llm_with_retries(
            query_type=f"CodeChainAgent_Reflect_Step: {self.current_step}_Segment_{segment_name.replace(' ', '_')}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=reflection_llm,
            temperature=self.acfg.code.temp, 

            convert_system_to_user=self.acfg.convert_system_to_user,
            retries=self.acfg.get('reflection_retries', 1),
            max_tokens=self.acfg.code.max_new_tokens
        )

        if reflection_completion_text is None:
            logger.warning(f"{log_prefix_reflect}: Reflection LLM query returned None. Using initial snippet.")
            return "Reflection failed: No LLM response.", initial_segment_snippet

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

            system_prompt = get_chunked_reflection_system_prompt()  
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

            return summary, revised


    def _draft(self, parent_node=None) -> Node:
        log_prefix = f""
        logger.info(f"{log_prefix} Starting drafting process. Parent: {parent_node.id if parent_node else 'None'}")
        memory=self.journal.generate_summary(include_code=False) 

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

            final_plan_text = master_plan_text 
            
            generated_code = self._draft_generate_code_chained(task_summary, master_plan_text)
            final_summary = task_summary 

            if not generated_code or generated_code.strip().startswith("# FAILED TO GENERATE CODE FOR SEGMENT:") or generated_code.strip() == "# SEGMENT 1 (Data Loading & Initial Setup) FAILED TO GENERATE":
                logger.error(f"{log_prefix} Chained code generation resulted in failure or predominantly error messages.")

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