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
    get_tot_evaluate_master_plan_func_call_user_prompt,
    CHAINED_CODER_USER_PROMPT_CONSTRUCTORS, 
    CHAINED_CODER_SYSTEM_PROMPT_GETTERS,
    tot_evaluate_master_plan_func_spec,
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
    get_tot_elaborate_high_level_plan_user_prompt,
    AGENT_debug_SYSTEM_PROMPT_DICT, 
    AGENT_improve_SYSTEM_PROMPT_DICT, 
    get_chunked_reflection_system_prompt,
    get_chunked_reflection_user_prompt,
    get_tot_generate_initial_master_plans_user_prompt,
    get_tot_planner_system_prompt,
    get_tot_evaluator_system_prompt,
    get_tot_segment_coder_system_prompt,
    get_tot_generate_segment_code_snippets_user_prompt,
    tot_evaluate_code_segment_func_spec,
    get_tot_segment_evaluator_system_prompt,
    get_tot_evaluate_segment_code_snippet_func_call_user_prompt,
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
    
    def _query_llm_with_retries( self, query_type: str, system_prompt: Dict[str, Any], user_prompt: Dict[str, Any], model: str, temperature: float, planner_flag: bool, convert_system_to_user: bool, func_spec: Optional[Dict[str, Any]] = None, retries: int = 3, max_tokens: Optional[int] = None) -> Any: # Add max_tokens
        completion_text = None
        log_prefix_query = f"TOTAGENT_LLM_QUERY_{query_type.upper()}_STEP{self.current_step}" # More specific log prefix
        effective_max_tokens = max_tokens if max_tokens is not None else self.acfg.code.max_new_tokens
        for attempt in range(retries):
            logger.info(f"{log_prefix_query}_ATTEMPT{attempt+1}/{retries}: Sending request. Model: {model}, Temp: {temperature}, PlannerFlag: {planner_flag}", extra={"verbose": True})
            try:
                completion_text = query(
                    system_message=system_prompt, 
                    user_message=user_prompt, 
                    model=model, 
                    temperature=temperature, 
                    planner=planner_flag, 
                    current_step=self.current_step, 
                    convert_system_to_user=convert_system_to_user,
                    max_tokens=effective_max_tokens,
                    func_spec=func_spec
                )
                # Log the full LLM output ONCE at debug level (verbose log only)
                logger.debug(f"{log_prefix_query}_LLM_OUTPUT_START\n{completion_text}\n{log_prefix_query}_LLM_OUTPUT_END", extra={"verbose": True})
                # Log a concise message to the terminal
                logger.info(f"{log_prefix_query}_ATTEMPT{attempt+1}: LLM response received.", extra={"verbose": True})
                return completion_text
            except ContextLengthExceededError as cle:
                logger.error(f"{log_prefix_query}_ATTEMPT{attempt+1}: Context Length Exceeded: {cle}. Aborting.", exc_info=False, extra={"verbose": True})
                return f"ERROR: Context Length Exceeded - {str(cle)}"
            except Exception as e:
                logger.error(f"{log_prefix_query}_ATTEMPT{attempt+1}: Error during LLM query: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: 
                    logger.error(f"{log_prefix_query}: All {retries} retries failed.", extra={"verbose": True})
                    return f"ERROR: LLM Query Failed after {retries} retries - {str(e)}"
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
        return "ERROR: LLM Query failed and exhausted retries without throwing specific exception."

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
        system_prompt = get_planner_agent_code_system_prompt(); log_prefix = f"CoderAgent_Code_QUERY_STEP: {self.current_step}"
        completion_text = self._query_llm_with_retries(query_type="PLANNER_CODER", system_prompt=system_prompt, user_prompt=user_prompt_dict,
                                                       model=self.acfg.code.model, temperature=self.acfg.code.temp,
                                                       planner_flag=False, convert_system_to_user=self.acfg.convert_system_to_user, retries=retries)
        if completion_text is None:
            return "", "", ""
        code = extract_code(completion_text)
        if not code:
            code = str(completion_text)
            logger.debug(f"{log_prefix}_LLM_OUTPUT_START\n{code}\n{log_prefix}_LLM_OUTPUT_END", extra={"verbose": True})
            logger.info(f"{log_prefix}: LLM response received, but code extraction failed.", extra={"verbose": True})
            return "", code, ""
        logger.debug(f"{log_prefix}_LLM_OUTPUT_START\n{code}\n{log_prefix}_LLM_OUTPUT_END", extra={"verbose": True})
        logger.info(f"{log_prefix}: LLM response received and code extracted.", extra={"verbose": True})
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
    
    def _query_llm_with_retries( self, query_type: str, system_prompt: Dict[str, Any], user_prompt: Dict[str, Any], model: str, temperature: float, planner_flag: bool, convert_system_to_user: bool, func_spec: Optional[Dict[str, Any]] = None, retries: int = 3, max_tokens: Optional[int] = None) -> Any: # Add max_tokens
        completion_text = None
        log_prefix_query = f"TOTAGENT_LLM_QUERY_{query_type.upper()}_STEP{self.current_step}" # More specific log prefix
        effective_max_tokens = max_tokens if max_tokens is not None else self.acfg.code.max_new_tokens
        # print the size of the system_prompt, user_prompt, and effective_max_tokens
        print(f"System prompt size: {len(system_prompt)}")
        print(f"User prompt size: {len(user_prompt)}")
        print(f"Effective max tokens: {effective_max_tokens}")
        print(f"--------------------------------------")
        for attempt in range(retries):
            logger.info(f"{log_prefix_query}_ATTEMPT{attempt+1}/{retries}: Sending request. Model: {model}, Temp: {temperature}, PlannerFlag: {planner_flag}", extra={"verbose": True})
            try:
                completion_text = query(
                    system_message=system_prompt, 
                    user_message=user_prompt, 
                    model=model, 
                    temperature=temperature, 
                    planner=planner_flag, 
                    current_step=self.current_step, 
                    convert_system_to_user=convert_system_to_user,
                    max_tokens=effective_max_tokens,
                    func_spec=func_spec
                )
                # Log the full LLM output ONCE at debug level (verbose log only)
                logger.debug(f"{log_prefix_query}_LLM_OUTPUT_START\n{completion_text}\n{log_prefix_query}_LLM_OUTPUT_END", extra={"verbose": True})
                # Log a concise message to the terminal
                logger.info(f"{log_prefix_query}_ATTEMPT{attempt+1}: LLM response received.", extra={"verbose": True})
                return completion_text
            except ContextLengthExceededError as cle:
                logger.error(f"{log_prefix_query}_ATTEMPT{attempt+1}: Context Length Exceeded: {cle}. Aborting.", exc_info=False, extra={"verbose": True})
                return f"ERROR: Context Length Exceeded - {str(cle)}"
            except Exception as e:
                logger.error(f"{log_prefix_query}_ATTEMPT{attempt+1}: Error during LLM query: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: 
                    logger.error(f"{log_prefix_query}: All {retries} retries failed.", extra={"verbose": True})
                    return f"ERROR: LLM Query Failed after {retries} retries - {str(e)}"
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
        return "ERROR: LLM Query failed and exhausted retries without throwing specific exception."

    def code_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        system_prompt = get_planner_agent_code_system_prompt()
        log_prefix = f"CoderAgent_Code_QUERY_STEP: {self.current_step}"
        completion_text = self._query_llm_with_retries(query_type="PLANNER_CODER", system_prompt=system_prompt, user_prompt=user_prompt_dict,
                                                       model=self.acfg.code.model, temperature=self.acfg.code.temp,
                                                       planner_flag=False, convert_system_to_user=self.acfg.convert_system_to_user, retries=retries)
        if completion_text is None:
            return "", "", ""
        code = extract_code(completion_text)
        if not code:
            code = str(completion_text)
            logger.debug(f"{log_prefix}_LLM_OUTPUT_START\n{code}\n{log_prefix}_LLM_OUTPUT_END", extra={"verbose": True})
            logger.info(f"{log_prefix}: LLM response received, but code extraction failed.", extra={"verbose": True})
            return "", code, ""
        logger.debug(f"{log_prefix}_LLM_OUTPUT_START\n{code}\n{log_prefix}_LLM_OUTPUT_END", extra={"verbose": True})
        logger.info(f"{log_prefix}: LLM response received and code extracted.", extra={"verbose": True})
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
        logger.debug(f"Segment-Generation_LLM_OUTPUT_START\n{completion_text}\nSegment-Generation_LLM_OUTPUT_END", extra={"verbose": True})
        return completion_text.strip() if completion_text else ""

    def _generate_code_segment(self,
                               segment_name: str,
                               task_summary: str,
                               master_plan_text: str,
                               code_accumulator: str,
                               chain_reflection: bool=False,
                               ) -> str:
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
        logger.debug(f"{log_prefix_segment}_SEGMENT_LLM_OUTPUT_START\n{code_snippet.strip()}\n{log_prefix_segment}_SEGMENT_LLM_OUTPUT_END", extra={"verbose": True})
        logger.info(f"{log_prefix_segment}: Code segment generated for {segment_name}.", extra={"verbose": True})
        if chain_reflection:
            logger.info(f"{log_prefix_segment}: Initial snippet generated. Now reflecting.")
            reflection_summary, code_snippet = self._reflect_on_segment(
                task_summary=task_summary,
                master_plan_text=master_plan_text,
                segment_name=segment_name,
                code_before_segment=code_accumulator,
                initial_segment_snippet=code_snippet
            )
        return code_snippet.strip() 

    def _parse_multiple_master_plans(self, llm_response_text: str, num_expected: int) -> List[str]:
        log_prefix = f"ToTAgent_ParseMasterPlans_Step_{self.current_step}"
        if not llm_response_text or not llm_response_text.strip():
            logger.warning(f"{log_prefix}: Received empty or whitespace-only response from LLM for plan generation.")
            return []
        separator = "<!--- PLAN_SEPARATOR --->"
        candidate_plan_strings = [plan.strip() for plan in llm_response_text.split(separator) if plan.strip()]
        if not candidate_plan_strings:
            logger.warning(f"{log_prefix}: No plans found using separator '{separator}'. Treating entire response as a single plan.")
            logger.debug(f"{log_prefix}_LLM_OUTPUT_START\n{llm_response_text.strip()}\n{log_prefix}_LLM_OUTPUT_END", extra={"verbose": True})
            return [llm_response_text.strip()]
        if len(candidate_plan_strings) != num_expected:
            logger.warning(f"{log_prefix}: Parsed {len(candidate_plan_strings)} plans, but expected {num_expected}. Proceeding with parsed plans.")
            for i, p_text in enumerate(candidate_plan_strings):
                logger.debug(f"{log_prefix}_ParsedPlanCandidate_{i+1}_LLM_OUTPUT_START\n{p_text}\n{log_prefix}_ParsedPlanCandidate_{i+1}_LLM_OUTPUT_END", extra={"verbose": True})
        return candidate_plan_strings

    def _parse_multiple_code_snippets(self, llm_response_text: str, num_expected: int) -> List[str]:
        log_prefix = f"ToTAgent_ParseSnippets_Step_{self.current_step}"
        if not llm_response_text or not llm_response_text.strip():
            logger.warning(f"{log_prefix}: Received empty or whitespace-only response from LLM for snippet generation.")
            return []
        separator = "<!--- SNIPPET_SEPARATOR --->"
        raw_snippet_blocks = [block.strip() for block in llm_response_text.split(separator) if block.strip()]

        clean_snippets = []
        if not raw_snippet_blocks and llm_response_text.strip(): 
            logger.warning(f"{log_prefix}: Snippet separator '{separator}' not found. Attempting to extract code from entire response as one block.")
            extracted = extract_code(llm_response_text) 
            if extracted:
                clean_snippets.append(extracted)
        else:
            for i, block_text in enumerate(raw_snippet_blocks):
                extracted = extract_code(block_text) 
                logger.debug(f"DEBUG_PARSE_SNIPPET - Snippet after extract_code: >>>{extracted[:100]}...<<<") # TEMP DEBUG
                if extracted:
                    clean_snippets.append(extracted)
                else:
                    logger.warning(f"{log_prefix}: Could not extract valid code from snippet block {i+1}. Block content (truncated): {block_text[:200]}...")
        
        if not clean_snippets:
            logger.error(f"{log_prefix}: Failed to parse or extract any valid code snippets from LLM response: {llm_response_text[:500]}...")
            return []

        logger.info(f"{log_prefix}: Successfully parsed and extracted {len(clean_snippets)} code snippets.")
        return clean_snippets
    
    def _evaluate_single_plan_thought(self, aide_input_context: Dict, plan_text_candidate: str, log_prefix_eval: str, plan_candidate_idx: int) -> Dict:
        user_prompt_eval = get_tot_evaluate_master_plan_func_call_user_prompt(
            aide_input_context,
            plan_text_candidate
        )
        evaluator_model_name = self.cfg.agent.tot.planning.get("evaluator_model_name") or self.cfg.agent.feedback.model
        logger.info(f"{log_prefix_eval}: Evaluating Plan Candidate {plan_candidate_idx} (truncated): '{plan_text_candidate[:70].replace(chr(10), ' ')}...' with model {evaluator_model_name}")
        evaluation_dict_response = self._query_llm_with_retries(
            query_type=f"TOT_PLAN_EVAL_FUNC_CALL_STEP{self.current_step}_PLAN{plan_candidate_idx}",
            system_prompt=get_tot_evaluator_system_prompt(),
            user_prompt=user_prompt_eval,
            model=evaluator_model_name,
            temperature=self.cfg.agent.feedback.temp,
            func_spec=tot_evaluate_master_plan_func_spec, 
            planner_flag=False,
            convert_system_to_user=self.acfg.convert_system_to_user,
        )
        logger.debug(f"{log_prefix_eval}_EVAL_LLM_OUTPUT_START\n{evaluation_dict_response}\n{log_prefix_eval}_EVAL_LLM_OUTPUT_END", extra={"verbose": True})
        score = 0.0
        is_single_plan = False
        if isinstance(evaluation_dict_response, dict):
            score = evaluation_dict_response.get("plan_score", 0.0)
            is_single_plan = evaluation_dict_response.get("is_single_coherent_plan", False)
            if not is_single_plan:
                logger.warning(f"{log_prefix_eval}: Evaluator determined plan candidate {plan_candidate_idx} is not a single coherent plan. Score set to 0 or penalized.")
                score = 0.0
            logger.info(f"{log_prefix_eval}: Plan Candidate {plan_candidate_idx} received score: {score}.")
            return {
                "plan_text": plan_text_candidate,
                "score": float(score) if score is not None else 0.0,
                "evaluation_details": evaluation_dict_response
            }
        else:
            logger.error(f"{log_prefix_eval}: Evaluation for Plan Candidate {plan_candidate_idx} did not return a dictionary. Response: {evaluation_dict_response}")
            return {"plan_text": plan_text_candidate, "score": 0.0, "evaluation_details": {"error": "Invalid evaluation response format"}}

    
    def _evaluate_single_code_snippet_thought(
        self, 
        aide_context_seg: Dict[str, Any],
        master_plan_text: str,
        current_segment_name: str,
        code_generated_so_far: str,
        segment_snippet_to_evaluate: str,
        log_prefix_eval_seg: str,
        snippet_candidate_idx: int
    ) -> Dict: 
        """
        Evaluates a single code snippet thought for a given segment using LLM function call.
        Returns the dictionary from the function call, or an error dict.
        """
        user_prompt_eval_snippet = get_tot_evaluate_segment_code_snippet_func_call_user_prompt(
            aide_context_seg,
            master_plan_text,
            current_segment_name,
            code_generated_so_far,
            segment_snippet_to_evaluate
        )
        
        evaluator_model_name_seg = self.cfg.agent.tot.segment_coding.get("evaluator_model_name") or self.cfg.agent.feedback.model
        
        log_eval_prefix_full = f"{log_prefix_eval_seg}_SNIP{snippet_candidate_idx}"
        logger.info(f"{log_eval_prefix_full}: Evaluating snippet for segment '{current_segment_name}' ('{segment_snippet_to_evaluate[:70].replace(chr(10),' ')}...') with model {evaluator_model_name_seg}")

        evaluation_dict_response = self._query_llm_with_retries(
            query_type=f"TOT_SEG_SNIP_EVAL_{current_segment_name.replace(' ','_')}_IDX{snippet_candidate_idx}",
            system_prompt=get_tot_segment_evaluator_system_prompt(),
            user_prompt=user_prompt_eval_snippet,
            model=evaluator_model_name_seg,
            temperature=self.cfg.agent.feedback.temp, 
            func_spec=tot_evaluate_code_segment_func_spec,
            planner_flag=False,
            convert_system_to_user=self.acfg.convert_system_to_user,
        )

        if isinstance(evaluation_dict_response, dict):
                logger.info(f"{log_eval_prefix_full}: Snippet for segment '{current_segment_name}' received detailed evaluation: Score {evaluation_dict_response.get('overall_quality_score', 'N/A')}, Issues: {evaluation_dict_response.get('identified_issues_or_risks', 'N/A')}")
                logger.debug(f"{log_eval_prefix_full}_FullEvalDict: {json.dumps(evaluation_dict_response, indent=2)}", extra={"verbose": True})
                return evaluation_dict_response
        else:
            logger.error(f"{log_eval_prefix_full}: Evaluation for snippet (segment '{current_segment_name}') did not return a dictionary. Response: {evaluation_dict_response}")
            return { 
                "overall_quality_score": 0.0, "correctness_and_robustness_score": 0.0, 
                "plan_adherence_score": 0.0, "integration_score": 0.0, 
                "clarity_and_best_practices_score": 0.0,
                "identified_issues_or_risks": "Evaluation LLM did not return a valid structured response.",
                "positive_remarks": "None",
                "error": "Invalid evaluation response format"
            }

    def _generate_master_plan_with_tot(self, aide_input_context: Dict) -> str:
        cfg_planning_tot = self.cfg.agent.tot.planning
        log_prefix_base = f"ToTAgent_MasterPlanToT_Step_{self.current_step}"

        if not cfg_planning_tot.enabled:
            logger.info(f"{log_prefix_base}: ToT for planning is disabled. Falling back to standard planner.")
            # This fallback needs self.plan_query. If TOTAgent doesn't inherit PlannerAgent, this needs adjustment.
            # For now, assuming self.plan_query exists (e.g., copied from PlannerAgent or TOTAgent inherits it)
            plan_user_prompt = get_planner_agent_draft_plan_user_prompt(
                task_desc=aide_input_context.get("task_desc"),
                journal_summary=aide_input_context.get("journal_summary"),
                competition_name=aide_input_context.get("competition_name"),
                acfg_data_preview=self.acfg.data_preview,
                data_preview_content=aide_input_context.get("data_preview_content")
            )
            _summary, single_plan, _err = self.plan_query(plan_user_prompt) # Call the PlannerAgent's method
            return single_plan if single_plan else "FALLBACK_PLAN_GENERATION_FAILED"

        logger.info(f"{log_prefix_base}: Starting 2-Step ToT for Master Plan generation. Config: {cfg_planning_tot}")

        # --- ToT Step 1: Generate and Select High-Level Strategic Plans ---
        log_prefix_step1 = f"{log_prefix_base}_ToTStep1_HighLevel"
        logger.info(f"{log_prefix_step1}: Generating {cfg_planning_tot.n_generate_sample} initial high-level Master Plan thoughts.")
        
        user_prompt_gen_hl = get_tot_generate_initial_master_plans_user_prompt(
            aide_input_context,
            cfg_planning_tot.n_generate_sample # Ask for k initial plans
        )
        raw_generated_hl_plans_text = self._query_llm_with_retries(
            query_type="TOT_HL_PLAN_GEN",
            system_prompt=get_tot_planner_system_prompt(), # System prompt for generating diverse high-level plans
            user_prompt=user_prompt_gen_hl,
            model=self.acfg.code.planner_model,
            temperature=self.acfg.code.temp, # Consider a slightly higher temp for diversity here
            planner_flag=True,
            convert_system_to_user=self.acfg.convert_system_to_user,
        )
        candidate_hl_plan_strings = self._parse_multiple_master_plans(raw_generated_hl_plans_text, cfg_planning_tot.n_generate_sample)

        if not candidate_hl_plan_strings:
            logger.error(f"{log_prefix_step1}: Failed to generate or parse any high-level Master Plan thoughts.")
            return "MASTER_PLAN_TOT_HL_GENERATION_FAILED"
        logger.info(f"{log_prefix_step1}: Generated {len(candidate_hl_plan_strings)} high-level plan thoughts.")

        evaluated_hl_plans = []
        for i, hl_plan_text in enumerate(candidate_hl_plan_strings):
            eval_data = self._evaluate_single_plan_thought(aide_input_context, hl_plan_text, log_prefix_step1, i + 1)
            evaluated_hl_plans.append(eval_data)

        if not any(p['score'] > 0 for p in evaluated_hl_plans): # Check if any plan got a positive score
            logger.error(f"{log_prefix_step1}: No high-level plans were successfully evaluated with a positive score.")
            return "MASTER_PLAN_TOT_HL_EVALUATION_FAILED"
        
        evaluated_hl_plans.sort(key=lambda x: x["score"], reverse=True)
      
        num_hl_plans_to_elaborate = self.cfg.agent.tot.planning.get("n_high_level_select_for_elaboration", 2) # New config field
        
        best_n_hl_plans_data = evaluated_hl_plans[:num_hl_plans_to_elaborate]
        if not best_n_hl_plans_data:
            logger.error(f"{log_prefix_step1}: No high-level plans selected for elaboration.")
            return "MASTER_PLAN_TOT_HL_SELECTION_FAILED"
        logger.info(f"{log_prefix_step1}: Selected {len(best_n_hl_plans_data)} high-level plans for elaboration.")


        # --- ToT Step 2: Elaborate Selected High-Level Plans into Detailed Plans ---
        log_prefix_step2 = f"{log_prefix_base}_ToTStep2_Detailed"
        all_detailed_plans_evaluated = []
        
        for hl_plan_idx, hl_plan_data in enumerate(best_n_hl_plans_data):
            high_level_plan_text = hl_plan_data["plan_text"]
            logger.info(f"{log_prefix_step2}: Elaborating high-level plan {hl_plan_idx+1} (Score: {hl_plan_data['score']}): '{high_level_plan_text[:70].replace(chr(10), ' ')}...'")

            num_detailed_versions = cfg_planning_tot.n_generate_sample 

            user_prompt_detail = get_tot_elaborate_high_level_plan_user_prompt(
                aide_input_context,
                high_level_plan_text,
                num_detailed_versions
            )
            raw_generated_detailed_plans_text = self._query_llm_with_retries(
                query_type=f"TOT_DETAILED_PLAN_GEN_FROM_HL{hl_plan_idx+1}",
                system_prompt=get_planner_agent_plan_system_prompt(), 
                user_prompt=user_prompt_detail,
                model=self.acfg.code.planner_model,
                temperature=self.acfg.code.temp, 
                planner_flag=True,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )
            candidate_detailed_plan_strings = self._parse_multiple_master_plans(raw_generated_detailed_plans_text, num_detailed_versions)

            if not candidate_detailed_plan_strings:
                logger.warning(f"{log_prefix_step2}: Failed to generate or parse detailed plans for high-level plan {hl_plan_idx+1}.")
                continue

            logger.info(f"{log_prefix_step2}: Generated {len(candidate_detailed_plan_strings)} detailed plan thoughts for HL plan {hl_plan_idx+1}.")
            
            for detail_idx, detailed_plan_text in enumerate(candidate_detailed_plan_strings):
                eval_data = self._evaluate_single_plan_thought(aide_input_context, detailed_plan_text, log_prefix_step2, detail_idx + 1)
                eval_data["source_high_level_plan"] = high_level_plan_text # Keep track of origin
                all_detailed_plans_evaluated.append(eval_data)

        if not all_detailed_plans_evaluated:
            logger.error(f"{log_prefix_step2}: No detailed plans were generated or evaluated successfully across all high-level branches.")
            return best_n_hl_plans_data[0]["plan_text"] if best_n_hl_plans_data else "MASTER_PLAN_TOT_DETAIL_GENERATION_FAILED"
        all_detailed_plans_evaluated.sort(key=lambda x: x["score"], reverse=True)
        final_best_n_detailed_plans = all_detailed_plans_evaluated[:cfg_planning_tot.n_select_sample]

        if not final_best_n_detailed_plans:
            logger.error(f"{log_prefix_base}: No detailed plans selected after ToT Step 2.")
            # Fallback:
            return best_n_hl_plans_data[0]["plan_text"] if best_n_hl_plans_data else "MASTER_PLAN_TOT_FINAL_SELECTION_FAILED"

        # We want to return just ONE best plan for the coder
        chosen_detailed_plan_data = final_best_n_detailed_plans[0]
        logger.info(f"{log_prefix_base}: Final selected detailed Master Plan (Score: {chosen_detailed_plan_data['score']}). Source HL Plan: '{chosen_detailed_plan_data.get('source_high_level_plan', 'N/A')[:70].replace(chr(10), ' ')}...'")
        logger.debug(f"{log_prefix_base}_CHOSEN_DETAILED_MASTER_PLAN_START\n{chosen_detailed_plan_data['plan_text']}\n{log_prefix_base}_CHOSEN_DETAILED_MASTER_PLAN_END", extra={"verbose":True})
        logger.debug(f"{log_prefix_base}_CHOSEN_DETAILED_MASTER_PLAN_RAW_EVAL_START\n{chosen_detailed_plan_data['evaluation_details']}\n{log_prefix_base}_CHOSEN_DETAILED_MASTER_PLAN_RAW_EVAL_END", extra={"verbose":True})
        return chosen_detailed_plan_data["plan_text"]

    def _prepare_aide_input_for_tot(self, parent_node_being_expanded: Optional[Node] = None) -> Dict[str, Any]:
        log_prefix = f"ToTAgent_PrepContext_Step_{self.current_step}"
        logger.info(f"{log_prefix}: Preparing AIDE input context for ToT.", extra={"verbose": True})

        if self.data_preview is None: 
            self.update_data_preview() 


        # investigate how many tokens are in the context
        print(f"data_preview size: {len(self.data_preview)}")
        print(f"--------------------------------------")
        context = {
            "task_desc": self.task_desc, # From Agent.__init__
            "data_preview_content": self.data_preview, # From Agent.update_data_preview()
            "journal_summary": self.journal.generate_summary(include_code=False), # From Agent.journal
            "competition_name": self.cfg.competition_name # From Agent.cfg
        }
        if parent_node_being_expanded:
            context["current_code_or_plan"] = parent_node_being_expanded.plan or parent_node_being_expanded.code
            context["previous_analysis"] = parent_node_being_expanded.analysis
        # investigate how many tokens are in the journal_summary
        print(f"Journal summary size: {len(context['journal_summary'])}")
        print(f"--------------------------------------")
        logger.debug(f"{log_prefix}_ContextDetails_START\n"
                     f"TaskDesc: {str(context['task_desc'])[:200]}...\n"
                     f"DataPreview: {str(context['data_preview_content'])[:200]}...\n"
                     f"JournalSummary: {str(context['journal_summary'])[:200]}...\n"
                     f"Competition: {context['competition_name']}\n"
                     f"{log_prefix}_ContextDetails_END", extra={"verbose": True})
        return context
    
    def _draft_generate_code_chained(self, task_summary: str, master_plan_text: str) -> str:
        cfg_segment_tot = self.cfg.agent.tot.segment_coding
        log_prefix_chain = f"ToTAgent_ChainedDraft_AIDESTEP{self.current_step}"
        logger.info(f"{log_prefix_chain}: Starting ToT-BFS chained code generation for draft. Segment ToT enabled: {cfg_segment_tot.enabled}")

        initial_boilerplate = f"# Script generated by AIDE TOTAgent - AIDE Step {self.current_step}\n"
        initial_boilerplate += f"# Competition: {self.competition_name}\n"
        initial_boilerplate += f"# Task Summary: {task_summary.splitlines()[0]}...\n"
        initial_boilerplate += "# --- Master Plan (Selected by ToT Planning Phase) ---\n"
        for i, plan_step_line in enumerate(master_plan_text.splitlines()):
            if plan_step_line.strip() and not plan_step_line.strip().startswith("##"):
                initial_boilerplate += f"# {plan_step_line.strip()}\n"
        initial_boilerplate += "# --- End Master Plan ---\n\n"
        current_beam: List[Dict[str, Any]] = [{"code_acc": initial_boilerplate, "last_snippet_score": 10.0, "path_eval_details": []}] 

        segments_order = [
            "Setup & Imports", "Data Loading", "Data Preprocessing",
            "Modeling", "Training & Validation", "Prediction & Submission"
        ]

        aide_context_seg = self._prepare_aide_input_for_tot() 

        for segment_idx, segment_name in enumerate(segments_order):
            log_prefix_segment_loop = f"{log_prefix_chain}_Segment_{segment_name.replace(' ', '_')}"
            logger.info(f"{log_prefix_segment_loop}: Processing segment {segment_idx+1}/{len(segments_order)}: '{segment_name}'")
            
            next_beam_candidates: List[Dict[str, Any]] = []

            for beam_item_idx, current_path_data in enumerate(current_beam):
                current_code_acc = current_path_data["code_acc"]
                log_prefix_beam_item = f"{log_prefix_segment_loop}_BeamItem{beam_item_idx+1}"
                logger.info(f"{log_prefix_beam_item}: Expanding from accumulator (last snippet score: {current_path_data['last_snippet_score']})")
                 
            

                top_snippet_thoughts_for_this_segment: List[Dict] = self._generate_and_select_top_snippet_thoughts_for_segment(
                    segment_name,
                    task_summary,
                    master_plan_text,
                    current_code_acc,
                    aide_context_seg 
                )

                if not top_snippet_thoughts_for_this_segment:
                    logger.warning(f"{log_prefix_beam_item}: ToT for segment '{segment_name}' yielded no viable snippets. Path terminated.")
                    continue 
                logger.info(f"{log_prefix_beam_item}: ToT for segment '{segment_name}' yielded {len(top_snippet_thoughts_for_this_segment)} viable snippets.")

                for snippet_data in top_snippet_thoughts_for_this_segment:
                    snippet_text = snippet_data["snippet_text"] 
                    snippet_score = snippet_data["score"]
                    logger.info(f"{log_prefix_beam_item}: Adding snippet to beam: '{snippet_text[:70].replace(chr(10), ' ')}...' (Score: {snippet_score})")
                    
                    next_beam_candidates.append({
                        "code_acc": current_code_acc + snippet_text + "\n\n",
                        "last_snippet_score": snippet_score,
                        "path_eval_details": current_path_data["path_eval_details"] + [{"segment": segment_name, "score": snippet_score, "method": "ToT"}]
                    })


            if not next_beam_candidates:
                logger.error(f"{log_prefix_chain}: No viable candidates generated for segment '{segment_name}'. Halting script generation.")
                current_beam.sort(key=lambda x: x["last_snippet_score"], reverse=True)
                return current_beam[0]["code_acc"] + f"\n\n# ERROR: Failed to generate next segment '{segment_name}' for any path.\n"

            inter_segment_beam_width = self.cfg.agent.tot.segment_coding.get("inter_segment_beam_width", 1) 
            next_beam_candidates.sort(key=lambda x: x["last_snippet_score"], reverse=True)
            current_beam = next_beam_candidates[:inter_segment_beam_width]
            
            logger.info(f"{log_prefix_segment_loop}: Selected {len(current_beam)} paths for next segment. Best score for this segment: {current_beam[0]['last_snippet_score'] if current_beam else 'N/A'}")
            for i, cb_data in enumerate(current_beam):
                logger.debug(f"{log_prefix_segment_loop}_SelectedPath{i+1}_LastSnippetScore: {cb_data['last_snippet_score']}", extra={"verbose":True})
                logger.debug(f"{log_prefix_segment_loop}_SelectedPath{i+1}_AccumulatedCodeSoFar_END:\n{cb_data['code_acc'][-500:]}...", extra={"verbose":True})


        # After all segments are processed
        if not current_beam:
            logger.error(f"{log_prefix_chain}: BFS code generation failed, final beam is empty.")
            return initial_boilerplate + "\n\n# ERROR: ToT-BFS code generation resulted in an empty final beam.\n"

        # Return the code from the best path in the final beam
        current_beam.sort(key=lambda x: x["last_snippet_score"], reverse=True) 
        final_best_code_acc = current_beam[0]["code_acc"]
        final_path_eval_details = current_beam[0]["path_eval_details"]
        
        logger.info(f"{log_prefix_chain}: ToT-BFS chained code generation complete.")
        logger.info(f"{log_prefix_chain}: Final chosen path evaluation details: {json.dumps(final_path_eval_details, indent=2)}")
        logger.debug(f"{log_prefix_chain}_FINAL_SCRIPT_START\n{final_best_code_acc}\n{log_prefix_chain}_FINAL_SCRIPT_END", extra={"verbose":True})
        
        return final_best_code_acc.strip()

    def _generate_and_select_top_snippet_thoughts_for_segment(self,
                                                            segment_name: str,
                                                            task_summary: str,
                                                            master_plan_text: str,
                                                            current_code_accumulator_context: str, # Code before this segment
                                                            aide_context_seg: Dict[str, Any]
                                                        ) -> List[Dict[str, Any]]: # List of {"snippet_text": str, "score": float, "eval_dict": dict}
        """
        Encapsulates the Gen-Eval-Select ToT logic for a single code segment.
        Returns a list of the top N selected snippet data dictionaries.
        """
        cfg_segment_tot = self.cfg.agent.tot.segment_coding # Already has n_generate_sample, n_select_sample
        log_prefix = f"ToTAgent_GenEvalSelect_AIDESTEP{self.current_step}_SEG_{segment_name.replace(' ', '_')}"

        # 1. Generation
        user_prompt_gen_snip = get_tot_generate_segment_code_snippets_user_prompt(
            aide_context_seg, master_plan_text, segment_name, current_code_accumulator_context,
            cfg_segment_tot.n_generate_sample # k_thoughts for snippets
        )
        raw_snippets_text_response = self._query_llm_with_retries(
            query_type=f"TOT_SEG_SNIP_GEN_{segment_name.replace(' ','_').upper()}",
            system_prompt=get_tot_segment_coder_system_prompt(),
            user_prompt=user_prompt_gen_snip,
            model=self.acfg.code.model,
            temperature=self.acfg.code.temp,
            planner_flag=False,
            convert_system_to_user=self.acfg.convert_system_to_user,
            max_tokens=self.acfg.code.max_new_tokens
        )
        candidate_snippet_strings = self._parse_multiple_code_snippets(raw_snippets_text_response, cfg_segment_tot.n_generate_sample)

        if not candidate_snippet_strings:
            logger.error(f"{log_prefix}: Failed to generate/parse any code snippets.")
            return []

        # # 2. Evaluation
        # evaluated_snippets = []
        # for i, snippet_text in enumerate(candidate_snippet_strings):
        #     eval_dict = self._evaluate_single_code_snippet_thought(
        #         aide_context_seg, master_plan_text, segment_name, current_code_accumulator_context, snippet_text,
        #         log_prefix, i + 1
        #     )
        #     # Store the snippet text with its full evaluation dictionary
        #     evaluated_snippets.append({
        #         "snippet_text": snippet_text, # clean snippet
        #         "score": float(eval_dict.get("snippet_score", 0.0)),
        #         "evaluation_dict": eval_dict
        #     })

        evaluated_snippets = []
        for i, snippet_text in enumerate(candidate_snippet_strings):
            full_eval_dict = self._evaluate_single_code_snippet_thought( 
                aide_context_seg, master_plan_text, segment_name, current_code_accumulator_context, snippet_text,
                log_prefix, i + 1
            )
            evaluated_snippets.append({
                "snippet_text": snippet_text,
                "score": float(full_eval_dict.get("overall_quality_score", 0.0)),
                "evaluation_dict": full_eval_dict 
            })
        
        viable_snippets = []
        for data in evaluated_snippets:
            eval_d = data["evaluation_dict"]
            # and no critical identified issues.
            is_viable = (
                eval_d.get("overall_quality_score", 0.0) >= 5 and # Example threshold
                eval_d.get("correctness_and_robustness_score", 0.0) >= 5 and
                eval_d.get("plan_adherence_score", 0.0) >= 6 and
                eval_d.get("integration_score", 0.0) >= 5 and
                (eval_d.get("identified_issues_or_risks", "").lower() == "none" or 
                "minor" in eval_d.get("identified_issues_or_risks", "").lower()) # Allow minor issues if scores are high
            )
            if is_viable:
                viable_snippets.append(data)
            else:
                logger.info(f"{log_prefix}: Snippet '{data['snippet_text'][:50].replace(chr(10),' ')}...' deemed not viable. Eval: {eval_d}")


        if not viable_snippets:
            logger.warning(f"{log_prefix}: No viable snippets after detailed evaluation. Using highest originally scored if any.")
            if not evaluated_snippets: return []
            evaluated_snippets.sort(key=lambda x: x["score"], reverse=True)
            return evaluated_snippets[:cfg_segment_tot.n_select_sample] 

        # 3. Selection from viable snippets
        viable_snippets.sort(key=lambda x: x["score"], reverse=True) # Sort viable by overall_quality_score
        return viable_snippets[:cfg_segment_tot.n_select_sample]

    def _draft(self, parent_node_being_expanded: Optional[Node] = None) -> Node: 
        log_prefix_draft = f"ToTAgent_DRAFT_Step_{self.current_step}"
        logger.info(f"{log_prefix_draft}: Initiating draft process.")
        
        aide_input_context = self._prepare_aide_input_for_tot(parent_node_being_expanded)
        
        logger.info(f"{log_prefix_draft}: Entering ToT Phase 1 - Master Plan Selection.")
        selected_master_plan_text = self._generate_master_plan_with_tot(aide_input_context)

        if not selected_master_plan_text or "FAILED" in selected_master_plan_text:
            logger.error(f"{log_prefix_draft}: Master Plan generation via ToT failed. Plan: {selected_master_plan_text}")
            return Node(plan=selected_master_plan_text,
                        code="# ToT Master Plan generation failed.",
                        summary="ToT Master Plan generation failed.",
                        parent=parent_node_being_expanded, 
                        is_buggy=True)

        logger.info(f"{log_prefix_draft}: ToT Phase 1 completed. Selected Master Plan:\n{selected_master_plan_text[:500]}...")

        # --- Phase 2: Code Generation from Selected Master Plan (using standard CodeChain logic) ---
        logger.info(f"{log_prefix_draft}: Entering Code Generation Phase from ToT-selected plan.")
       
        task_summary_for_coder = str(aide_input_context.get("task_desc", "No detailed task summary available for coder."))

        try:
            generated_code = self._draft_generate_code_chained(
                task_summary=task_summary_for_coder, 
                master_plan_text=selected_master_plan_text
            )
        except AttributeError as e:
            if "_draft_generate_code_chained" in str(e):
                logger.error(f"{log_prefix_draft}: _draft_generate_code_chained method not found in TOTAgent. Did you copy it from CodeChainAgent or set up inheritance?")
                return Node(plan=selected_master_plan_text,
                            code="# ERROR: _draft_generate_code_chained missing.",
                            summary="Internal error: Code generation method missing.",
                            parent=parent_node_being_expanded,
                            is_buggy=True)
            raise e
        
        if not generated_code or generated_code.strip().startswith("# FAILED TO GENERATE CODE FOR SEGMENT:"):
            logger.error(f"{log_prefix_draft}: Code generation from ToT-selected plan failed or resulted in error placeholders.")

        else:
            logger.info(f"{log_prefix_draft}: Successfully generated code from ToT-selected plan.")

        draft_node = Node(plan=selected_master_plan_text, 
                        code=generated_code,
                        summary="Initial draft: Plan by ToT, Code by Chained Coder.",
                        task_summary=task_summary_for_coder, 
                        parent=parent_node_being_expanded) 
        
        logger.info(f"{log_prefix_draft}: Draft node {draft_node.id} created.")
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
            logger.debug(f"{log_prefix}_LLM_OUTPUT_START\n{code}\n{log_prefix}_LLM_OUTPUT_END", extra={"verbose": True})
            logger.info(f"{log_prefix}: LLM response received, but code extraction failed.", extra={"verbose": True})
            return "", code, ""
        logger.debug(f"{log_prefix}_LLM_OUTPUT_START\n{code}\n{log_prefix}_LLM_OUTPUT_END", extra={"verbose": True})
        logger.info(f"{log_prefix}: LLM response received and code extracted.", extra={"verbose": True})
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
        logger.debug(f"Segment-Generation_LLM_OUTPUT_START\n{completion_text}\nSegment-Generation_LLM_OUTPUT_END", extra={"verbose": True})
        return completion_text.strip() if completion_text else ""

    def _generate_code_segment(self,
                               segment_name: str,
                               task_summary: str,
                               master_plan_text: str,
                               code_accumulator: str,
                               chain_reflection: bool=False,
                               ) -> str:
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
        logger.debug(f"{log_prefix_segment}_SEGMENT_LLM_OUTPUT_START\n{code_snippet.strip()}\n{log_prefix_segment}_SEGMENT_LLM_OUTPUT_END", extra={"verbose": True})
        logger.info(f"{log_prefix_segment}: Code segment generated for {segment_name}.", extra={"verbose": True})
        if chain_reflection:
            logger.info(f"{log_prefix_segment}: Initial snippet generated. Now reflecting.")
            reflection_summary, code_snippet = self._reflect_on_segment(
                task_summary=task_summary,
                master_plan_text=master_plan_text,
                segment_name=segment_name,
                code_before_segment=code_accumulator,
                initial_segment_snippet=code_snippet
            )
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
    



















    # def _draft_generate_code_chained(self, task_summary: str, master_plan_text: str) -> str:
    #     log_prefix_chain = f"CodeChainAgent_Chained_Draft_Step: {self.current_step}"
    #     logger.info(f"Starting chained code generation for draft.")
        
    #     code_accumulator = f"# Script generated by AIDE CodeChainAgent (Chained Coder) - Step {self.current_step}\n"
    #     code_accumulator += f"# Competition: {self.competition_name}\n"
    #     code_accumulator += f"# Task Summary: {task_summary.splitlines()[0]}...\n" 
    #     code_accumulator += "# --- Master Plan ---\n"
    #     for i, plan_step_line in enumerate(master_plan_text.splitlines()):
    #         if plan_step_line.strip() and not plan_step_line.strip().startswith("##"): 
    #             code_accumulator += f"# {plan_step_line.strip()}\n"
    #     code_accumulator += "# --- End Master Plan ---\n\n"


    #     segments_order = [
    #         "Setup & Imports",
    #         "Data Loading",
    #         "Data Preprocessing",
    #         "Modeling",
    #         "Training & Validation", 
    #         "Prediction & Submission"
    #     ]

    #     for segment_name in segments_order:
    #         code_snippet = self._generate_code_segment_with_tot(
    #             segment_name, task_summary, master_plan_text, code_accumulator
    #         )
    #         clean_code_snippet = extract_code(code_snippet) # From aide.utils.response

    #         if not clean_code_snippet.strip() or "# TOT_ERROR" in clean_code_snippet :
    #             logger.warning(f"{log_prefix_chain}: Halting chain for {self.current_step} due to failure or empty snippet in segment: {segment_name}")
    #             code_accumulator += f"\n\n# FAILED TO GENERATE RELIABLE CODE FOR SEGMENT: {segment_name} via ToT\n"
    #             break 

    #         code_accumulator += code_snippet + "\n\n" 
    #         if f"# FAILED TO GENERATE CODE FOR SEGMENT: {segment_name}" in code_snippet:
    #             logger.warning(f"{log_prefix_chain}: Halting chain due to failure in segment: {segment_name}")
    #             break 

    #     logger.info(f"{log_prefix_chain}: Chained code generation process complete.")
    #     return code_accumulator.strip()

    # def _generate_code_segment_with_tot(self, 
    #                                     segment_name: str, 
    #                                     task_summary_for_coder: str, 
    #                                     master_plan_text: str,       
    #                                     code_accumulator: str        
    #                                     ) -> str: 
    #     cfg_segment_tot = self.cfg.agent.tot.segment_coding
    #     log_prefix_seg_tot = f"{self.current_step}_SegToT_{segment_name.replace(' ', '_')}"

    #     if not cfg_segment_tot.enabled:
    #         return self._generate_code_segment( 
    #             segment_name, task_summary_for_coder, master_plan_text, code_accumulator
    #         )

    #     logger.info(f"{log_prefix_seg_tot}: Starting ToT for segment '{segment_name}'.")
    #     aide_context_seg = self._prepare_aide_input_for_tot() 

    #     # 1. Generate k code snippet "thoughts" for this segment
    #     user_prompt_gen_snip = get_tot_generate_segment_code_snippets_user_prompt(
    #         aide_context_seg, master_plan_text, segment_name, code_accumulator, 
    #         cfg_segment_tot.n_generate_sample
    #     )

    #     raw_snippets_text = self._query_llm_with_retries(
    #         query_type=f"TOT_SEG_SNIP_GEN_{segment_name.replace(' ','_')}",
    #         system_prompt=get_tot_segment_coder_system_prompt(), 
    #         user_prompt=user_prompt_gen_snip,
    #         model=self.acfg.code.model, 
    #         temperature=self.acfg.code.temp,
    #         planner_flag=False,
    #         convert_system_to_user=self.acfg.convert_system_to_user, 
    #         retries=3
    #     )
    #     candidate_snippet_strings = self._parse_multiple_code_snippets(raw_snippets_text, cfg_segment_tot.n_generate_sample)

    #     if not candidate_snippet_strings:
    #         logger.error(f"{log_prefix_seg_tot}: Failed to generate/parse any code snippets for segment '{segment_name}'.")
    #         return list(f"# TOT_ERROR: No code snippets generated for segment {segment_name}\n")

    #     # 2. Evaluate these k snippet-thoughts
    #     evaluated_snippets_data = []
    #     for i, snippet_text in enumerate(candidate_snippet_strings):
    #         eval_data = self._evaluate_single_code_snippet_thought( 
    #             aide_context_seg, master_plan_text, segment_name, code_accumulator, snippet_text,
    #             log_prefix_seg_tot, i + 1
    #         )
    #         evaluated_snippets_data.append(eval_data)

    #     viable_snippets_data = [
    #             data for data in evaluated_snippets_data 
    #             if data["evaluation_dict"].get("likely_correct_and_integrates", False) and \
    #             data["evaluation_dict"].get("adheres_to_segment_plan", False) and \
    #             (data["evaluation_dict"].get("snippet_score", 0.0) > 0) 
    #         ]
   
    #     if not viable_snippets_data:
    #         logger.error(f"{log_prefix_seg_tot}: No snippets positively evaluated for segment '{segment_name}'. Returning first generated snippet as fallback or error.") 
    #         return list(candidate_snippet_strings[0]) if candidate_snippet_strings else list(f"# TOT_ERROR: Snippet evaluation failed for segment {segment_name}\n")

    #     # 3. Select the best single snippet (n_select_sample for segments is usually 1)
    #     viable_snippets_data.sort(key=lambda x: x.get("evaluation_details",{}).get("snippet_score", 0.0), reverse=True)
        
    #     chosen_snippet_data = viable_snippets_data[0:self.cfg.agent.tot.segment_coding.n_generate_sample]
    #     logger.info(f"{log_prefix_seg_tot}: Selected snippet for segment '{segment_name}' with score {chosen_snippet_data.get('evaluation_details',{}).get('snippet_score', 'N/A')}.")

    #     return chosen_snippet_data
