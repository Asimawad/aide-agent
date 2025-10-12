# aide/agents/base.py
import shutil
import logging
import random
import json
import time
from pathlib import Path
from rich.console import Console 
from rich.syntax import Syntax 
from typing import Any, Callable, cast, Optional, Dict ,List, Union, Tuple
from aide.backend import query
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils import data_preview
from aide.backend.utils import OutputType, ContextLengthExceededError
from aide.utils.config import Config
from aide.utils.pretty_logging import log_step 
from aide.utils.wandb_logger import WandbLogger
from aide.utils.self_reflection import perform_two_step_reflection
from aide.utils.metric import MetricValue, WorstMetricValue 
from aide.utils.prompt_utils import *
from aide.utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code, 
    trim_long_string,
    extract_reflection_summary_and_revised_code,
    extract_summary_and_plan,
)

logger = logging.getLogger("aide") 
console = Console()

def format_time(time_in_sec: int): # Should be float for more precision
    time_in_sec = int(time_in_sec) # Cast to int if original signature is intended
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

ExecCallbackType = Callable[[str, bool], ExecutionResult]


# Parent class for all agents
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
            from aide.backend import compile_prompt_to_md
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
        console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Search Policy")

        log_prefix_base = f"Search_Policy-Step: {self.current_step}"
        num_drafts = self.acfg.search.num_drafts

        search_cfg = self.acfg.search
        logger.info("[search_policy] Determining next action.", extra={"verbose": True})

    
        if len(self.journal.draft_nodes) < num_drafts:
            logger.info(f"{log_prefix_base}: Selected: Draft new node (drafts: {len(self.journal.draft_nodes)} < {num_drafts}).", extra={"verbose": True})
            return None

        if random.random() < self.acfg.search.debug_prob:
            debuggable_nodes = [
                n for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= self.acfg.search.max_debug_depth)
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
                logger.error(f"Context length exceeded: {cle}. Failing this operation.", extra={"verbose": True})
                return "", f"LLM Query Error: Context Length Exceeded - {str(cle)}", "CONTEXT_LENGTH_EXCEEDED"
            
            except Exception as e: 
                if "ContextLengthExceededError" in str(type(e)) or "context length" in str(e).lower(): # Heuristic check
                    logger.error(f"Context length exceeded: {e}. Failing this operation.", extra={"verbose": True})
                    return "", f"LLM Query Error: Context Length Exceeded - {str(e)}", "CONTEXT_LENGTH_EXCEEDED"
                logger.error(f"Query failed: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: return "", f"LLM Query Error: {e}", "LLM_QUERY_ERROR"
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5)) # Make delay configurable
                continue
            
            if completion_text == "Exceeded context length limit":
                return "", completion_text or "No LLM response received", "EXTRACTION_FAILED"
            code = extract_code(completion_text)            
            nl_text = extract_text_up_to_code(completion_text)
            if code and nl_text:
                logger.info(f"Successfully extracted plan and code.", extra={"verbose": True})
                return nl_text, code, "execution_summary_placeholder"
            logger.warning(f"Plan or code extraction failed. Raw text: '{trim_long_string(completion_text)}'", extra={"verbose": True})
        logger.error(f"All {retries} attempts for plan+code extraction failed.", extra={"verbose": True})
        return "", completion_text or "No LLM response received", "EXTRACTION_FAILED"

    def _query_llm_with_retries(
        self,
        query_type: str, # e.g., "PLANNER_PLAN", "PLANNER_CODER", "Segment-Generation"
        system_prompt: Dict[str, Any],
        user_prompt: Dict[str, Any],
        model: str,
        convert_system_to_user: bool,
        retries: int = 3,
        max_tokens: int = None,
        num_responses: int = 1,
        temperature: float=0.7,
        planner_flag: bool=False, # Number of desired completions
        ) -> Union[OutputType, List[OutputType], None]: # Return type can be single, list, or None on total failure
        
        completion_text = None
        log_prefix = f"" 
        for attempt in range(retries):
            logger.info(f"Generation Attempt {attempt+1}/{retries}: Sending request. Model: {model}, Temp: {temperature}, PlannerFlag: {planner_flag}", extra={"verbose": True})
            try:
                raw_llm_output_from_backend: Union[OutputType, List[OutputType], None] = None
                raw_llm_output_from_backend = query(
                    system_message=system_prompt,
                    user_message=user_prompt,
                    model=model,
                    temperature=temperature,
                    planner=planner_flag,
                    current_step=self.current_step,
                    convert_system_to_user=convert_system_to_user,
                    max_tokens=max_tokens if max_tokens is not None else self.acfg.code.max_new_tokens,
                    num_responses=num_responses,
                )

                if isinstance(raw_llm_output_from_backend, str) and \
                   ("Exceeded context length limit" in raw_llm_output_from_backend or \
                    "CONTEXT_LENGTH_EXCEEDED" in raw_llm_output_from_backend): # Check common error strings
                    logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Backend returned Context Length Exceeded string: {raw_llm_output_from_backend}")

                    raise ContextLengthExceededError(f"CLE from backend: {raw_llm_output_from_backend}")

                if isinstance(raw_llm_output_from_backend, list):

                    for item_idx, item_content in enumerate(raw_llm_output_from_backend):
                        if isinstance(item_content, str) and \
                           ("Exceeded context length limit" in item_content or \
                            "CONTEXT_LENGTH_EXCEEDED" in item_content):
                            logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Item {item_idx} in list from backend signals Context Length Exceeded: {item_content}")
                            raise ContextLengthExceededError(f"CLE in list item from backend: {item_content}")
                
                if num_responses == 1 : 
                    if not isinstance(raw_llm_output_from_backend, (str)) :
                        logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Expected single str/dict from backend (n=1), got {type(raw_llm_output_from_backend)}. Content: {str(raw_llm_output_from_backend)[:200]}")
                        if attempt == retries -1 : return None # Total failure
                        time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
                        continue # Retry

                    single_completion_text = cast(OutputType, raw_llm_output_from_backend)  

                    if query_type == "Segment-Generation":
                            if not isinstance(single_completion_text, str):
                                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Segment-Generation expected string, got {type(single_completion_text)}. Cannot extract code.")
                                if attempt == retries -1: return None
                                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
                                continue # Retry
                            
                            code_snippet = extract_code(single_completion_text)
                            if not code_snippet or not code_snippet.strip():
                                logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}: Segment-Generation - extracted empty code. Raw: '{trim_long_string(single_completion_text)}'. Retrying...")
                                if attempt == retries -1: return "#EMPTY_CODE_SNIPPET_AFTER_RETRIES" # Or None
                                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
                                continue # Retry
                            logger.info(f"{log_prefix}_ATTEMPT{attempt+1}: Segment-Generation successful.", extra={"verbose": True})
                            return code_snippet.strip()
                        
                        # For other query types when n=1, return the single completion
                    logger.info(f"{log_prefix}_ATTEMPT{attempt+1}: Query successful (n=1).", extra={"verbose": True})
                    return single_completion_text


                else: 
                    if not isinstance(raw_llm_output_from_backend, list):
                        logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Expected list from backend (n>1), got {type(raw_llm_output_from_backend)}. Content: {str(raw_llm_output_from_backend)[:200]}")
                        if attempt == retries -1: return None # Total failure
                        time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))
                        continue 

                    logger.info(f"{log_prefix}_ATTEMPT{attempt+1}: Query successful (n={num_responses}). Returning list of {len(raw_llm_output_from_backend)} items.", extra={"verbose": True})
                    return raw_llm_output_from_backend

            except ContextLengthExceededError as cle:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Context Length Exceeded: {cle}. Failing this operation permanently.", exc_info=False)
                return None 
            
            except Exception as e:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Error during LLM query or processing: {e}", exc_info=True)
                if attempt == retries - 1: 
                    logger.error(f"{log_prefix}: All {retries} retries failed for query type {query_type}.")
                    return None 
                
                time.sleep(self.cfg.agent.get("retry_delay_seconds", 5))

        
        logger.error(f"{log_prefix}: All {retries} attempts failed for query type {query_type}. Returning None.")
        return None

    def plan_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3, planner_flag: bool=True) -> tuple[str, str, str]:
        system_prompt = get_planner_agent_plan_system_prompt(); log_prefix = f"Plan_Step: {self.current_step}"

        logger.info(f"{log_prefix}: Sending PLANNER_PLAN query to LLM.", extra={"verbose": True})
        logger.debug(f"{log_prefix}: System prompt: {system_prompt}", extra={"verbose": True})
        logger.debug(f"{log_prefix}: User prompt: {user_prompt_dict}", extra={"verbose": True})
        completion_text = self._query_llm_with_retries(query_type="PLANNER_PLAN", system_prompt=system_prompt, user_prompt=user_prompt_dict,
                                               model=self.acfg.code.planner_model, temperature=self.acfg.code.temp,
                                               convert_system_to_user=self.acfg.convert_system_to_user, retries=retries, planner_flag=planner_flag)
        if completion_text is None: return "", "", ""

        summary, plan = extract_summary_and_plan(completion_text)
        if not (plan and summary): plan = plan or str(completion_text); summary = summary or "SUMMARY_EXTRACTION_FAILED"
        logger.info(f"{log_prefix}: Extracted summary and plan: {summary} \n ------ \n {plan} \n ------ \n END", extra={"verbose": True})
        return summary, plan, " "
    
    def code_query(self, 
                   user_prompt_dict: Dict[str, Any], 
                   retries: int = 3, 
                   num_responses: int = 1, # Add num_responses here
                   temperature: float = 0.7) -> Union[Tuple[str, str, str], List[Tuple[str, str, str]]]: # Return can be single or list of (plan, code, summary)
                                                                                      # For code_query, plan and summary are empty strings.
        system_prompt = get_planner_agent_code_system_prompt() # This system prompt is for generating ONLY code
        log_prefix = f"AGENT_CODE_QUERY_Step:{self.current_step}"
        
        # _query_llm_with_retries will call backend.query with n=num_responses.
        # backend.query will return a single string if num_responses=1, or List[str] if num_responses > 1.
        raw_llm_output = self._query_llm_with_retries(
            query_type="PLANNER_CODER", # Or a more generic "CODE_GENERATION"
            system_prompt=system_prompt, 
            user_prompt=user_prompt_dict,
            temperature=temperature,
            model=self.acfg.code.model, # Use the primary coder model
            planner_flag=False, # It's a coder model call
            convert_system_to_user=self.acfg.convert_system_to_user, 
            retries=retries,
            num_responses=num_responses, # Pass N here
        )

        if raw_llm_output is None: # Indicates total failure in _query_llm_with_retries
            if num_responses > 1:
                return [("", "#LLM_QUERY_RETURNED_NONE", "Query returned None")] * num_responses
            else:
                return "", "#LLM_QUERY_RETURNED_NONE", "Query returned None"

        if isinstance(raw_llm_output, list):
            # We received multiple raw text responses
            extracted_codes_tuples: List[Tuple[str, str, str]] = []
            for text_item in raw_llm_output:
                if not isinstance(text_item, str):
                    logger.warning(f"{log_prefix}: Received non-string item in list from LLM: {type(text_item)}. Skipping.")
                    extracted_codes_tuples.append(("", "#NON_STRING_RESPONSE_ITEM", "Non-string item"))
                    continue
                if text_item.startswith("ERROR:") or text_item == "Exceeded context length limit":
                     logger.warning(f"{log_prefix}: Received error string from LLM: {text_item}.")
                     extracted_codes_tuples.append(("", f"#{text_item.replace(' ','_')}", text_item)) # Make it a valid comment
                     continue

                code = extract_code(text_item)
                if code:
                    logger.info(f"{log_prefix}: Successfully extracted code from one of N responses.", extra={"verbose": True})
                    extracted_codes_tuples.append(("", code, "code_candidate_summary_placeholder"))
                else:
                    print(f"{log_prefix}: Code extraction failed for one of N responses.'")
                    logger.debug(f"{log_prefix}: Code extraction failed for one of N responses. Raw: '{trim_long_string(text_item)}'", extra={"verbose": True})
                    extracted_codes_tuples.append(("", f"#CODE_EXTRACTION_FAILED\n#Raw:\n#{text_item.replace(chr(10), '#')}", "Code extraction failed"))
            return extracted_codes_tuples
        
        elif isinstance(raw_llm_output, str): # Single response (num_responses was likely 1)
            if raw_llm_output.startswith("ERROR:") or raw_llm_output == "Exceeded context length limit":
                logger.warning(f"{log_prefix}: Received error string from LLM: {raw_llm_output}.")
                return "", f"#{raw_llm_output.replace(' ','_')}", raw_llm_output
        
            code = extract_code(raw_llm_output)
            if code:
                logger.info(f"{log_prefix}: Successfully extracted code.", extra={"verbose": True})
                # logger.debug(f"{log_prefix} \n EXTRACTED_CODE_START\n{code}\nEXTRACTED_CODE_END", extra={"verbose": True})
                return "", code, "code_generation_summary_placeholder" # Plan is empty, summary is placeholder
            else:
                logger.warning(f"{log_prefix}: Code extraction failed. Raw: '{trim_long_string(raw_llm_output)}'")
                # Return the raw output as code if extraction fails, prepended with a comment
                return "", f"#CODE_EXTRACTION_FAILED\n#Raw Response:\n#{raw_llm_output.replace(chr(10),'#')}", "Code extraction failed"
        else:
            # Should not happen if _query_llm_with_retries behaves as expected
            logger.error(f"{log_prefix}: Unexpected output type from _query_llm_with_retries: {type(raw_llm_output)}")
            err_placeholder = ("", "#UNEXPECTED_LLM_OUTPUT_TYPE", "Unexpected LLM output type")
            return [err_placeholder] * num_responses if num_responses > 1 else err_placeholder

    def _draft(self, parent_node=None) -> Node:
        log_prefix_base = f"{self.__class__.__name__}_DRAFT_STEP:{self.current_step}" 
        logger.info(f"Starting drafting. Parent: {parent_node.id if parent_node else 'None'}", extra={"verbose": True})
        draft_sys_prompt=get_agent_draft_system_prompt()
        journal_summary=self.journal.generate_summary(include_code=False)
        logger.info(f"Journal summary: {journal_summary}", extra={"verbose": True})

        prompt_user_message = get_agent_draft_user_prompt( 
            task_desc=self.task_desc, journal_summary=journal_summary,
            competition_name=self.competition_name, obfuscate=self.acfg.obfuscate,
            acfg_data_preview=self.acfg.data_preview, data_preview_content=self.data_preview)
        agent_plan_for_step, generated_code, exec_summary = (
            self.plan_and_code_query(user_prompt_dict=prompt_user_message, excute=False,system_prompt_dict = draft_sys_prompt, retries=self.acfg.get('query_retries', 1)))
        if not agent_plan_for_step: agent_plan_for_step = "PLAN_GENERATION_FAILED"
        if not generated_code: generated_code = "# CODE_GENERATION_FAILED"
        logger.debug(f"Draft plan", extra={"verbose": True})
        logger.debug(f"{log_prefix_base}_DRAFT_CODE_RAW_START\n{generated_code}\n{log_prefix_base}_DRAFT_CODE_RAW_END", extra={"verbose": True})
        new_node = Node(plan=agent_plan_for_step, code=generated_code, summary=exec_summary)
        if parent_node: new_node.parent = parent_node
        logger.info(f"Drafted new node {new_node.id}.", extra={"verbose": True})
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
        print(f"code: {wrap_code(result_node.code)}")
        exec_start_time = time.time()
        exec_result = exec_callback(result_node.code, reset_session=True)
        exec_duration = time.time() - exec_start_time
        logger.info(f"AGENT_STEP {current_step_number}: Code execution finished in {exec_duration:.2f}s. Success: {exec_result.term_out}", extra={"verbose": True})
        logger.debug(f"AGENT_STEP {current_step_number}_EXEC_RESULT_STDOUT_START\n{exec_result.term_out}\nAGENT_STEP {current_step_number}_EXEC_RESULT_STDOUT_END", extra={"verbose": True})
        logger.debug(f"AGENT_STEP {current_step_number}_EXEC_RESULT_STDERR_START\n{exec_result.term_out}\nAGENT_STEP{current_step_number}_EXEC_RESULT_STDERR_END", extra={"verbose": True})
        exec_duration = result_node.exec_time or 0

        is_pre_evaluated_by_sc = (
            hasattr(result_node, 'analysis') and result_node.analysis is not None and
            hasattr(result_node, 'metric') and result_node.metric is not None and
            hasattr(result_node, '_term_out') and result_node._term_out is not None and # Ensure _term_out is there
            result_node.exec_time is not None # Ensure exec_time is there
        )



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
        self.exec_callback = exec_callback 
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
            logger.info(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Querying feedback LLM.", extra={"verbose": True})
            logger.debug(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_SYSTEM_PROMPT_START\n{feedback_system_prompt}\n{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_SYSTEM_PROMPT_END", extra={"verbose": True})
            logger.debug(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_FUNC_SPEC_START\n{review_func_spec.to_dict()}\n{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_FUNC_SPEC_END", extra={"verbose": True})

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