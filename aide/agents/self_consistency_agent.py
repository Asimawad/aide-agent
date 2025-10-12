import logging
import time
from rich.console import Console 
from typing import Any, Callable, cast, Optional, Dict ,List, Union, Tuple
from aide.interpreter import ExecutionResult
from aide.journal import Node
from aide.backend.utils import OutputType, ContextLengthExceededError
from aide.utils.metric import WorstMetricValue 
from aide.agents.base import Agent
from aide.backend import query
from aide.utils.prompt_utils import *
from aide.utils.response import (
    extract_code,
    extract_text_up_to_code,
    trim_long_string,
    extract_summary_and_plan,
)

logger = logging.getLogger("aide") 
console = Console()

def format_time(time_in_sec: int): # Should be float for more precision
    time_in_sec = int(time_in_sec) # Cast to int if original signature is intended
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

ExecCallbackType = Callable[[str, bool], ExecutionResult]

class SelfConsistencyAgent(Agent):
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
            retries=3, # Using a potentially specific retry count for planning
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
        N = self.acfg.selfConsistency.num_responses
        print(f"self.acfg.selfConsistency: {self.acfg.selfConsistency}")
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
        logger.info(f"{log_prefix_eval_select}: Evaluating {len(code_candidate_tuples)} code candidates using strategy '{self.acfg.selfConsistency.selection_strategy}'.")

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
        selection_strategy = self.acfg.selfConsistency.selection_strategy

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
        logger.info(f"{log_prefix_eval_select}: Evaluating {len(candidate_plan_code_summary_tuples)} (plan,code) candidate pairs using strategy '{self.acfg.selfConsistency.selection_strategy}'.")

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
        selection_strategy = self.acfg.selfConsistency.selection_strategy

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
        logger.info(f"{log_prefix_draft}: Starting self-consistency draft operation (N={self.acfg.selfConsistency.num_responses}).")


        task_summary, master_plan_text = self._get_master_plan()

        if master_plan_text == "MASTER_PLAN_GENERATION_FAILED_SELFCONSISTENCY":
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
        logger.info(f"{log_prefix_improve}: Starting self-consistency improve for node {parent_node.id} (N={self.acfg.selfConsistency.num_responses}).", extra={"verbose": True})

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
            num_responses=self.acfg.selfConsistency.num_responses,  # Tell it to ask backend for N
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
        
        logger.info(f"{log_prefix_debug}: Starting self-consistency debug for node {parent_node.id} (N={self.acfg.selfConsistency.num_responses}).", extra={"verbose": True})

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
            num_responses=self.acfg.selfConsistency.num_responses
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


