import shutil
import logging
import time
from pathlib import Path
from rich.console import Console 
from typing import Callable, Tuple
from aide.interpreter import ExecutionResult
from aide.journal import Node
from aide.agents.base import Agent
from aide.utils.metric import WorstMetricValue 
from aide.utils.prompt_utils import *
from aide.backend import query
from aide.utils.pretty_logging import log_step 
logger = logging.getLogger("aide") 
console = Console()

def format_time(time_in_sec: int): # Should be float for more precision
    time_in_sec = int(time_in_sec) # Cast to int if original signature is intended
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

ExecCallbackType = Callable[[str, bool], ExecutionResult]


class SelfDebugAgent(Agent): # Inherit from Agent

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
        logger.info(f"Final buggy status", extra={"verbose": True})

        if should_apply_reflection:
            logger.info(f"Node {result_node.id} is buggy after initial {initial_node_stage}. Applying self-reflection.")
            reflection_plan, reflection_code = self.reflect(node=result_node) # self.reflect is from base Agent
            
            if reflection_code and reflection_code.strip() and reflection_code != result_node.code:
                result_node.code = reflection_code
                result_node.plan += f"\n\n--- Self-Reflection Applied ---\n{reflection_plan}" # Append reflection plan
                self.reflection_applied = True
                
                logger.info(f"Re-executing Node {result_node.id} after self-reflection.")
                result_node, reflection_exec_duration = self._execute_and_evaluate_node(
                    result_node, exec_callback, "PostReflection"
                )
                exec_duration += reflection_exec_duration # Add to total exec time for the step
            else:
                logger.info(f"Self-reflection did not result in code changes for Node {result_node.id}.")

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
