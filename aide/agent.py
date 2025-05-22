import shutil
import logging
import random
import time
from rich.syntax import Syntax # Keep for logging if used
from rich.console import Console # Keep for console output
from typing import Any, Callable, cast, Optional, Dict # Added Dict
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview # data_preview.generate
from .utils.config import Config
from .utils.pretty_logging import log_step # logger from pretty_logging might conflict, be careful
# from .utils.metric import MetricValue, WorstMetricValue # Defined in journal or imported there

from .utils.prompt_utils import (
    get_agent_draft_user_prompt,
    get_agent_improve_user_prompt,
    get_agent_debug_user_prompt,
    get_agent_system_prompt,
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
    extract_plan, # For PlannerAgent
    extract_summary, # For PlannerAgent
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
        task_desc: str, # This is a dict or string. Prompt utils expect string.
        cfg: Config,
        journal: Journal,
        wandb_run=None,
        competition_benchmarks=None,
    ):
        self.task_desc_orig = task_desc # Store original, can be dict or str
        if isinstance(task_desc, dict):
            from .backend import compile_prompt_to_md # Local import
            self.task_desc = compile_prompt_to_md(task_desc)
        else:
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
        self._code_quality: float = 0.0 # Initialize for parse_exec_result

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search
        # console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Search Policy")
        # This console rule is now typically logged by the main run loop or step method
        logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Determining next action.", extra={"verbose": True})

        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            # console.print(f"[bold yellow]Drafting new node (drafts: {len(self.journal.draft_nodes)})[/bold yellow]")
            logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Draft new node (drafts: {len(self.journal.draft_nodes)} < {search_cfg.num_drafts}).", extra={"verbose": True})
            return None

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                n for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                # console.print(f"[bold red]Debugging node {node_to_debug.id} (debug depth: {node_to_debug.debug_depth})[/bold red]")
                logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Debug node {node_to_debug.id} (debug_prob triggered, depth {node_to_debug.debug_depth}).", extra={"verbose": True})
                return node_to_debug
            else:
                logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Attempted debug (debug_prob triggered), but no debuggable nodes found.", extra={"verbose": True})


        good_nodes = self.journal.good_nodes
        if not good_nodes:
            # console.print("[bold yellow]Drafting new node (no good nodes)[/bold yellow]")
            logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Draft new node (no good nodes to improve).", extra={"verbose": True})
            return None

        greedy_node = self.journal.get_best_node() # get_best_node already implies only_good=True by default in its typical use.
        if greedy_node: # Should always exist if good_nodes is not empty
            if greedy_node.is_buggy: # This case might be rare if get_best_node filters buggy ones unless only_good=False
                # console.print(f"[bold red]Debugging node {greedy_node.id} (buggy)[/bold red]")
                logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Debug greedy node {greedy_node.id} (it was marked buggy).", extra={"verbose": True})
                return greedy_node # Debug the best node if it's buggy
            # console.print(f"[bold green]Improving node {greedy_node.id} (metric: {greedy_node.metric.value:.3f})[/bold green]")
            logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Improve greedy node {greedy_node.id} (metric: {greedy_node.metric.value:.3f if greedy_node.metric else 'N/A'}).", extra={"verbose": True})
            return greedy_node
        else: # Fallback, should ideally not be reached if good_nodes exist
            logger.info(f"AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Draft new node (no best_node found, fallback).", extra={"verbose": True})
            return None


    # REMOVE: _prompt_environment, _prompt_impl_guideline, _prompt_resp_fmt
    # These are now handled by functions in prompt_utils.py

    def plan_and_code_query(self, user_prompt_dict: Dict[str, Any], excute: bool, retries: int = 1) -> tuple[str, str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        # `excute` param seems unused by `query` directly, but kept for signature consistency if used elsewhere.
        # `retries` here defines how many times this whole method retries, distinct from backend retries.
        
        system_prompt = get_agent_system_prompt() # Get system prompt from utils
        log_prefix = f"AGENT_PLAN_CODE_QUERY_STEP{self.current_step}"

        completion_text = None
        
        for attempt in range(retries):
            logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Sending request.", extra={"verbose": True})
            # Example: logger.debug(f"{log_prefix}_SYSTEM_PROMPT: {system_prompt}", extra={"verbose": True})


            try:
                completion_text = query(
                    system_message=system_prompt,
                    user_message=user_prompt_dict,
                    model=self.acfg.code.model,
                    temperature=self.acfg.code.temp,
                    max_tokens=self.acfg.code.max_new_tokens, # Renamed from max_new_tokens for clarity with backend
                    current_step=self.current_step,
                    inference_engine=self.cfg.inference_engine,
                    num_responses=self.acfg.code.num_return_sequences, # Passed as **model_kwargs if backend supports it
                    convert_system_to_user=self.acfg.convert_system_to_user,
                    # **{'num_return_sequences': self.acfg.code.num_return_sequences} # Alternative way to pass
                )
            except Exception as e:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Query failed: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1: # Last attempt
                    return "", f"LLM Query Error: {e}", "LLM_QUERY_ERROR"
                time.sleep(2) # Simple backoff before retrying this method's loop
                continue


            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text) # This is the "PLAN:" part

            if code and nl_text:
                logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Successfully extracted plan and code.", extra={"verbose": True})
                return nl_text, code, "execution_summary_placeholder" # Placeholder, actual summary from feedback

            logger.warning(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Plan or code extraction failed. Raw text: '{trim_long_string(completion_text)}'", extra={"verbose": True})
            logger.debug(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}_EXTRACTION_FAILED_RAW_COMPLETION_START\n{completion_text}\n{log_prefix}_EXTRACTION_FAILED_RAW_COMPLETION_END", extra={"verbose": True})
        
        logger.error(f"{log_prefix}: All {retries} attempts for plan+code extraction failed.", extra={"verbose": True})
        return "", completion_text or "No LLM response received", "EXTRACTION_FAILED"


    def _draft(self, parent_node=None) -> Node:
        # console.rule(f"[cyan]Agent Step {self.current_step} - Stage : Drafting") # Handled by main step log
        log_prefix = f"AGENT_DRAFT_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting drafting process. Parent: {parent_node.id if parent_node else 'None'}", extra={"verbose": True})
        
        # comp_data = self.competition_benchmarks # Unused in original prompt, kept for reference

        prompt_user_message = get_agent_draft_user_prompt(
            task_desc=self.task_desc,
            journal_summary=self.journal.generate_summary(include_code=False), # Typically don't include code in memory for draft
            competition_name=self.competition_name,
            obfuscate=self.acfg.obfuscate,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )

        agent_plan_for_step, generated_code, execution_summary_placeholder = (
            self.plan_and_code_query(prompt_user_message, excute=False, retries=self.acfg.get('query_retries', 1)) # Assuming 1 retry default
        )

        # Logging of plan and code
        if agent_plan_for_step:
            logger.debug(f"{log_prefix}_DRAFT_PLAN_START\n{agent_plan_for_step}\n{log_prefix}_DRAFT_PLAN_END", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix}: Plan generation failed or was empty.", extra={"verbose": True})
            agent_plan_for_step = "PLAN GENERATION FAILED" # Placeholder

        formatted_extracted_code = format_code(generated_code) if generated_code else ""
        if formatted_extracted_code:
            # console.print(f"[bold green]Extracted a valid Code for step {self.current_step}[/bold green]") # Console output
            logger.info(f"{log_prefix}: Code generated for drafting stage.", extra={"verbose": True})
            # logger.debug(f"{log_prefix}_DRAFT_CODE_SYNTAXHIGHLIGHT_START\n{Syntax(formatted_extracted_code, 'python', theme='default', line_numbers=True)}\n{log_prefix}_DRAFT_CODE_SYNTAXHIGHLIGHT_END", extra={"verbose": True})
            logger.debug(f"{log_prefix}_DRAFT_CODE_RAW_START\n{formatted_extracted_code}\n{log_prefix}_DRAFT_CODE_RAW_END", extra={"verbose": True})
            # console.print("-" * 60) # Console output
        else:
            logger.warning(f"{log_prefix}: Code generation failed or was empty. Using placeholder.", extra={"verbose": True})
            generated_code = "# CODE GENERATION FAILED" # Placeholder

        new_node = Node(
            plan=agent_plan_for_step,
            code=generated_code, # Store raw generated code before formatting
            summary=execution_summary_placeholder, # This is just a placeholder string
        )
        if parent_node:
            new_node.parent = parent_node

        logger.info(f"{log_prefix}: Drafted new node {new_node.id}.", extra={"verbose": True})
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        # console.rule(f"[cyan]Stage : Improving") # Handled by main step log
        log_prefix = f"AGENT_IMPROVE_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting improvement process for node {parent_node.id}.", extra={"verbose": True})

        prompt_user_message = get_agent_improve_user_prompt(
            task_desc=self.task_desc,
            journal_summary=self.journal.generate_summary(include_code=False), # Usually code not needed for high-level summary
            competition_name=self.competition_name,
            parent_node_code=parent_node.code,
        )

        plan, code, _ = self.plan_and_code_query(prompt_user_message, excute=False, retries=self.acfg.get('query_retries', 1))

        if not plan:
            logger.warning(f"{log_prefix}: Improvement plan generation failed for node {parent_node.id}. Using placeholder.", extra={"verbose": True})
            plan = "IMPROVEMENT PLAN GENERATION FAILED"
        if not code:
            logger.warning(f"{log_prefix}: Improvement code generation failed for node {parent_node.id}. Reverting to parent code.", extra={"verbose": True})
            code = parent_node.code # Revert to parent's code if generation fails

        new_node = Node(plan=plan, code=code, parent=parent_node)
        
        logger.info(f"{log_prefix}: Improvement plan for node {parent_node.id}: {trim_long_string(plan)}", extra={"verbose": True})
        logger.debug(f"{log_prefix}_IMPROVE_PLAN_START\n{plan}\n{log_prefix}_IMPROVE_PLAN_END", extra={"verbose": True})
        logger.info(f"{log_prefix}: Improved node {parent_node.id} to create new node {new_node.id}.", extra={"verbose": True})
        # logger.debug(f"{log_prefix}_IMPROVED_CODE_START\n{wrap_code(code)}\n{log_prefix}_IMPROVED_CODE_END", extra={"verbose": True}) # wrap_code for logging
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        # console.rule(f"[cyan]Stage : Debugging") # Handled by main step log
        log_prefix = f"AGENT_DEBUG_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting debugging process for node {parent_node.id}.", extra={"verbose": True})
        logger.debug(f"{log_prefix}_PARENT_CODE_START\n{parent_node.code}\n{log_prefix}_PARENT_CODE_END", extra={"verbose": True})
        logger.debug(f"{log_prefix}_PARENT_TERM_OUT_START\n{parent_node.term_out}\n{log_prefix}_PARENT_TERM_OUT_END", extra={"verbose": True})


        prompt_user_message = get_agent_debug_user_prompt(
            task_desc=self.task_desc,
            competition_name=self.competition_name,
            parent_node_code=parent_node.code,
            parent_node_term_out=parent_node.term_out, # term_out on Node is already trimmed
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )

        plan, code, _ = self.plan_and_code_query(prompt_user_message, excute=False, retries=self.acfg.get('query_retries', 1))

        if not plan:
            logger.warning(f"{log_prefix}: Debug plan generation failed for node {parent_node.id}. Using placeholder.", extra={"verbose": True})
            plan = "DEBUG PLAN GENERATION FAILED"
        if not code:
            logger.warning(f"{log_prefix}: Debug code generation failed for node {parent_node.id}. Reverting to parent code.", extra={"verbose": True})
            code = parent_node.code # Revert to parent's code

        new_node = Node(plan=plan, code=code, parent=parent_node)

        logger.info(f"{log_prefix}: Debugged node {parent_node.id} to create new node {new_node.id}", extra={"verbose": True})
        logger.debug(f"{log_prefix}_DEBUG_PLAN_START\n{plan}\n{log_prefix}_DEBUG_PLAN_END", extra={"verbose": True})
        # logger.debug(f"{log_prefix}_DEBUG_CODE_START\n{wrap_code(code)}\n{log_prefix}_DEBUG_CODE_END", extra={"verbose": True})
        return new_node

    def reflect(self, node: Node) -> tuple[str, str]:
        """
        Performs a two-step self-reflection using the external utility function.
        Returns: Tuple: (reflection_plan, revised_code)
        """
        log_prefix = f"AGENT_REFLECT_STEP{self.current_step}_NODE{node.id}"
        logger.info(f"{log_prefix}: Initiating self-reflection.", extra={"verbose": True})
        # Detailed logging of inputs to perform_two_step_reflection can be added if needed.

        try:
            reflection_plan, revised_code = perform_two_step_reflection(
                code=node.code,
                analysis=node.analysis, # Textual summary from feedback LLM
                term_out=node.term_out, # Already trimmed string from Node property
                task_desc=self.task_desc,
                model_name=self.acfg.code.model, # Use the main coder model for reflection edits
                temperature=self.acfg.code.temp, # Use coder temp
                convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query,
                wrap_code_func=prompt_utils_wrap_code, # Use the one from prompt_utils
                extract_code_func=extract_code,
                current_step=self.current_step # Pass current_step for logging within reflection
            )
        except Exception as e:
            logger.error(f"{log_prefix}: Error during self-reflection call: {e}", exc_info=True, extra={"verbose": True})
            return f"REFLECTION_ERROR: {e}", node.code # Return original code on error


        if revised_code and revised_code.strip() and revised_code != node.code:
            logger.info(f"{log_prefix}: Self-reflection resulted in code changes.", extra={"verbose": True})
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info(f"{log_prefix}: Self-reflection found no errors requiring changes.", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix}: Self-reflection finished, but revised code is same as original or empty. Plan: {trim_long_string(reflection_plan)}", extra={"verbose": True})
        
        logger.debug(f"{log_prefix}_REFLECTION_PLAN_START\n{reflection_plan}\n{log_prefix}_REFLECTION_PLAN_END", extra={"verbose": True})
        # logger.debug(f"{log_prefix}_REVISED_CODE_BY_REFLECTION_START\n{wrap_code(revised_code)}\n{log_prefix}_REVISED_CODE_BY_REFLECTION_END", extra={"verbose": True})
        return reflection_plan, revised_code


    def double_reflect(self, code: str) -> tuple[str, str]:
        """
        Performs a two-step self-reflection using the external utility function.
        This version doesn't have `analysis` or `term_out` from a node.
        Returns: Tuple: (reflection_plan, revised_code)
        """
        log_prefix = f"AGENT_DOUBLE_REFLECT_STEP{self.current_step}" # No node ID here
        logger.info(f"{log_prefix}: Initiating self-reflection (double_reflect variant).", extra={"verbose": True})

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
            logger.error(f"{log_prefix}: Error during double_reflect call: {e}", exc_info=True, extra={"verbose": True})
            return f"DOUBLE_REFLECTION_ERROR: {e}", code


        if revised_code and revised_code.strip() and revised_code != code:
            logger.info(f"{log_prefix}: Self-reflection (double_reflect) resulted in code changes.", extra={"verbose": True})
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info(f"{log_prefix}: Self-reflection (double_reflect) found no errors requiring changes.", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix}: Self-reflection (double_reflect) finished, but revised code is same as original or empty. Plan: {trim_long_string(reflection_plan)}", extra={"verbose": True})

        logger.debug(f"{log_prefix}_REFLECTION_PLAN_START\n{reflection_plan}\n{log_prefix}_REFLECTION_PLAN_END", extra={"verbose": True})
        # logger.debug(f"{log_prefix}_REVISED_CODE_BY_DOUBLE_REFLECTION_START\n{wrap_code(revised_code)}\n{log_prefix}_REVISED_CODE_BY_DOUBLE_REFLECTION_END", extra={"verbose": True})
        return reflection_plan, revised_code


    def update_data_preview(self):
        log_prefix = f"AGENT_DATA_PREVIEW_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Updating data preview.", extra={"verbose": True})
        try:
            self.data_preview = data_preview.generate(self.cfg.workspace_dir / "input") # Generate from 'input' subdir
            logger.info(f"{log_prefix}: Data preview updated.", extra={"verbose": True})
            logger.debug(f"{log_prefix}_DATA_PREVIEW_CONTENT_START\n{self.data_preview}\n{log_prefix}_DATA_PREVIEW_CONTENT_END", extra={"verbose": True})
        except Exception as e:
            logger.error(f"{log_prefix}: Failed to update data preview: {e}", exc_info=True, extra={"verbose": True})
            self.data_preview = "Error generating data preview."


    def step(self, exec_callback: ExecCallbackType, current_step_number: int):
        log_prefix_main = f"AGENT_STEP{current_step_number}"
        logger.info(f"{log_prefix_main}_START: Total Steps Configured: {self.acfg.steps}", extra={"verbose": True})
        t_step_start = time.time()

        submission_dir = self.cfg.workspace_dir / "submission"
        logger.info(f"{log_prefix_main}: Clearing submission directory: {submission_dir}", extra={"verbose": True})
        shutil.rmtree(submission_dir, ignore_errors=True)
        submission_dir.mkdir(exist_ok=True)

        self.current_step = current_step_number

        if not self.journal.nodes or self.data_preview is None: # Update preview if first step or not yet generated
            self.update_data_preview()

        parent_node = self.search_policy()
        result_node: Node

        draft_flag = False
        node_stage = "unknown" # Initialize
        if parent_node is None:
            draft_flag = True
            node_stage = "draft"
            logger.info(f"{log_prefix_main}: Stage selected: DRAFTING.", extra={"verbose": True})
            result_node = self._draft(parent_node) # parent_node is None here
        elif parent_node.is_buggy:
            node_stage = "debug"
            logger.info(f"{log_prefix_main}: Stage selected: DEBUGGING node {parent_node.id}.", extra={"verbose": True})
            result_node = self._debug(parent_node)
        else:
            node_stage = "improve"
            logger.info(f"{log_prefix_main}: Stage selected: IMPROVING node {parent_node.id}.", extra={"verbose": True})
            result_node = self._improve(parent_node)

        logger.info(f"{log_prefix_main}: Executing code for node {result_node.id} (stage: {node_stage}). Code length: {len(result_node.code)}", extra={"verbose": True})
        logger.debug(f"{log_prefix_main}_CODE_TO_EXECUTE_NODE_{result_node.id}_START\n{result_node.code}\n{log_prefix_main}_CODE_TO_EXECUTE_NODE_{result_node.id}_END", extra={"verbose": True})

        exec_start_time = time.time()
        exec_result = exec_callback(result_node.code, reset_session=True)
        exec_duration = time.time() - exec_start_time
        
        logger.info(f"{log_prefix_main}: Code execution for node {result_node.id} finished in {exec_duration:.2f}s.", extra={"verbose": True})
        logger.debug(f"{log_prefix_main}_EXEC_RESULT_NODE_{result_node.id}_TERM_OUT_START\n{exec_result.term_out}\n{log_prefix_main}_EXEC_RESULT_NODE_{result_node.id}_TERM_OUT_END", extra={"verbose": True})
        if exec_result.exc_type:
             logger.warning(f"{log_prefix_main}_EXEC_RESULT_NODE_{result_node.id}_EXCEPTION: {exec_result.exc_type}", extra={"verbose": True})


        logger.info(f"{log_prefix_main}: Parsing execution results for node {result_node.id}.", extra={"verbose": True})
        result_node = self.parse_exec_result(node=result_node, exec_result=exec_result)
        
        # Store current buggy status before reflection potentially changes it
        buggy_status_before_reflection = result_node.is_buggy

        reflection_applied = False
        if draft_flag and self.acfg.ITS_Strategy == "self-reflection" and result_node.is_buggy:
            logger.info(f"{log_prefix_main}: Condition met for self-reflection on drafted buggy node {result_node.id}.", extra={"verbose": True})
            # console.rule(f"[cyan]Stage : Self Reflection") # For interactive console
            
            reflection_plan, reflection_code = self.reflect(node=result_node)
            if reflection_code and reflection_code.strip() and reflection_code != result_node.code:
                logger.info(f"{log_prefix_main}: Self-reflection yielded new code for node {result_node.id}. Re-executing.", extra={"verbose": True})
                result_node.code = reflection_code # Update node's code
                reflection_applied = True

                logger.info(f"{log_prefix_main}: Re-executing reflected code for node {result_node.id}. Code length: {len(result_node.code)}", extra={"verbose": True})
                logger.debug(f"{log_prefix_main}_REFLECTED_CODE_TO_EXECUTE_NODE_{result_node.id}_START\n{result_node.code}\n{log_prefix_main}_REFLECTED_CODE_TO_EXECUTE_NODE_{result_node.id}_END", extra={"verbose": True})
                
                exec_start_time_reflect = time.time()
                exec_result_reflect = exec_callback(result_node.code, reset_session=True)
                exec_duration = time.time() - exec_start_time_reflect # Update exec_duration to reflected code's time
                
                logger.info(f"{log_prefix_main}: Reflected code execution for node {result_node.id} finished in {exec_duration:.2f}s.", extra={"verbose": True})
                logger.debug(f"{log_prefix_main}_REFLECTED_EXEC_RESULT_NODE_{result_node.id}_TERM_OUT_START\n{exec_result_reflect.term_out}\n{log_prefix_main}_REFLECTED_EXEC_RESULT_NODE_{result_node.id}_TERM_OUT_END", extra={"verbose": True})
                if exec_result_reflect.exc_type:
                    logger.warning(f"{log_prefix_main}_REFLECTED_EXEC_RESULT_NODE_{result_node.id}_EXCEPTION: {exec_result_reflect.exc_type}", extra={"verbose": True})

                logger.info(f"{log_prefix_main}: Parsing execution results for reflected code of node {result_node.id}.", extra={"verbose": True})
                result_node = self.parse_exec_result(node=result_node, exec_result=exec_result_reflect)
            else:
                logger.info(f"{log_prefix_main}: Self-reflection did not result in applicable code changes for node {result_node.id}.", extra={"verbose": True})
        
        # Determine effectiveness based on buggy status *before* reflection and *after* (potentially reflected) execution
        if buggy_status_before_reflection and not result_node.is_buggy:
            result_node.effective_debug_step = True # The step (overall) fixed a bug
            result_node.effective_reflections = reflection_applied # If reflection was the part that fixed it
        else:
            result_node.effective_debug_step = False
            result_node.effective_reflections = False
        
        self._prev_buggy = result_node.is_buggy # Update for the next step's logic

        # --- Logging to W&B and Journal ---
        logger.info(f"{log_prefix_main}: Preparing step log data for W&B.", extra={"verbose": True})
        step_log_data = {
            f"exec/exec_time_s": exec_duration,
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name": self.competition_name,
            "exec/exception_type": result_node.exc_type if result_node.exc_type else "None", # Ensure string
            f"code/estimated_quality": int(self._code_quality), # From parse_exec_result
            f"eval/reflection_applied_successfully": 1 if reflection_applied and not result_node.is_buggy else 0,
            f"eval/effective_fix_this_step": 1 if result_node.effective_debug_step else 0,
        }

        # Metric logging and benchmark comparison
        # (Your existing detailed W&B logging for metrics, thresholds, and plots - kept as is)
        # Small adjustment: Ensure self.competition_benchmarks is checked before use
        agent_validation_metrics_defined = False
        if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None:
            step_log_data[f"eval/validation_metric"] = result_node.metric.value
            agent_validation_metrics_defined = True
            if self.competition_benchmarks and wandb and self.wandb_run:
                # ... (Your existing medal and threshold plot logic using wandb.Table and wandb.plot) ...
                # This part is complex and W&B specific, so I'm assuming it's correct.
                # Key is to ensure self.competition_benchmarks is not None.
                for threshold_name, key_suffix in [
                    ("median_threshold", "above_median"), ("gold_threshold", "gold_medal"),
                    ("silver_threshold", "silver_medal"), ("bronze_threshold", "bronze_medal")
                ]:
                    flag_attr = f"_{key_suffix}_flags"
                    if not hasattr(self, flag_attr): setattr(self, flag_attr, [])
                    
                    threshold_value = self.competition_benchmarks.get(threshold_name, float('inf'))
                    is_met = 1 if result_node.metric.value > threshold_value else 0
                    getattr(self, flag_attr).append(is_met)

                    true_count = sum(getattr(self, flag_attr))
                    false_count = len(getattr(self, flag_attr)) - true_count
                    table = wandb.Table(
                        data=[[key_suffix.replace('_', ' ').title(), true_count], [f"Not {key_suffix.replace('_', ' ').title()}", false_count]],
                        columns=["label", "count"]
                    )
                    step_log_data[f"plots/{key_suffix}_bar"] = wandb.plot.bar(
                        table, "label", "count", title=f"{key_suffix.replace('_', ' ').title()} Steps"
                    )
        else:
            step_log_data[f"eval/validation_metric"] = float("nan")

        # Submission check
        submission_path = submission_dir / "submission.csv"
        submission_exists = submission_path.exists()
        if not result_node.is_buggy and not submission_exists:
            logger.warning(f"{log_prefix_main}: Node {result_node.id} was not buggy BUT submission.csv MISSING. Marking as buggy.", extra={"verbose": True})
            result_node.is_buggy = True
            result_node.metric = WorstMetricValue()
            step_log_data[f"eval/validation_metric"] = float("nan")
            step_log_data[f"eval/is_buggy"] = 1 # Ensure this is updated
            # Potentially remove from _metric_hist if added under false pretenses
            if agent_validation_metrics_defined and hasattr(self, "_metric_hist") and self._metric_hist and \
               result_node.metric.original_value_before_reset_to_worst is not None and \
               self._metric_hist[-1] == result_node.metric.original_value_before_reset_to_worst:
                self._metric_hist.pop()


        step_log_data[f"eval/submission_produced"] = 1 if submission_exists else 0

        # Metric histogram
        if not hasattr(self, "_metric_hist"): self._metric_hist = []
        if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None:
            self._metric_hist.append(result_node.metric.value)
        
        if wandb and self.wandb_run:
            if len(self._metric_hist) >= 1:
                metric_table_data = [[v] for v in self._metric_hist if isinstance(v, (int, float))]
                if metric_table_data:
                    tbl = wandb.Table(data=metric_table_data, columns=["val"])
                    step_log_data["plots/val_metric_scatter"] = wandb.plot.scatter(tbl, "val", "val", title="Validation Metric Values")
            
            # Bug vs Clean plot
            if not hasattr(self, "_bug_flags"): self._bug_flags = []
            self._bug_flags.append(1 if result_node.is_buggy else 0)
            bug_count = sum(self._bug_flags); clean_count = len(self._bug_flags) - bug_count
            bug_table = wandb.Table(data=[["Buggy", bug_count], ["Clean", clean_count]], columns=["label", "count"])
            step_log_data["plots/bug_vs_clean"] = wandb.plot.bar(bug_table, "label", "count", title="Buggy vs Clean Steps")

            # Submission presence plot
            if not hasattr(self, "_sub_flags"): self._sub_flags = []
            self._sub_flags.append(1 if submission_exists else 0)
            with_sub = sum(self._sub_flags); without_sub = len(self._sub_flags) - with_sub
            sub_table = wandb.Table(data=[["Has submission", with_sub], ["No submission", without_sub]], columns=["label", "count"])
            step_log_data["plots/submission_presence"] = wandb.plot.bar(sub_table, "label", "count", title="Submission Produced vs Missing")

        if self.wandb_run:
            logger.info(f"{log_prefix_main}: Logging data to W&B. Keys: {list(step_log_data.keys())}", extra={"verbose": True})
            try:
                self.wandb_run.log(step_log_data, step=current_step_number)
            except Exception as e_wandb:
                logger.error(f"{log_prefix_main}: Error logging to W&B: {e_wandb}", exc_info=True, extra={"verbose": True})
        
        result_node.stage = node_stage # Set stage on the node for journal
        result_node.exec_time = exec_duration # Set final exec time

        self.journal.append(result_node)
        logger.info(f"{log_prefix_main}: Appended node {result_node.id} to journal. Journal size: {len(self.journal.nodes)}", extra={"verbose": True})

        # Cache best solution
        best_node = self.journal.get_best_node() # Default only_good=True
        if best_node and best_node.id == result_node.id : # If current node is the new best
            logger.info(f"{log_prefix_main}: Node {result_node.id} is new best (Metric: {best_node.metric.value if best_node.metric else 'N/A':.4f}). Caching solution.", extra={"verbose": True})
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_submission_dir = self.cfg.workspace_dir / "best_submission" # Not used directly here but good to know
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            # best_submission_dir.mkdir(exist_ok=True, parents=True) # Ensure it exists if used

            if submission_exists: # Check if current node produced submission
                 shutil.copy(submission_path, best_solution_dir / "submission.csv") # Store with best code
            
            with open(best_solution_dir / "solution.py", "w") as f: f.write(result_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f: f.write(str(result_node.id))
        elif best_node:
            logger.info(f"{log_prefix_main}: Current best node is {best_node.id} (Metric: {best_node.metric.value if best_node.metric else 'N/A':.4f})", extra={"verbose": True})


        log_step( # Console logging via pretty_logging
            step=current_step_number,
            total=self.acfg.steps,
            stage=node_stage,
            is_buggy=result_node.is_buggy,
            exec_time=exec_duration,
            metric=(result_node.metric.value if result_node.metric and result_node.metric.value is not None else None),
        )
        t_step_end = time.time()
        logger.info(f"{log_prefix_main}_END: Duration: {t_step_end - t_step_start:.2f}s", extra={"verbose": True})


    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        log_prefix = f"AGENT_PARSE_EXEC_STEP{self.current_step}_NODE{node.id}"
        logger.info(f"{log_prefix}: Parsing execution result.", extra={"verbose": True})

        node.absorb_exec_result(exec_result) # Populates node.term_out, exc_type etc.
        logger.debug(f"{log_prefix}_ABSORBED_NODE_EXC_TYPE: {node.exc_type}", extra={"verbose": True})
        # node.term_out is already a trimmed string property

        introduction = ("You are a Kaggle grandmaster ... evaluate the output ... empirical findings.") # Truncated for brevity
        if self.acfg.obfuscate:
            introduction = ("You are an expert machine learning engineer ... evaluate the output ... empirical findings.")

        # Prompt for feedback LLM
        feedback_system_prompt = {
            "Introduction": introduction,
            "Task Description": self.task_desc,
            "Code Executed": prompt_utils_wrap_code(node.code), # Use alias to avoid conflict
            "Execution Output Log": prompt_utils_wrap_code(node.term_out, lang=""), # term_out is already a string
        }

        max_retries = self.acfg.feedback.get("retries", 3) # Get retries from config or default
        review_response_dict: Optional[Dict[str, Any]] = None

        for attempt in range(max_retries):
            logger.info(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}/{max_retries}: Querying feedback LLM.", extra={"verbose": True})
            # Verbose logging of feedback_system_prompt and func_spec can be added here if needed.
            
            try:
                raw_response = query(
                    system_message=feedback_system_prompt,
                    user_message=None, # User message is None for function calling with system prompt
                    func_spec=review_func_spec, # Defined globally in agent.py
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    convert_system_to_user=self.acfg.convert_system_to_user, # Pass this through
                    current_step=self.current_step # Pass current_step for backend logging
                    # `excute` param is not used by backend query when func_spec is present
                )
                logger.info(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}/{max_retries}: Received response.", extra={"verbose": True})
                logger.debug(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_RAW_RESPONSE_START\n{raw_response}\n{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_RAW_RESPONSE_END", extra={"verbose": True})

                if not isinstance(raw_response, dict):
                    logger.error(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Response is not a dict as expected. Type: {type(raw_response)}", extra={"verbose": True})
                    # Attempt to parse if string that looks like JSON
                    if isinstance(raw_response, str):
                        try:
                            parsed_raw_response = json.loads(raw_response)
                            if isinstance(parsed_raw_response, dict):
                                raw_response = parsed_raw_response
                                logger.info(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Successfully parsed string response to dict.", extra={"verbose": True})
                            else:
                                raise ValueError("Parsed JSON is not a dict.")
                        except Exception as json_e:
                            logger.error(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Failed to parse string response as JSON: {json_e}", extra={"verbose": True})
                            raw_response = None # Signal failure to parse
                    else:
                        raw_response = None # Signal failure if not dict or parsable string

                review_response_dict = cast(Dict[str, Any], raw_response) if isinstance(raw_response, dict) else None
                
                # Validate response
                if review_response_dict and all(k in review_response_dict for k in review_func_spec.json_schema["required"]):
                    logger.info(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Successfully received and validated feedback response.", extra={"verbose": True})
                    break # Success
                else:
                    logger.warning(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Feedback response missing required keys or is None. Response: {review_response_dict}", extra={"verbose": True})
                    review_response_dict = None # Force retry

            except Exception as e:
                logger.error(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Error querying feedback LLM: {e}", exc_info=True, extra={"verbose": True})
            
            if attempt == max_retries - 1 and review_response_dict is None: # All retries failed
                logger.error(f"{log_prefix}: Feedback LLM query failed after {max_retries} retries. Using default error values.", extra={"verbose": True})
                review_response_dict = {
                    "is_bug": True, "has_csv_submission": False,
                    "summary": "Failed to get feedback from LLM after multiple retries.",
                    "metric": None, "lower_is_better": True, "code_quality": 0,
                }
                break # Exit loop

        # Ensure review_response_dict is not None (should have default from above if all retries failed)
        if review_response_dict is None: # Safety net, should ideally not be hit
            review_response_dict = {
                "is_bug": True, "has_csv_submission": False,
                "summary": "CRITICAL_ERROR: review_response_dict became None unexpectedly.",
                "metric": None, "lower_is_better": True, "code_quality": 0,
            }

        metric_value = review_response_dict.get("metric")
        if not isinstance(metric_value, (float, int)):
            if metric_value is not None:
                logger.warning(f"{log_prefix}: Metric value from LLM ('{metric_value}') is not float/int. Setting to None.", extra={"verbose": True})
            metric_value = None
        
        self._code_quality = review_response_dict.get("code_quality", 0) # Store for main step logging
        if not isinstance(self._code_quality, (int, float)): # Ensure numeric
            logger.warning(f"{log_prefix}: Code quality from LLM ('{self._code_quality}') not int/float. Setting to 0.", extra={"verbose": True})
            self._code_quality = 0
        node.code_quality = int(self._code_quality) # Store on node

        # Submission check (actual vs. reported)
        submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
        has_csv_submission_actual = submission_path.exists()
        has_csv_submission_reported = review_response_dict.get("has_csv_submission", False)

        node.analysis = review_response_dict.get("summary", "Feedback LLM summary missing or failed.")
        logger.debug(f"{log_prefix}_LLM_ANALYSIS_SUMMARY_START\n{node.analysis}\n{log_prefix}_LLM_ANALYSIS_SUMMARY_END", extra={"verbose": True})

        # Determine final buggy status
        is_bug_llm = review_response_dict.get("is_bug", True) # Default to buggy if LLM fails to provide
        exc_type_exists = node.exc_type is not None
        metric_missing = metric_value is None
        # If LLM says CSV missing OR actual file is missing, consider it a failure for CSV
        csv_submission_failed = not has_csv_submission_reported or not has_csv_submission_actual

        node.is_buggy = (is_bug_llm or exc_type_exists or metric_missing or csv_submission_failed)
        
        logger.info(f"{log_prefix}: Final buggy status for node {node.id}: {node.is_buggy}", extra={"verbose": True})

        if node.is_buggy:
            console.print(f"[bold red]Result: Buggy[/bold red]") # Console output
            bug_reasons = []
            if is_bug_llm: bug_reasons.append(f"LLM_judged_buggy (Summary: {trim_long_string(node.analysis)})")
            if exc_type_exists: bug_reasons.append(f"Exception_occurred ({node.exc_type})")
            if metric_missing: bug_reasons.append("Metric_missing_or_invalid_from_LLM")
            if not has_csv_submission_reported: bug_reasons.append("LLM_reported_CSV_missing")
            if not has_csv_submission_actual: bug_reasons.append("Actual_CSV_file_missing_from_disk")
            
            logger.warning(f"{log_prefix}: Node {node.id} is buggy. Reasons: {'; '.join(bug_reasons)}", extra={"verbose": True})
            node.metric = WorstMetricValue()
        else:
            console.print(f"[bold green]Result: Not Buggy[/bold green]") # Console output
            logger.info(f"{log_prefix}: Node {node.id} determined not buggy. Metric value: {metric_value}, LLM_says_lower_is_better: {review_response_dict.get('lower_is_better', True)}", extra={"verbose": True})
            node.metric = MetricValue(
                metric_value,
                maximize=not review_response_dict.get("lower_is_better", True) # Maximize if LLM says lower is NOT better
            )
        return node


#############################################################################
# PlannerAgent Implementation
#############################################################################
class PlannerAgent:
    def __init__(
        self,
        task_desc: str, # Dict or string
        cfg: Config,
        journal: Journal,
        wandb_run=None,
        competition_benchmarks=None,
    ):
        # super().__init__() # Not needed
        if isinstance(task_desc, dict):
            from .backend import compile_prompt_to_md # Local import
            self.task_desc = compile_prompt_to_md(task_desc)
        else:
            self.task_desc = task_desc
            
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        self._prev_buggy: bool = False # Tracks buggy status *before* reflection
        self.wandb_run = wandb_run
        self.competition_benchmarks = competition_benchmarks
        self.competition_name = self.cfg.competition_name
        self._code_quality: float = 0.0

        # Initialize W&B plot data lists (as in original PlannerAgent)
        self._metric_hist: List[float] = []
        self._bug_flags: List[int] = []
        self._sub_flags: List[int] = []
        self._above_median_flags: List[int] = []
        self._gold_medal_flags: List[int] = []
        self._silver_medal_flags: List[int] = []
        self._bronze_medal_flags: List[int] = []

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        # This logic is identical to Agent's search_policy, kept for consistency
        # If it needs to differ for PlannerAgent, it can be modified here.
        search_cfg = self.acfg.search
        logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Determining next action.", extra={"verbose": True})

        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Draft new node (drafts: {len(self.journal.draft_nodes)} < {search_cfg.num_drafts}).", extra={"verbose": True})
            return None

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                n for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Debug node {node_to_debug.id} (debug_prob triggered, depth {node_to_debug.debug_depth}).", extra={"verbose": True})
                return node_to_debug
            else:
                logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Attempted debug (debug_prob triggered), but no debuggable nodes found.", extra={"verbose": True})

        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Draft new node (no good nodes to improve).", extra={"verbose": True})
            return None

        greedy_node = self.journal.get_best_node()
        if greedy_node:
            if greedy_node.is_buggy:
                 logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Debug greedy node {greedy_node.id} (it was marked buggy).", extra={"verbose": True})
                 return greedy_node
            logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Improve greedy node {greedy_node.id} (metric: {greedy_node.metric.value:.3f if greedy_node.metric else 'N/A'}).", extra={"verbose": True})
            return greedy_node
        else:
            logger.info(f"PLANNER_AGENT_SEARCH_POLICY_STEP{self.current_step}: Selected: Draft new node (no best_node found, fallback).", extra={"verbose": True})
            return None


    # REMOVE: _prompt_environment, _prompt_impl_guideline, _prompt_resp_fmt,
    # debug_prompt_resp_fmt, code_prompt_resp_fmt, plan_prompt_resp_fmt
    # These are now handled by functions in prompt_utils.py

    def _query_llm_with_retries( # This method is specific to PlannerAgent
        self,
        query_type: str, # "PLANNER" or "CODER" or "DEBUG_PLANNER" etc. for logging
        system_prompt: Dict[str, Any],
        user_prompt: Dict[str, Any],
        model: str,
        temperature: float,
        planner_flag: bool, # Passed to backend `query`
        # current_step: int, # Already an attribute self.current_step
        convert_system_to_user: bool,
        retries: int = 3,
    ) -> Any: # Returns raw completion text or None
        """Helper function to query LLM with retries and detailed logging for PlannerAgent."""
        completion_text = None
        # Use self.current_step for logging
        log_prefix = f"PLANNER_AGENT_LLM_QUERY_{query_type.upper()}_STEP{self.current_step}"

        for attempt in range(retries):
            logger.info(f"{log_prefix}_ATTEMPT{attempt+1}/{retries}: Sending request. Model: {model}, Temp: {temperature}, PlannerFlag: {planner_flag}", extra={"verbose": True})
            # Verbose logging of prompts can be done here.
            # logger.debug(f"{log_prefix}_SYSTEM_PROMPT_START\n{system_prompt}\n{log_prefix}_SYSTEM_PROMPT_END", extra={"verbose": True})
            # logger.debug(f"{log_prefix}_USER_PROMPT_START\n{user_prompt}\n{log_prefix}_USER_PROMPT_END", extra={"verbose": True})

            try:
                completion_text = query(
                    system_message=system_prompt,
                    user_message=user_prompt,
                    model=model,
                    temperature=temperature,
                    planner=planner_flag, # Backend param
                    current_step=self.current_step,
                    convert_system_to_user=convert_system_to_user,
                    max_tokens=self.acfg.code.max_new_tokens # Assuming this should be passed for planner too
                )
                logger.info(f"{log_prefix}_ATTEMPT{attempt+1}: Received response.", extra={"verbose": True})
                # logger.debug(f"{log_prefix}_RAW_RESPONSE_START\n{completion_text}\n{log_prefix}_RAW_RESPONSE_END", extra={"verbose": True})
                return completion_text
            except Exception as e:
                logger.error(f"{log_prefix}_ATTEMPT{attempt+1}: Error during LLM query: {e}", exc_info=True, extra={"verbose": True})
                if attempt == retries - 1:
                    logger.error(f"{log_prefix}: All {retries} retries failed.", extra={"verbose": True})
                    return None
                time.sleep(2) # Simple backoff
        return None


    def plan_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        """Generate a natural language plan (and summary for task)."""
        system_prompt = get_planner_agent_plan_system_prompt()
        log_prefix = f"PLANNER_AGENT_PLAN_QUERY_STEP{self.current_step}"

        completion_text = self._query_llm_with_retries(
            query_type="PLANNER_PLAN", # Specific log type
            system_prompt=system_prompt,
            user_prompt=user_prompt_dict,
            model=self.acfg.code.planner_model, # Use planner_model from config
            temperature=self.acfg.code.temp, # Use general temp or specific planner temp if available
            planner_flag=True, # This call is for planning
            convert_system_to_user=self.acfg.convert_system_to_user,
            retries=retries,
        )

        if completion_text is None:
            logger.error(f"{log_prefix}: Failed to get response. Returning empty plan and summary.", extra={"verbose": True})
            return "", "", "" # Summary, Plan, (empty string for exec_summary compatibility)

        summary = extract_summary(completion_text) # Assumes "## Task Summary:"
        plan = extract_plan(completion_text)       # Assumes "## Plan:"

        if plan and summary:
            logger.info(f"{log_prefix}: Successfully extracted plan and summary.", extra={"verbose": True})
            logger.debug(f"{log_prefix}_EXTRACTED_SUMMARY_START\n{summary}\n{log_prefix}_EXTRACTED_SUMMARY_END", extra={"verbose": True})
            logger.debug(f"{log_prefix}_EXTRACTED_PLAN_START\n{plan}\n{log_prefix}_EXTRACTED_PLAN_END", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix}: Plan or summary extraction failed. Raw text: '{trim_long_string(str(completion_text))}'", extra={"verbose": True})
            # Fallback: return raw text as plan if specific extraction fails
            summary = summary or "SUMMARY_EXTRACTION_FAILED"
            plan = plan or str(completion_text) # Use raw if plan not found

        return summary, plan, "" # Empty string for execution_summary compatibility


    def code_query(self, user_prompt_dict: Dict[str, Any], retries: int = 3) -> tuple[str, str, str]:
        """Generate code based on a plan."""
        system_prompt = get_planner_agent_code_system_prompt()
        log_prefix = f"PLANNER_AGENT_CODE_QUERY_STEP{self.current_step}"

        completion_text = self._query_llm_with_retries(
            query_type="PLANNER_CODER", # Specific log type
            system_prompt=system_prompt,
            user_prompt=user_prompt_dict,
            model=self.acfg.code.model, # Use coder_model from config
            temperature=self.acfg.code.temp,
            planner_flag=False, # This call is for coding
            convert_system_to_user=self.acfg.convert_system_to_user,
            retries=retries,
        )

        if completion_text is None:
            logger.error(f"{log_prefix}: Failed to get response. Returning empty code.", extra={"verbose": True})
            return "", "", "" # NL (empty), Code (empty), Exec_summary (empty)

        code = extract_code(completion_text)
        # nl_text for coder is usually empty with current prompts, but can be extracted if needed:
        # nl_text = extract_text_up_to_code(completion_text) 

        if code:
            logger.info(f"{log_prefix}: Successfully extracted code.", extra={"verbose": True})
            # logger.debug(f"{log_prefix}_EXTRACTED_CODE_START\n{code}\n{log_prefix}_EXTRACTED_CODE_END", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix}: Code extraction failed. Raw text: '{trim_long_string(str(completion_text))}'", extra={"verbose": True})
            code = str(completion_text) # Fallback to raw text as code

        return "", code, "" # NL (empty), Code, Exec_summary (empty)


    def _draft(self, parent_node=None) -> Node:
        log_prefix = f"PLANNER_AGENT_DRAFT_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting drafting process. Parent: {parent_node.id if parent_node else 'None'}", extra={"verbose": True})

        # --- Plan Generation ---
        plan_user_prompt = get_planner_agent_draft_plan_user_prompt(
            task_desc=self.task_desc,
            journal_summary=self.journal.generate_summary(include_code=False),
            competition_name=self.competition_name,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        logger.info(f"{log_prefix}: Calling plan_query for drafting plan.", extra={"verbose": True})
        task_summary, agent_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))

        if not agent_plan:
            logger.error(f"{log_prefix}: Plan generation FAILED. Cannot proceed with code generation.", extra={"verbose": True})
            return Node(plan="PLAN_GENERATION_FAILED", code="# PLAN_GENERATION_FAILED", summary=task_summary or "PLAN_GENERATION_FAILED", parent=parent_node)

        # --- Code Generation ---
        code_user_prompt = get_planner_agent_draft_code_user_prompt(
            task_summary_from_planner=task_summary,
            plan_from_planner=agent_plan,
            journal_summary=self.journal.generate_summary(include_code=False),
            competition_name=self.competition_name,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        logger.info(f"{log_prefix}: Calling code_query for drafting code.", extra={"verbose": True})
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))

        if not generated_code:
            logger.warning(f"{log_prefix}: Code generation FAILED. Using placeholder.", extra={"verbose": True})
            generated_code = "# CODE_GENERATION_FAILED"
        
        formatted_code = format_code(generated_code) # Format for logging/storage if desired
        logger.debug(f"{log_prefix}_FORMATTED_DRAFT_CODE_START\n{formatted_code}\n{log_prefix}_FORMATTED_DRAFT_CODE_END", extra={"verbose": True})


        new_node = Node(
            plan=agent_plan,
            code=generated_code, # Store raw code from LLM
            summary=task_summary, # Summary from planner
            task_summary = task_summary # Storing task_summary from planner if different from initial self.task_desc
        )
        if parent_node: new_node.parent = parent_node
        logger.info(f"{log_prefix}: Drafted new node {new_node.id}.", extra={"verbose": True})
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        log_prefix = f"PLANNER_AGENT_IMPROVE_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting improvement for node {parent_node.id}.", extra={"verbose": True})

        # --- Plan Generation for Improvement ---
        plan_user_prompt = get_planner_agent_improve_plan_user_prompt(
            task_desc=self.task_desc, # Could use parent_node.task_summary if it's more relevant
            parent_node_code=parent_node.code,
            competition_name=self.competition_name,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        logger.info(f"{log_prefix}: Calling plan_query for improvement plan.", extra={"verbose": True})
        task_summary_for_improve, improvement_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))

        if not improvement_plan:
            logger.error(f"{log_prefix}: Improvement plan generation FAILED for node {parent_node.id}.", extra={"verbose": True})
            return Node(plan="IMPROVEMENT_PLAN_FAILED", code=parent_node.code, summary=task_summary_for_improve or "IMPROVEMENT_PLAN_FAILED", parent=parent_node)

        # --- Code Generation for Improvement ---
        code_user_prompt = get_planner_agent_improve_code_user_prompt(
            task_summary_from_planner=task_summary_for_improve,
            improvement_plan_from_planner=improvement_plan,
            parent_node_code=parent_node.code,
            journal_summary=self.journal.generate_summary(include_code=False),
            competition_name=self.competition_name,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        logger.info(f"{log_prefix}: Calling code_query for improvement code.", extra={"verbose": True})
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))

        if not generated_code:
            logger.warning(f"{log_prefix}: Improvement code generation FAILED. Reverting to parent code for node {parent_node.id}.", extra={"verbose": True})
            generated_code = parent_node.code
        
        formatted_code = format_code(generated_code)
        logger.debug(f"{log_prefix}_FORMATTED_IMPROVED_CODE_START\n{formatted_code}\n{log_prefix}_FORMATTED_IMPROVED_CODE_END", extra={"verbose": True})

        new_node = Node(
            plan=improvement_plan,
            code=generated_code,
            summary=task_summary_for_improve, # Summary about the improvement task
            task_summary=task_summary_for_improve, # If planner provides a new task context
            parent=parent_node
        )
        logger.info(f"{log_prefix}: Improved node {parent_node.id} to new node {new_node.id}.", extra={"verbose": True})
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        log_prefix = f"PLANNER_AGENT_DEBUG_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Starting debugging for node {parent_node.id}.", extra={"verbose": True})
        logger.debug(f"{log_prefix}_PARENT_CODE_START\n{parent_node.code}\n{log_prefix}_PARENT_CODE_END", extra={"verbose": True})
        logger.debug(f"{log_prefix}_PARENT_TERM_OUT_START\n{parent_node.term_out}\n{log_prefix}_PARENT_TERM_OUT_END", extra={"verbose": True})

        # --- Plan Generation for Debugging ---
        plan_user_prompt = get_planner_agent_debug_plan_user_prompt(
            task_desc=self.task_desc, # Or parent_node.task_summary
            parent_node_code=parent_node.code,
            parent_node_term_out=parent_node.term_out, # Already trimmed string
            # competition_name=self.competition_name, # If needed for planner
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        logger.info(f"{log_prefix}: Calling plan_query for debug plan.", extra={"verbose": True})
        bug_summary, fix_plan, _ = self.plan_query(plan_user_prompt, retries=self.acfg.get('query_retries', 3))

        if not fix_plan:
            logger.error(f"{log_prefix}: Debug plan generation FAILED for node {parent_node.id}.", extra={"verbose": True})
            return Node(plan="DEBUG_PLAN_FAILED", code=parent_node.code, summary=bug_summary or "DEBUG_PLAN_FAILED", parent=parent_node)

        # --- Code Generation for Debugging ---
        code_user_prompt = get_planner_agent_debug_code_user_prompt(
            bug_summary_from_planner=bug_summary,
            fix_plan_from_planner=fix_plan,
            parent_node_code=parent_node.code,
            parent_node_term_out=parent_node.term_out,
            competition_name=self.competition_name,
            acfg_data_preview=self.acfg.data_preview,
            data_preview_content=self.data_preview
        )
        logger.info(f"{log_prefix}: Calling code_query for debug code.", extra={"verbose": True})
        _, generated_code, _ = self.code_query(code_user_prompt, retries=self.acfg.get('query_retries', 3))

        if not generated_code:
            logger.warning(f"{log_prefix}: Debug code generation FAILED. Reverting to parent code for node {parent_node.id}.", extra={"verbose": True})
            generated_code = parent_node.code
            
        formatted_code = format_code(generated_code)
        logger.debug(f"{log_prefix}_FORMATTED_DEBUG_CODE_START\n{formatted_code}\n{log_prefix}_FORMATTED_DEBUG_CODE_END", extra={"verbose": True})

        new_node = Node(
            plan=fix_plan,
            code=generated_code,
            summary=bug_summary, # Summary of bugs/analysis
            task_summary=bug_summary, # If this serves as new context
            parent=parent_node
        )
        logger.info(f"{log_prefix}: Debugged node {parent_node.id} to new node {new_node.id}.", extra={"verbose": True})
        return new_node

    # reflect, double_reflect, update_data_preview, step, parse_exec_result
    # can be inherited or copied from Agent class if the logic is identical.
    # For now, assuming they are similar enough to be copied/adapted.
    # If they are significantly different, they need separate implementations.

    # Copying methods from Agent that are likely to be similar:
    # reflect, double_reflect (if used), update_data_preview, step, parse_exec_result
    # Note: self-reflection in PlannerAgent might use planner_model for critique and coder_model for edit.
    
    def reflect(self, node: Node) -> tuple[str, str]:
        # This uses acfg.code.model (coder) for the edit stage.
        # Critique stage in perform_two_step_reflection also uses acfg.code.model currently.
        # If critique should use planner_model, perform_two_step_reflection needs adjustment or
        # a different reflection function for PlannerAgent.
        # For now, assume current `perform_two_step_reflection` is acceptable.
        log_prefix = f"PLANNER_AGENT_REFLECT_STEP{self.current_step}_NODE{node.id}"
        logger.info(f"{log_prefix}: Initiating self-reflection.", extra={"verbose": True})
        try:
            reflection_plan, revised_code = perform_two_step_reflection(
                code=node.code,
                analysis=node.analysis,
                term_out=node.term_out,
                task_desc=self.task_desc, # Could use node.task_summary if more specific
                model_name=self.acfg.code.model, # Coder model for applying edits
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query,
                wrap_code_func=prompt_utils_wrap_code,
                extract_code_func=extract_code,
                current_step=self.current_step
            )
        except Exception as e:
            logger.error(f"{log_prefix}: Error during self-reflection call: {e}", exc_info=True, extra={"verbose": True})
            return f"REFLECTION_ERROR: {e}", node.code
        
        # Logging of reflection outcome (same as Agent)
        if revised_code and revised_code.strip() and revised_code != node.code:
            logger.info(f"{log_prefix}: Self-reflection resulted in code changes.", extra={"verbose": True})
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info(f"{log_prefix}: Self-reflection found no errors requiring changes.", extra={"verbose": True})
        else:
            logger.warning(f"{log_prefix}: Self-reflection finished, but revised code is same as original or empty. Plan: {trim_long_string(reflection_plan)}", extra={"verbose": True})
        
        logger.debug(f"{log_prefix}_REFLECTION_PLAN_START\n{reflection_plan}\n{log_prefix}_REFLECTION_PLAN_END", extra={"verbose": True})
        # logger.debug(f"{log_prefix}_REVISED_CODE_BY_REFLECTION_START\n{wrap_code(revised_code)}\n{log_prefix}_REVISED_CODE_BY_REFLECTION_END", extra={"verbose": True})
        return reflection_plan, revised_code

    def update_data_preview(self): # Identical to Agent
        log_prefix = f"PLANNER_AGENT_DATA_PREVIEW_STEP{self.current_step}"
        logger.info(f"{log_prefix}: Updating data preview.", extra={"verbose": True})
        try:
            self.data_preview = data_preview.generate(self.cfg.workspace_dir / "input")
            logger.info(f"{log_prefix}: Data preview updated.", extra={"verbose": True})
            logger.debug(f"{log_prefix}_DATA_PREVIEW_CONTENT_START\n{self.data_preview}\n{log_prefix}_DATA_PREVIEW_CONTENT_END", extra={"verbose": True})
        except Exception as e:
            logger.error(f"{log_prefix}: Failed to update data preview: {e}", exc_info=True, extra={"verbose": True})
            self.data_preview = "Error generating data preview."

    # step() method is largely the same structure as Agent's step(), just calls PlannerAgent's _draft, _improve, _debug
    def step(self, exec_callback: ExecCallbackType, current_step_number: int):
        # This is mostly copied from Agent.step and adapted for PlannerAgent logging prefixes
        # The core logic (search_policy -> _draft/_improve/_debug -> execute -> parse -> reflect -> log) is the same.
        log_prefix_main = f"PLANNER_AGENT_STEP{current_step_number}"
        logger.info(f"{log_prefix_main}_START: Total Steps Configured: {self.acfg.steps}", extra={"verbose": True})
        t_step_start = time.time()

        submission_dir = self.cfg.workspace_dir / "submission"
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

        logger.info(f"{log_prefix_main}: Executing code for node {result_node.id} (stage: {node_stage}). Code length: {len(result_node.code)}", extra={"verbose": True})
        # logger.debug(f"{log_prefix_main}_CODE_TO_EXECUTE_NODE_{result_node.id}_START\n{result_node.code}\n{log_prefix_main}_CODE_TO_EXECUTE_NODE_{result_node.id}_END", extra={"verbose": True})
        
        exec_start_time = time.time()
        exec_result = exec_callback(result_node.code, reset_session=True)
        exec_duration = time.time() - exec_start_time
        
        logger.info(f"{log_prefix_main}: Code execution for node {result_node.id} finished in {exec_duration:.2f}s.", extra={"verbose": True})
        # logger.debug(f"{log_prefix_main}_EXEC_RESULT_NODE_{result_node.id}_TERM_OUT_START\n{exec_result.term_out}\n{log_prefix_main}_EXEC_RESULT_NODE_{result_node.id}_TERM_OUT_END", extra={"verbose": True})
        if exec_result.exc_type:
             logger.warning(f"{log_prefix_main}_EXEC_RESULT_NODE_{result_node.id}_EXCEPTION: {exec_result.exc_type}", extra={"verbose": True})

        logger.info(f"{log_prefix_main}: Parsing execution results for node {result_node.id}.", extra={"verbose": True})
        result_node = self.parse_exec_result(node=result_node, exec_result=exec_result)
        buggy_status_before_reflection = result_node.is_buggy

        reflection_applied = False
        if draft_flag and self.acfg.ITS_Strategy == "self-reflection" and result_node.is_buggy:
            logger.info(f"{log_prefix_main}: Condition met for self-reflection on drafted buggy node {result_node.id}.", extra={"verbose": True})
            reflection_plan, reflection_code = self.reflect(node=result_node)
            if reflection_code and reflection_code.strip() and reflection_code != result_node.code:
                logger.info(f"{log_prefix_main}: Self-reflection yielded new code for node {result_node.id}. Re-executing.", extra={"verbose": True})
                result_node.code = reflection_code
                reflection_applied = True
                # ... (re-execution logic as in Agent.step) ...
                exec_start_time_reflect = time.time()
                exec_result_reflect = exec_callback(result_node.code, reset_session=True)
                exec_duration = time.time() - exec_start_time_reflect
                logger.info(f"{log_prefix_main}: Reflected code execution for node {result_node.id} finished in {exec_duration:.2f}s.", extra={"verbose": True})
                result_node = self.parse_exec_result(node=result_node, exec_result=exec_result_reflect)
            else:
                logger.info(f"{log_prefix_main}: Self-reflection did not result in applicable code changes for node {result_node.id}.", extra={"verbose": True})
        
        if buggy_status_before_reflection and not result_node.is_buggy:
            result_node.effective_debug_step = True
            result_node.effective_reflections = reflection_applied
        else:
            result_node.effective_debug_step = False
            result_node.effective_reflections = False
        self._prev_buggy = result_node.is_buggy

        # --- W&B Logging (Identical to Agent.step's W&B logging section) ---
        # This section is quite long and W&B specific. It's copied verbatim here.
        # Ensure all attributes like self.competition_benchmarks, self.wandb_run are available.
        logger.info(f"{log_prefix_main}: Preparing step log data for W&B.", extra={"verbose": True})
        step_log_data = {
            f"exec/exec_time_s": exec_duration,
            f"eval/is_buggy": 1 if result_node.is_buggy else 0,
            f"progress/current_step": current_step_number,
            f"progress/competition_name": self.competition_name,
            "exec/exception_type": result_node.exc_type if result_node.exc_type else "None",
            f"code/estimated_quality": int(self._code_quality),
            f"eval/reflection_applied_successfully": 1 if reflection_applied and not result_node.is_buggy else 0,
            f"eval/effective_fix_this_step": 1 if result_node.effective_debug_step else 0,
        }
        agent_validation_metrics_defined = False
        if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None:
            step_log_data[f"eval/validation_metric"] = result_node.metric.value
            agent_validation_metrics_defined = True
            if self.competition_benchmarks and wandb and self.wandb_run:
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
        if wandb and self.wandb_run:
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
        if self.wandb_run:
            logger.info(f"{log_prefix_main}: Logging data to W&B. Keys: {list(step_log_data.keys())}", extra={"verbose": True})
            try: self.wandb_run.log(step_log_data, step=current_step_number)
            except Exception as e_wandb: logger.error(f"{log_prefix_main}: Error logging to W&B: {e_wandb}", exc_info=True, extra={"verbose": True})
        # --- End W&B Logging Section Copy ---

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
        t_step_end = time.time()
        logger.info(f"{log_prefix_main}_END: Duration: {t_step_end - t_step_start:.2f}s", extra={"verbose": True})

    # parse_exec_result is identical to Agent's, can be inherited or directly used if Agent is a base class
    # For now, copying it ensures PlannerAgent is self-contained if needed.
    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        log_prefix = f"PLANNER_AGENT_PARSE_EXEC_STEP{self.current_step}_NODE{node.id}"
        logger.info(f"{log_prefix}: Parsing execution result.", extra={"verbose": True})
        node.absorb_exec_result(exec_result)
        # logger.debug(f"{log_prefix}_ABSORBED_NODE_EXC_TYPE: {node.exc_type}", extra={"verbose": True})

        introduction = ("You are a Kaggle grandmaster ... evaluate the output ... empirical findings.")
        if self.acfg.obfuscate:
            introduction = ("You are an expert machine learning engineer ... evaluate the output ... empirical findings.")

        feedback_system_prompt = {
            "Introduction": introduction, "Task Description": self.task_desc,
            "Code Executed": prompt_utils_wrap_code(node.code),
            "Execution Output Log": prompt_utils_wrap_code(node.term_out, lang=""),
        }
        max_retries = self.acfg.feedback.get("retries", 3)
        review_response_dict: Optional[Dict[str, Any]] = None

        for attempt in range(max_retries):
            logger.info(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}/{max_retries}: Querying feedback LLM.", extra={"verbose": True})
            try:
                raw_response = query(
                    system_message=feedback_system_prompt, user_message=None,
                    func_spec=review_func_spec, model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    convert_system_to_user=self.acfg.convert_system_to_user,
                    current_step=self.current_step
                )
                # logger.debug(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_RAW_RESPONSE_START\n{raw_response}\n{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}_RAW_RESPONSE_END", extra={"verbose": True})
                if not isinstance(raw_response, dict):
                    logger.error(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Response not dict. Type: {type(raw_response)}", extra={"verbose": True})
                    if isinstance(raw_response, str):
                        try: parsed_raw_response = json.loads(raw_response)
                        except Exception: parsed_raw_response = None
                        if isinstance(parsed_raw_response, dict): raw_response = parsed_raw_response
                        else: raw_response = None
                    else: raw_response = None
                review_response_dict = cast(Dict[str, Any], raw_response) if isinstance(raw_response, dict) else None
                if review_response_dict and all(k in review_response_dict for k in review_func_spec.json_schema["required"]):
                    break
                else: review_response_dict = None # Force retry
            except Exception as e: logger.error(f"{log_prefix}_FEEDBACK_LLM_ATTEMPT{attempt+1}: Error: {e}", exc_info=True, extra={"verbose": True})
            if attempt == max_retries - 1 and review_response_dict is None:
                review_response_dict = {"is_bug": True, "has_csv_submission": False, "summary": "LLM feedback failed.", "metric": None, "lower_is_better": True, "code_quality": 0}
                break
        if review_response_dict is None: review_response_dict = {"is_bug": True, "has_csv_submission": False, "summary": "CRITICAL: review_response_dict None.", "metric": None, "lower_is_better": True, "code_quality": 0}

        metric_value = review_response_dict.get("metric")
        if not isinstance(metric_value, (float, int)): metric_value = None
        self._code_quality = review_response_dict.get("code_quality", 0)
        if not isinstance(self._code_quality, (int, float)): self._code_quality = 0
        node.code_quality = int(self._code_quality)
        has_csv_submission_actual = (self.cfg.workspace_dir / "submission" / "submission.csv").exists()
        has_csv_submission_reported = review_response_dict.get("has_csv_submission", False)
        node.analysis = review_response_dict.get("summary", "Feedback LLM summary missing.")
        # logger.debug(f"{log_prefix}_LLM_ANALYSIS_SUMMARY_START\n{node.analysis}\n{log_prefix}_LLM_ANALYSIS_SUMMARY_END", extra={"verbose": True})
        node.is_buggy = (review_response_dict.get("is_bug", True) or node.exc_type is not None or metric_value is None or not has_csv_submission_reported or not has_csv_submission_actual)
        # logger.info(f"{log_prefix}: Final buggy status for node {node.id}: {node.is_buggy}", extra={"verbose": True})
        if node.is_buggy:
            # ... (bug reason logging similar to Agent) ...
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(metric_value, maximize=not review_response_dict.get("lower_is_better", True))
        return node


