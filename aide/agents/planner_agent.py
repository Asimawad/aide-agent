import logging
from rich.console import Console 
from typing import Callable
from aide.interpreter import ExecutionResult
from aide.journal import Node
from aide.agents.base import Agent
from aide.utils.self_reflection import perform_two_step_reflection
from aide.backend import query
from aide.utils.prompt_utils import *
from aide.utils.response import (
    trim_long_string,
    extract_code,
)

logger = logging.getLogger("aide") 
console = Console()

def format_time(time_in_sec: int): # Should be float for more precision
    time_in_sec = int(time_in_sec) # Cast to int if original signature is intended
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

ExecCallbackType = Callable[[str, bool], ExecutionResult]

class PlannerAgent(Agent):

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