import logging
from rich.console import Console 
from typing import Any, Callable, Dict ,List
from aide.interpreter import ExecutionResult
from aide.journal import Node
from aide.agents.base import Agent
from aide.utils.prompt_utils import *
from aide.utils.response import (
    trim_long_string,
    extract_reflection_summary_and_revised_code,
)

logger = logging.getLogger("aide") 
console = Console()

def format_time(time_in_sec: int): # Should be float for more precision
    time_in_sec = int(time_in_sec) # Cast to int if original signature is intended
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

ExecCallbackType = Callable[[str, bool], ExecutionResult]

class CodeChainAgent(Agent): # Inherit from Agent
    
    

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