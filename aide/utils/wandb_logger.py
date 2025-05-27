# aide/utils/wandb_logger.py
import logging
import shutil
from pathlib import Path
import pandas as pd
import time 
import re
from typing import Optional, Dict, Any # Added Dict and Any
try:
    import wandb
    from omegaconf import OmegaConf 
except ImportError:
    wandb = None
    OmegaConf = None 

from aide.utils.config import Config 
from aide.journal import Journal, Node # Import Node
from . import copytree 
from ..utils.metric import WorstMetricValue # For type checking

logger = logging.getLogger("aide.wandb") 

class WandbLogger:
    def __init__(self, cfg: Config, app_logger: logging.Logger, competition_benchmarks: Optional[Dict[str, Any]] = None): # Added competition_benchmarks
        self.cfg = cfg
        self.wandb_run = None
        self.app_logger = app_logger 
        self.competition_benchmarks = competition_benchmarks # Store benchmarks
        
        # Attributes to store data for plots across steps
        self._metric_hist: list[float] = []
        self._bug_flags: list[int] = []
        self._sub_flags: list[int] = []
        self._above_median_flags: list[int] = []
        self._gold_medal_flags: list[int] = []
        self._silver_medal_flags: list[int] = []
        self._bronze_medal_flags: list[int] = []


    def _sanitize_artifact_name_component(self, name_component: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name_component)
        sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
        sanitized = re.sub(r'[^a-zA-Z0-9_]+$', '', sanitized)
        if not sanitized: 
            return "default_component"
        return sanitized

    def init_wandb(self):
        if wandb and OmegaConf and self.cfg.wandb.enabled:
            try:
                resolved_cfg_container = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=False)
                
                self.wandb_run = wandb.init(
                    project=self.cfg.wandb.project,
                    entity=self.cfg.wandb.entity,
                    name=self.cfg.wandb.run_name,
                    config=resolved_cfg_container, 
                    job_type="aide_run",
                    tags=["aide-agent", self.cfg.agent.ITS_Strategy, self.cfg.agent.code.model],
                )
                self.app_logger.info(f"W&B run initialized: {self.wandb_run.url if self.wandb_run else 'Failed'}")
            except Exception as e:
                self.app_logger.error(f"Failed to initialize W&B: {e}", exc_info=True)
                self.wandb_run = None
        elif not OmegaConf:
            self.app_logger.error("OmegaConf is not available. Cannot serialize config for W&B.")
            self.wandb_run = None
        else:
            self.app_logger.info("W&B logging is disabled in the configuration.")

    def log_step_data(self, step_data: dict, result_node: Node, current_step_number: int):
        if not self.wandb_run:
            return

        try:
            # --- Update internal accumulators ---
            is_buggy_val = 1 if result_node.is_buggy else 0
            # Check if submission.csv exists for this step's result_node
            # This check relies on the file system state *after* execution and parsing.
            submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
            submission_exists_for_node = submission_path.exists() and not result_node.is_buggy
            submission_produced_val = 1 if submission_exists_for_node else 0
            
            metric_val = result_node.metric.value if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None else float('nan')

            self._bug_flags.append(is_buggy_val)
            self._sub_flags.append(submission_produced_val) # Use our calculated flag
            
            if not pd.isna(metric_val) and is_buggy_val == 0:
                self._metric_hist.append(metric_val)

            # --- Medal and Threshold Plots ---
            if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None and self.competition_benchmarks and wandb:
                current_metric_value = result_node.metric.value
                
                # Above Median
                self._above_median_flags.append(1 if current_metric_value >= self.competition_benchmarks.get("median_threshold", float('inf')) else 0)
                median_true = sum(self._above_median_flags); median_false = len(self._above_median_flags) - median_true
                if median_true + median_false > 0:
                    median_table = wandb.Table(data=[["Above Median", median_true], ["Below Median", median_false]], columns=["label","count"])
                    step_data["plots/above_median_bar"] = wandb.plot.bar(median_table, "label", "count", title="Above Median Steps")

                # Gold Medal
                self._gold_medal_flags.append(1 if current_metric_value >= self.competition_benchmarks.get("gold_threshold", float('inf')) else 0)
                gold_true = sum(self._gold_medal_flags); gold_false = len(self._gold_medal_flags) - gold_true
                if gold_true + gold_false > 0:
                    gold_table = wandb.Table(data=[["Gold Medal", gold_true], ["No Gold Medal", gold_false]], columns=["label","count"])
                    step_data["plots/gold_medal_bar"] = wandb.plot.bar(gold_table, "label", "count", title="Gold Medal Steps")

                # Silver Medal
                self._silver_medal_flags.append(1 if current_metric_value >= self.competition_benchmarks.get("silver_threshold", float('inf')) else 0)
                silver_true = sum(self._silver_medal_flags); silver_false = len(self._silver_medal_flags) - silver_true
                if silver_true + silver_false > 0:
                    silver_table = wandb.Table(data=[["Silver Medal", silver_true], ["No Silver Medal", silver_false]], columns=["label","count"])
                    step_data["plots/silver_medal_bar"] = wandb.plot.bar(silver_table, "label", "count", title="Silver Medal Steps")

                # Bronze Medal
                self._bronze_medal_flags.append(1 if current_metric_value >= self.competition_benchmarks.get("bronze_threshold", float('inf')) else 0)
                bronze_true = sum(self._bronze_medal_flags); bronze_false = len(self._bronze_medal_flags) - bronze_true
                if bronze_true + bronze_false > 0:
                    bronze_table = wandb.Table(data=[["Bronze Medal", bronze_true], ["No Bronze Medal", bronze_false]], columns=["label","count"])
                    step_data["plots/bronze_medal_bar"] = wandb.plot.bar(bronze_table, "label", "count", title="Bronze Medal Steps")
            
            # --- Buggy vs Clean Plot ---
            if wandb:
                bug_count = sum(self._bug_flags); clean_count = len(self._bug_flags) - bug_count
                if bug_count + clean_count > 0: 
                    bug_table = wandb.Table(data=[["Buggy", bug_count], ["Clean", clean_count]], columns=["label", "count"])
                    step_data["plots/bug_vs_clean_summary"] = wandb.plot.bar(bug_table, "label", "count", title="Buggy vs Clean Steps (Summary)")

                # --- Submission Presence Plot ---
                with_sub = sum(self._sub_flags); without_sub = len(self._sub_flags) - with_sub
                if with_sub + without_sub > 0:
                    sub_table = wandb.Table(data=[["Has submission", with_sub], ["No submission", without_sub]], columns=["label", "count"])
                    step_data["plots/submission_presence_summary"] = wandb.plot.bar(sub_table, "label", "count", title="Submission Produced vs Missing (Summary)")

                # --- Metric History Scatter Plot ---
                if self._metric_hist: # Check if list is not empty
                    # Filter out NaNs just in case, though append logic should prevent it
                    valid_metrics_for_plot = [m for m in self._metric_hist if isinstance(m, (int, float)) and not pd.isna(m)]
                    if valid_metrics_for_plot:
                        # Create data for scatter: (step_index, metric_value)
                        # Using a simple range for step_index for now
                        metric_scatter_data = [[i, m] for i, m in enumerate(valid_metrics_for_plot)]
                        if metric_scatter_data:
                             tbl = wandb.Table(data=metric_scatter_data, columns=["step_idx", "metric_value"])
                             step_data["plots/val_metric_history_scatter"] = wandb.plot.scatter(tbl, "step_idx", "metric_value", title="Validation Metric History")
            
            # Log the combined step_data
            self.wandb_run.log(step_data, step=current_step_number)
            logger.debug(f"W&B: Logged step {current_step_number} data. Keys: {list(step_data.keys())}", extra={"verbose": True})
        except Exception as e:
            logger.error(f"W&B: Error logging step data for step {current_step_number}: {e}", exc_info=True)


    def _copy_best_solution_and_submission_for_wandb(self):
        logs_exp_dir = Path("logs") / self.cfg.exp_name
        workspaces_exp_dir = self.cfg.workspace_dir 
        
        best_solution_src = workspaces_exp_dir / "best_solution"
        # Staging dir for W&B artifacts within the run's log directory
        wandb_artifacts_staging_dir = logs_exp_dir / "wandb_artifacts" 
        wandb_artifacts_staging_dir.mkdir(parents=True, exist_ok=True)

        best_solution_dst_dir = wandb_artifacts_staging_dir / "best_solution"
        best_submission_dst_dir = wandb_artifacts_staging_dir / "best_submission"

        if best_solution_src.exists():
            if best_solution_dst_dir.exists(): shutil.rmtree(best_solution_dst_dir) # Clean before copy
            copytree(best_solution_src, best_solution_dst_dir, use_symlinks=False) # Copy to specific artifact dir
            logger.info(f"W&B: Copied best_solution to W&B staging: {best_solution_dst_dir}")
        
        # Check for best_submission in workspace, not just logs_exp_dir
        best_submission_src_workspace = workspaces_exp_dir / "best_submission" 
        if best_submission_src_workspace.exists(): 
            if best_submission_dst_dir.exists(): shutil.rmtree(best_submission_dst_dir) # Clean before copy
            copytree(best_submission_src_workspace, best_submission_dst_dir, use_symlinks=False)
            logger.info(f"W&B: Copied best_submission from workspace to W&B staging: {best_submission_dst_dir}")
        else:
            logger.info(f"W&B: best_submission directory not found at {best_submission_src_workspace}, skipping copy for artifact.")


    def finalize_run(self, journal: Journal, competition_benchmarks: Optional[dict]): # competition_benchmarks passed here
        if not self.wandb_run:
            self.app_logger.info("W&B run not available, skipping finalization.")
            return

        self.app_logger.info("W&B: Finalizing run...")
        try:
            summary_data = {}
            wo_step = None; num_working_code_steps = 0; buggy_nodes_count = 0
            total_code_quality_non_buggy = 0; count_code_quality_non_buggy = 0
            gold_medals = 0; silver_medals = 0; bronze_medals = 0
            above_median_count = 0; effective_debugs_this_run = 0 # Renamed for clarity
            # Track if a submission was ever produced by a non-buggy node
            any_submission_produced_by_non_buggy = False

            for node in journal.nodes:
                if not node.is_buggy:
                    num_working_code_steps +=1
                    if wo_step is None: wo_step = node.step 
                    
                    if node.code_quality is not None:
                        total_code_quality_non_buggy += node.code_quality
                        count_code_quality_non_buggy +=1
                    
                    if node.effective_debug_step: # This flag is set on the node
                        effective_debugs_this_run += 1 
                    
                    # Check if this specific non-buggy node produced a submission
                    # This relies on how submission_produced was logged per step
                    # or a more robust check here. For now, let's assume the agent's
                    # step_log_data already included "eval/submission_produced" correctly.
                    # We'd need to fetch that from W&B history or store on node.
                    # A simpler proxy: check if submission file exists *now* and best_node is this node
                    # For now, let's count any non-buggy node as contributing to "submission produced" if the file exists for that node
                    # This part is tricky without per-node submission tracking.
                    # The old code just incremented `no_of_csvs`.
                    # Let's use `any_submission_produced_by_non_buggy`
                    submission_path_for_node_check = self.cfg.workspace_dir / "best_solution" / "submission.csv" # Check cached best
                    if node.id == journal.get_best_node().id and submission_path_for_node_check.exists():
                        any_submission_produced_by_non_buggy = True


                    if self.competition_benchmarks and node.metric and node.metric.value is not None:
                        metric_val = node.metric.value
                        if metric_val >= self.competition_benchmarks.get("gold_threshold", float('inf')): gold_medals += 1
                        elif metric_val >= self.competition_benchmarks.get("silver_threshold", float('inf')): silver_medals += 1
                        elif metric_val >= self.competition_benchmarks.get("bronze_threshold", float('inf')): bronze_medals += 1
                        
                        if metric_val >= self.competition_benchmarks.get("median_threshold", float('inf')): above_median_count += 1
                else:
                    buggy_nodes_count += 1
            
            summary_data["summary/steps_to_first_working_code"] = (wo_step + 1) if wo_step is not None else (self.cfg.agent.steps + 10)
            summary_data["summary/num_working_code_steps"] = num_working_code_steps
            summary_data["summary/num_buggy_nodes"] = buggy_nodes_count
            summary_data["summary/avg_code_quality_non_buggy"] = (total_code_quality_non_buggy / count_code_quality_non_buggy) if count_code_quality_non_buggy > 0 else 0
            if self.competition_benchmarks: # Check if benchmarks are available
                summary_data["summary/gold_medals_achieved"] = gold_medals
                summary_data["summary/silver_medals_achieved"] = silver_medals
                summary_data["summary/bronze_medals_achieved"] = bronze_medals
                summary_data["summary/steps_above_median"] = above_median_count
            summary_data["summary/effective_debug_steps_total"] = effective_debugs_this_run # Total effective fixes
            summary_data["summary/best_solution_produced_submission"] = 1 if any_submission_produced_by_non_buggy else 0


            best_node = journal.get_best_node(only_good=True) # Ensure it's a good node
            if best_node and best_node.metric and best_node.metric.value is not None:
                summary_data["summary/best_validation_metric"] = best_node.metric.value
                summary_data["summary/best_node_id"] = best_node.id
                summary_data["summary/best_node_step"] = best_node.step + 1

            self.wandb_run.summary.update(summary_data)
            self.app_logger.info(f"W&B: Updated run summary: {summary_data}")

            # --- Artifact Logging ---
            self._copy_best_solution_and_submission_for_wandb() # Ensure staging dir is populated
            
            log_dir_for_run = Path("logs") / self.cfg.exp_name # Main log dir for this run
            wandb_artifacts_staging_dir = log_dir_for_run / "wandb_artifacts" # Staging dir

            sanitized_exp_name = self._sanitize_artifact_name_component(self.cfg.exp_name)

            # Log journal artifact
            journal_file_path = log_dir_for_run / "journal.json"
            if journal_file_path.exists():
                artifact_name_journal = f"{sanitized_exp_name}_journal"
                self.app_logger.info(f"W&B: Logging journal artifact as: {artifact_name_journal}")
                artifact = wandb.Artifact(artifact_name_journal, type="run-journal")
                artifact.add_file(str(journal_file_path))
                self.wandb_run.log_artifact(artifact)

            # Log best solution code artifact (from staging directory)
            best_solution_artifact_dir = wandb_artifacts_staging_dir / "best_solution"
            if best_solution_artifact_dir.exists() and any(best_solution_artifact_dir.iterdir()):
                artifact_name_code = f"{sanitized_exp_name}_best_solution_code"
                self.app_logger.info(f"W&B: Logging best solution code artifact as: {artifact_name_code}")
                artifact_code = wandb.Artifact(artifact_name_code, type="solution-code")
                artifact_code.add_dir(str(best_solution_artifact_dir))
                self.wandb_run.log_artifact(artifact_code)
            
            # Log best submission artifact (from staging directory)
            best_submission_artifact_dir = wandb_artifacts_staging_dir / "best_submission"
            if best_submission_artifact_dir.exists() and any(best_submission_artifact_dir.iterdir()):
                artifact_name_submission = f"{sanitized_exp_name}_best_submission_file"
                self.app_logger.info(f"W&B: Logging best submission artifact as: {artifact_name_submission}")
                artifact_submission = wandb.Artifact(artifact_name_submission, type="submission-file")
                artifact_submission.add_dir(str(best_submission_artifact_dir)) # Log the directory
                self.wandb_run.log_artifact(artifact_submission)


            # Save main log files (these are direct saves, not versioned artifacts)
            if (self.cfg.log_dir / "aide.log").exists():
                 self.wandb_run.save(str(self.cfg.log_dir / "aide.log"), base_path=str(self.cfg.log_dir.parent))
            if (self.cfg.log_dir / "aide.verbose.log").exists():
                 self.wandb_run.save(str(self.cfg.log_dir / "aide.verbose.log"), base_path=str(self.cfg.log_dir.parent))
            
            # Save other generated reports if they exist
            for report_file in ["calculated_metrics_results.json", "advanced_metrics.json", "report.md", "history.csv"]:
                file_path = log_dir_for_run / report_file
                if file_path.exists():
                    self.wandb_run.save(str(file_path), base_path=str(log_dir_for_run.parent))
                    self.app_logger.info(f"W&B: Saved report file: {file_path.name}")


        except Exception as e:
            self.app_logger.error(f"W&B: Error during summary/artifact logging: {e}", exc_info=True)
        finally:
            if self.wandb_run: 
                self.wandb_run.finish()
                self.app_logger.info("W&B run finished.")