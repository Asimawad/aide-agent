# aide/utils/wandb_logger.py
import logging
import shutil
from pathlib import Path
import pandas as pd
import time 
import re
from typing import Optional
try:
    import wandb
    from omegaconf import OmegaConf # Import OmegaConf here
except ImportError:
    wandb = None
    OmegaConf = None # Handle if OmegaConf is also optional, though likely not

from aide.utils.config import Config 
from aide.journal import Journal 
from . import copytree 

logger = logging.getLogger("aide.wandb") 

class WandbLogger:
    def __init__(self, cfg: Config, app_logger: logging.Logger):
        self.cfg = cfg
        self.wandb_run = None
        self.app_logger = app_logger 
        
        self._metric_hist: list[float] = []
        self._bug_flags: list[int] = []
        self._sub_flags: list[int] = []
        self._above_median_flags: list[int] = []
        self._gold_medal_flags: list[int] = []
        self._silver_medal_flags: list[int] = []
        self._bronze_medal_flags: list[int] = []


    def _sanitize_artifact_name_component(self, name_component: str) -> str:
        """Sanitizes a string component for use in a W&B artifact name."""
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name_component)
        # Ensure it doesn't start or end with non-alphanumeric (except dots for versioning)
        sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
        sanitized = re.sub(r'[^a-zA-Z0-9_]+$', '', sanitized) # Allow underscore at the end
        if not sanitized: # if all chars were invalid
            return "default_component"
        return sanitized


    def init_wandb(self):
        if wandb and OmegaConf and self.cfg.wandb.enabled: # Check OmegaConf too
            try:
                # Convert OmegaConf to a plain Python dictionary
                resolved_cfg_container = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=False)
                
                self.wandb_run = wandb.init(
                    project=self.cfg.wandb.project,
                    entity=self.cfg.wandb.entity,
                    name=self.cfg.wandb.run_name,
                    config=resolved_cfg_container, # USE THE RESOLVED CONTAINER
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

    # ... rest of the WandbLogger class remains the same ...
    def log_step_data(self, step_data: dict, current_step_number: int):
        if self.wandb_run:
            try:
                is_buggy_val = step_data.get("eval/is_buggy", 1) 
                submission_produced_val = step_data.get("eval/submission_produced", 0)
                metric_val = step_data.get("eval/validation_metric", float('nan'))

                self._bug_flags.append(is_buggy_val)
                self._sub_flags.append(submission_produced_val)
                if not pd.isna(metric_val) and is_buggy_val == 0 : 
                    self._metric_hist.append(metric_val)
                
                if wandb: 
                    bug_count = sum(self._bug_flags); clean_count = len(self._bug_flags) - bug_count
                    if bug_count + clean_count > 0: 
                        bug_table = wandb.Table(data=[["Buggy", bug_count], ["Clean", clean_count]], columns=["label", "count"])
                        step_data["plots/bug_vs_clean_summary"] = wandb.plot.bar(bug_table, "label", "count", title="Buggy vs Clean Steps (Summary)")

                    with_sub = sum(self._sub_flags); without_sub = len(self._sub_flags) - with_sub
                    if with_sub + without_sub > 0:
                        sub_table = wandb.Table(data=[["Has submission", with_sub], ["No submission", without_sub]], columns=["label", "count"])
                        step_data["plots/submission_presence_summary"] = wandb.plot.bar(sub_table, "label", "count", title="Submission Produced vs Missing (Summary)")

                    if self._metric_hist:
                        metric_table_data = [[v] for v in self._metric_hist if isinstance(v, (int, float))]
                        if metric_table_data:
                            tbl = wandb.Table(data=metric_table_data, columns=["val"])
                            step_data["plots/val_metric_scatter_summary"] = wandb.plot.scatter(tbl, "val", "val", title="All Valid Metrics (Summary)")

                self.wandb_run.log(step_data, step=current_step_number)
                logger.debug(f"W&B: Logged step {current_step_number} data.", extra={"verbose": True})
            except Exception as e:
                logger.error(f"W&B: Error logging step data: {e}", exc_info=True)

    def _copy_best_solution_and_submission_for_wandb(self):
        logs_exp_dir = Path("logs") / self.cfg.exp_name
        # logs_exp_dir.mkdir(parents=True, exist_ok=True) # This creates the parent, not the artifact dir

        workspaces_exp_dir = self.cfg.workspace_dir # This is already Path(cfg.workspace_dir)
        
        best_solution_src = workspaces_exp_dir / "best_solution"
        best_solution_dst_dir = logs_exp_dir / "best_solution_wandb_artifact" # Define full dst path

        best_submission_src = workspaces_exp_dir / "best_submission" # Assuming you create this
        best_submission_dst_dir = logs_exp_dir / "best_submission_wandb_artifact"

        if best_solution_src.exists():
            best_solution_dst_dir.mkdir(parents=True, exist_ok=True) # Create the specific dst dir
            copytree(best_solution_src, best_solution_dst_dir, use_symlinks=False)
            logger.info(f"W&B: Copied best_solution to W&B staging: {best_solution_dst_dir}")
        
        if best_submission_src.exists(): # You might not be creating a 'best_submission' folder yet
            best_submission_dst_dir.mkdir(parents=True, exist_ok=True) # Create the specific dst dir
            copytree(best_submission_src, best_submission_dst_dir, use_symlinks=False)
            logger.info(f"W&B: Copied best_submission to W&B staging: {best_submission_dst_dir}")
        else:
            logger.info(f"W&B: best_submission directory not found at {best_submission_src}, skipping copy.")

    def finalize_run(self, journal: Journal, competition_benchmarks: Optional[dict]):
        if self.wandb_run:
            self.app_logger.info("W&B: Finalizing run...")
            try:
                # ... (your existing summary_data calculation) ...
                summary_data = {}
                wo_step = None; no_of_csvs = 0; buggy_nodes_count = 0
                total_code_quality = 0; valid_code_quality_nodes = 0
                gold_medals = 0; silver_medals = 0; bronze_medals = 0
                above_median_count = 0; effective_debugs_count = 0

                for node in journal.nodes:
                    if not node.is_buggy:
                        if wo_step is None: wo_step = node.step # step is 0-indexed in journal
                        # Check for submission.csv for this specific node if you save them per node
                        # For now, let's assume submission.csv is overwritten or a general check
                        # submission_path_for_node = self.cfg.workspace_dir / "submission" / f"submission_node_{node.id}.csv"
                        # For simplicity, let's check the general submission.csv for now
                        if (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                             no_of_csvs += 1 # This will count it for every non-buggy step if file exists. Be careful.
                                             # Consider only counting if the node's code *produced* it.

                        if node.code_quality is not None: # code_quality is float
                            total_code_quality += node.code_quality
                            valid_code_quality_nodes +=1
                        if node.effective_debug_step: effective_debugs_count += 1 # This seems to be a boolean
                        
                        if competition_benchmarks and node.metric and node.metric.value is not None:
                            metric_val = node.metric.value
                            if metric_val >= competition_benchmarks.get("gold_threshold", float('inf')): gold_medals += 1
                            elif metric_val >= competition_benchmarks.get("silver_threshold", float('inf')): silver_medals += 1
                            elif metric_val >= competition_benchmarks.get("bronze_threshold", float('inf')): bronze_medals += 1
                            
                            if metric_val >= competition_benchmarks.get("median_threshold", float('inf')): above_median_count += 1
                    else:
                        buggy_nodes_count += 1
                
                summary_data["summary/steps_to_first_working_code"] = (wo_step + 1) if wo_step is not None else (self.cfg.agent.steps + 10) # Adjust for 1-based display
                summary_data["summary/num_successful_submissions"] = no_of_csvs # This needs careful thought on how it's counted
                summary_data["summary/num_buggy_nodes"] = buggy_nodes_count
                summary_data["summary/avg_code_quality_non_buggy"] = (total_code_quality / valid_code_quality_nodes) if valid_code_quality_nodes > 0 else 0
                if competition_benchmarks:
                    summary_data["summary/gold_medals_achieved"] = gold_medals
                    summary_data["summary/silver_medals_achieved"] = silver_medals
                    summary_data["summary/bronze_medals_achieved"] = bronze_medals
                    summary_data["summary/steps_above_median"] = above_median_count
                summary_data["summary/effective_debug_steps"] = effective_debugs_count
                
                best_node = journal.get_best_node(only_good=True)
                if best_node and best_node.metric and best_node.metric.value is not None:
                    summary_data["summary/best_validation_metric"] = best_node.metric.value
                    summary_data["summary/best_node_id"] = best_node.id
                    summary_data["summary/best_node_step"] = best_node.step + 1 # Adjust for 1-based display

                self.wandb_run.summary.update(summary_data)
                self.app_logger.info(f"W&B: Updated run summary: {summary_data}")

                self._copy_best_solution_and_submission_for_wandb()
                log_dir_for_artifacts = Path("logs") / self.cfg.exp_name

                # Sanitize artifact names
                sanitized_exp_name = self._sanitize_artifact_name_component(self.cfg.exp_name)

                if (log_dir_for_artifacts / "journal.json").exists():
                    artifact_name_journal = f"{sanitized_exp_name}_journal"
                    self.app_logger.info(f"W&B: Logging journal artifact as: {artifact_name_journal}")
                    artifact = wandb.Artifact(artifact_name_journal, type="journal")
                    artifact.add_file(str(log_dir_for_artifacts / "journal.json"))
                    self.wandb_run.log_artifact(artifact)

                if (log_dir_for_artifacts / "best_solution_wandb_artifact").exists():
                    artifact_name_code = f"{sanitized_exp_name}_best_solution"
                    self.app_logger.info(f"W&B: Logging best solution artifact as: {artifact_name_code}")
                    artifact_code = wandb.Artifact(artifact_name_code, type="solution-code")
                    artifact_code.add_dir(str(log_dir_for_artifacts / "best_solution_wandb_artifact"))
                    self.wandb_run.log_artifact(artifact_code)
                
                # ... (saving aide.log and aide.verbose.log - these are direct file saves, not artifacts, so name is less critical) ...
                if (self.cfg.log_dir / "aide.log").exists():
                     self.wandb_run.save(str(self.cfg.log_dir / "aide.log"), base_path=str(self.cfg.log_dir.parent))
                if (self.cfg.log_dir / "aide.verbose.log").exists():
                     self.wandb_run.save(str(self.cfg.log_dir / "aide.verbose.log"), base_path=str(self.cfg.log_dir.parent))


            except Exception as e:
                self.app_logger.error(f"W&B: Error during summary/artifact logging: {e}", exc_info=True)
            finally:
                if self.wandb_run: # Check if it's still valid before finishing
                    self.wandb_run.finish()
                    self.app_logger.info("W&B run finished.")
        else:
            self.app_logger.info("W&B run not available, skipping finalization.")