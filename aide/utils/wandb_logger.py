# aide/utils/wandb_logger.py
import logging
import shutil
from pathlib import Path
import pandas as pd
import numpy as np # For np.isnan
from typing import Optional
from omegaconf import OmegaConf
try:
    import wandb
except ImportError:
    wandb = None

from ..journal import Journal # For type hinting in finalize_run
from ..utils.config import Config # For type hinting cfg

class WandbLogger:
    def __init__(self, cfg: Config, app_logger: logging.Logger):
        self.cfg = cfg
        self.app_logger = app_logger # Main application logger for internal messages
        self.wandb_run = None
        self._metric_hist = [] # For plotting validation metric history
        self._bug_flags = []   # For plotting buggy vs clean steps
        self._sub_flags = []   # For plotting submission presence
        # For competition benchmark related plots (if any)
        self._above_median_flags = []
        self._gold_medal_flags = []
        self._silver_medal_flags = []
        self._bronze_medal_flags = []


    def init_wandb(self):
        if not wandb or not self.cfg.wandb.enabled:
            self.app_logger.info("W&B logging is disabled or wandb is not installed.")
            return

        try:
            # Ensure wandb directory exists if not default
            if self.cfg.wandb.get("dir"):
                Path(self.cfg.wandb.dir).mkdir(parents=True, exist_ok=True)

            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                name=self.cfg.wandb.run_name or self.cfg.exp_name,
                dir=self.cfg.wandb.get("dir", "./"), # Use wandb.dir from cfg or default to current
                config=OmegaConf.to_container(self.cfg, resolve=True),
                job_type="aide_run",
                tags=["aide-ds", self.cfg.agent.code.model, self.cfg.agent.ITS_Strategy, self.cfg.competition_name],
                save_code=False, # Usually better to manage code artifacts explicitly
                reinit=True # Allows re-initializing in the same process (e.g. testing)
            )
            self.app_logger.info(f"W&B Run initialized: {self.wandb_run.url if self.wandb_run else 'Failed'}")
        except Exception as e:
            self.app_logger.error(f"Failed to initialize W&B: {e}", exc_info=True)
            self.wandb_run = None

    def log_step_data(self, step_data: dict, current_step: int, result_node=None, competition_benchmarks=None):
        """
        Logs step data, including custom plots if a result_node is provided.
        `result_node` is the Node object from the agent after execution and parsing.
        `competition_benchmarks` is the dict loaded for the current competition.
        """
        if not self.wandb_run:
            return

        # --- Plotting Logic (moved from Agent.step) ---
        # This logic now uses self.wandb_run if needed, but preferably uses wandb module directly for plots
        # It also updates internal lists like self._metric_hist
        
        # Ensure result_node attributes are available for plotting
        is_buggy = step_data.get(f"eval/is_buggy", 1) == 1 # Default to buggy if not specified
        metric_value = step_data.get(f"eval/validation_metric") 
        submission_exists = step_data.get(f"eval/submission_produced", 0) == 1

        # --- Histogram/Scatter of validation metric ---
        if metric_value is not None and not (isinstance(metric_value, float) and np.isnan(metric_value)):
            self._metric_hist.append(metric_value)
        
        if len(self._metric_hist) >= 1: # Plot if at least one valid metric
            metric_table_data = [[v] for v in self._metric_hist if isinstance(v, (int, float)) and not np.isnan(v)]
            if metric_table_data: # Ensure there's data after filtering NaNs
                try:
                    tbl = wandb.Table(data=metric_table_data, columns=["validation_metric_value"])
                    # Scatter plot of metric values over steps
                    # To make a meaningful scatter, we need step numbers.
                    # We can accumulate (step, metric_value) pairs if needed, or just plot distribution.
                    # For now, let's keep it simple with a histogram.
                    step_data["plots/val_metric_histogram"] = wandb.plot.histogram(
                        tbl, "validation_metric_value", title="Validation Metric Distribution"
                    )
                except Exception as e:
                    self.app_logger.warning(f"W&B: Failed to create validation metric histogram: {e}")


        # --- Bar chart: Buggy (1) vs Clean (0) ---
        self._bug_flags.append(1 if is_buggy else 0)
        bug_count = sum(self._bug_flags)
        clean_count = len(self._bug_flags) - bug_count
        try:
            bug_table = wandb.Table(data=[["Buggy Steps", bug_count], ["Clean Steps", clean_count]], columns=["label", "count"])
            step_data["plots/bug_vs_clean_bar"] = wandb.plot.bar(bug_table, "label", "count", title="Buggy vs Clean Steps")
        except Exception as e:
            self.app_logger.warning(f"W&B: Failed to create bug_vs_clean bar chart: {e}")

        # --- Bar chart: Submission produced vs missing ---
        self._sub_flags.append(1 if submission_exists else 0)
        with_sub = sum(self._sub_flags)
        without_sub = len(self._sub_flags) - with_sub
        try:
            sub_table = wandb.Table(data=[["Submission Produced", with_sub], ["No Submission", without_sub]], columns=["label", "count"])
            step_data["plots/submission_presence_bar"] = wandb.plot.bar(sub_table, "label", "count", title="Submission Produced vs Missing")
        except Exception as e:
            self.app_logger.warning(f"W&B: Failed to create submission presence bar chart: {e}")
        
        # --- Competition Benchmark Plots ---
        if result_node and not is_buggy and metric_value is not None and not np.isnan(metric_value) and competition_benchmarks:
            current_metric = metric_value
            is_lower_better = result_node.metric.maximize is False # Assuming result_node.metric exists and has maximize

            def check_threshold(val, threshold, lower_is_better_flag):
                if threshold is None: return False
                return val <= threshold if lower_is_better_flag else val >= threshold

            threshold_map = {
                "above_median": (self._above_median_flags, competition_benchmarks.get("median_threshold")),
                "gold_medal": (self._gold_medal_flags, competition_benchmarks.get("gold_threshold")),
                "silver_medal": (self._silver_medal_flags, competition_benchmarks.get("silver_threshold")),
                "bronze_medal": (self._bronze_medal_flags, competition_benchmarks.get("bronze_threshold")),
            }

            for key_suffix, (flag_list, threshold_value) in threshold_map.items():
                if threshold_value is not None: # Only plot if threshold exists
                    is_met = 1 if check_threshold(current_metric, threshold_value, is_lower_better) else 0
                    flag_list.append(is_met)
                    true_count = sum(flag_list)
                    false_count = len(flag_list) - true_count
                    title_suffix = key_suffix.replace('_', ' ').title()
                    try:
                        table = wandb.Table(
                            data=[[f"Met {title_suffix}", true_count], [f"Not Met {title_suffix}", false_count]],
                            columns=["label", "count"]
                        )
                        step_data[f"plots/benchmark/{key_suffix}_bar"] = wandb.plot.bar(
                            table, "label", "count", title=f"Steps Meeting {title_suffix}"
                        )
                    except Exception as e:
                        self.app_logger.warning(f"W&B: Failed to create {key_suffix} bar chart: {e}")
        
        # --- Log all collected step_data ---
        try:
            self.wandb_run.log(step_data, step=current_step)
            self.app_logger.debug(f"W&B: Logged data for step {current_step}. Keys: {list(step_data.keys())}")
        except Exception as e:
            self.app_logger.error(f"W&B: Failed to log step data: {e}", exc_info=True)


    def finalize_run(self, journal: Journal, competition_benchmarks: Optional[dict] = None):
        if not self.wandb_run:
            self.app_logger.info("W&B: No active run to finalize.")
            return

        self.app_logger.info("W&B: Finalizing run...")
        try:
            # --- Log Summary Statistics ---
            summary_data = {}
            # Steps to first working code
            wo_step = None
            for node in journal.nodes:
                if not node.is_buggy: # Assuming is_buggy is False for working code
                    wo_step = node.step
                    break
            summary_data["summary/steps_to_first_working_code"] = wo_step if wo_step is not None else self.cfg.agent.steps + 10

            # Num successful submissions
            num_successful_submissions = sum(1 for node in journal.nodes if not node.is_buggy and (self.cfg.workspace_dir / "best_solution" / "submission.csv").exists()) # A bit simplistic, assumes best_solution is from a non-buggy node
            summary_data["summary/num_successful_submissions"] = num_successful_submissions
            
            # Num buggy nodes
            summary_data["summary/num_buggy_nodes"] = len(journal.buggy_nodes)

            # Avg code quality for non-buggy nodes
            non_buggy_qualities = [node.code_quality for node in journal.good_nodes if hasattr(node, 'code_quality') and node.code_quality is not None]
            summary_data["summary/avg_code_quality_non_buggy"] = np.mean(non_buggy_qualities) if non_buggy_qualities else 0
            
            # Benchmark related summaries
            if competition_benchmarks:
                summary_data["summary/gold_medals_achieved"] = sum(self._gold_medal_flags)
                summary_data["summary/silver_medals_achieved"] = sum(self._silver_medal_flags)
                summary_data["summary/bronze_medals_achieved"] = sum(self._bronze_medal_flags)
                summary_data["summary/steps_above_median"] = sum(self._above_median_flags)

            # Effective debug steps (from Node attribute)
            summary_data["summary/effective_debug_steps"] = sum(1 for node in journal.nodes if hasattr(node, 'effective_debug_step') and node.effective_debug_step)

            self.wandb_run.summary.update(summary_data)
            self.app_logger.info(f"W&B: Updated run summary: {summary_data}")

            # --- Save Logs and Artifacts ---
            # Save best solution and submission if they exist
            best_solution_dir_src = self.cfg.workspace_dir / "best_solution"
            best_submission_src = best_solution_dir_src / "submission.csv"
            best_code_src = best_solution_dir_src / "solution.py"

            if best_submission_src.exists():
                self.wandb_run.save(str(best_submission_src), base_path=str(best_solution_dir_src.parent), policy="live")
                self.app_logger.info(f"W&B: Saved best submission: {best_submission_src}")
            else:
                self.app_logger.info(f"W&B: best_submission file not found at {best_submission_src}, not saving.")
            
            if best_code_src.exists():
                self.wandb_run.save(str(best_code_src), base_path=str(best_solution_dir_src.parent), policy="live")
                self.app_logger.info(f"W&B: Saved best solution code: {best_code_src}")
            else:
                self.app_logger.info(f"W&B: best_solution.py not found at {best_code_src}, not saving.")


            # Save general log files
            log_files_to_save = ["aide.log", "aide.verbose.log"]
            for log_file in log_files_to_save:
                if (self.cfg.log_dir / log_file).exists():
                    self.wandb_run.save(str(self.cfg.log_dir / log_file), base_path=str(self.cfg.log_dir.parent), policy="end")
            
            # Save journal.json and config.yaml
            if (self.cfg.log_dir / "journal.json").exists():
                self.wandb_run.save(str(self.cfg.log_dir / "journal.json"), base_path=str(self.cfg.log_dir.parent), policy="end")
            if (self.cfg.log_dir / "config.yaml").exists():
                self.wandb_run.save(str(self.cfg.log_dir / "config.yaml"), base_path=str(self.cfg.log_dir.parent), policy="end")


        except Exception as e_sum:
            self.app_logger.error(f"W&B: Error during summary/artifact logging: {e_sum}", exc_info=True)
        finally:
            self.wandb_run.finish()
            self.app_logger.info("W&B run finished.")
            self.wandb_run = None # Reset for potential future runs in same process

    def get_wandb_run_obj(self):
        """Returns the raw wandb run object if initialized, else None."""
        return self.wandb_run