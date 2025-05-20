import wandb
import os
import pandas as pd
import json
import sys
import numpy as np
import logging


exp_name = None
try:
    from . import config

    cfg = config.load_cfg()
    exp_name = cfg.exp_name
except Exception as e:
    print("e")


WANDB_ENTITY = "asim_awad"
WANDB_PROJECT = "MLE_BENCH_AIDE"
logger = logging.getLogger("aide")
# This is CRITICAL. Adjust these filters to select the specific set of runs

RUN_FILTERS = {
    # # You need to know the exact value of cfg.task.name used in your runs
    # "config.goal": cfg.goal,
    # # "config.goal": cfg.goal,
    "display_name": cfg.exp_name,
    # You need to know the exact value of cfg.agent.code.model used in your runs
    "config.agent.code.model": cfg.agent.code.model,
    # Example: Filter runs for a specific strategy (if you logged it)
    "config.agent.ITS_Strategy": cfg.agent.ITS_Strategy,
}

MAX_STEPS_PER_ATTEMPT = 25

INF_STEPS_REPLACEMENT = MAX_STEPS_PER_ATTEMPT + 10  # penalty


def calculate_empirical_metrics(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    filters=RUN_FILTERS,
    max_steps=MAX_STEPS_PER_ATTEMPT,
    inf_steps_replacement=INF_STEPS_REPLACEMENT,
    run_name=exp_name,
    id=None,
):
    """
    Calculates empirical and other informative metrics for a set of runs.

    Args:
        entity (str): W&B entity name.
        project (str): W&B project name.
        filters (dict): Dictionary of W&B config filters to select runs.
        max_steps (int): Maximum number of steps per run (attempt), used as denominator for some rates.
        inf_steps_replacement (int): Value to use for 'infinity' when averaging steps to WO.

    Returns:
        dict: A dictionary containing the calculated metrics, or None if no runs found.
    """
    try:
        api = wandb.Api()
        project_path = f"{entity}/{project}"

        print(f"Searching for runs in '{project_path}' with filters: {filters}")
        if id:
            run = api.run(f"{entity}/{project}/{id}")
            print(f"Found run matching the filters. Proceeding to calculate metrics.")
        else:
            # Get runs matching the filters
            runs = api.runs(project_path, filters=filters)
            print(
                f"Found {len(runs)} run(s) matching the filters. Proceeding to calculate metrics."
            )

        # Lists to store metrics for each individual run before averaging
        run_vcgrs = []
        run_csars = []
        run_steps_to_wo = []  # Steps to First Working Code
        run_total_times = []  # Total Run Time
        run_avg_exec_times = []  # Average Step Execution Time
        run_avg_code_quality = []  # Average LLM Estimated Code Quality

        # Dictionary to aggregate exception counts across all runs
        aggregated_exception_counts = {}

        # Track runs that didn't find working code
        runs_without_working_code = 0
        calc_only_one = True  # False
        if calc_only_one:
            runs = [run]
        for i, run in enumerate(runs):
            print(f"\nProcessing run {i+1}/{len(runs)}: '{run.name}' ({run.id})")

            # Retrieve the run's history (logged data per step)
            history = run.history(pandas=True)

            history.to_csv(f"logs/{run.name}/history.csv")
            if history.empty:
                print(
                    f"Warning: Run '{run.name}' ({run.id}) has no logged history steps. Skipping."
                )
                continue

            total_logged_steps = len(history)
            # Use fixed max_steps as denominator as per paper definition
            denominator_steps_for_rates = max_steps

            # Steps are considered "valid code" if 'exec/exception_type' is null.
            valid_code_count = 0
            history_df = pd.DataFrame(history)

            if "eval/is_buggy" in history.columns:
                valid_code_count = (history_df["eval/is_buggy"] == 0).sum()
                # Aggregate exception types frequency
            else:
                print(
                    f"Warning: 'eval/is_buggy' column not found in history for run '{run.name}'. Cannot calculate VCGR or exception counts for this run."
                )
                pass

            vcgr = (
                (valid_code_count / denominator_steps_for_rates) * 100
                if denominator_steps_for_rates > 0
                else 0
            )
            print(
                f"  Valid Code Steps: {valid_code_count}/{total_logged_steps} (using {denominator_steps_for_rates} for % calculation)"
            )
            run_vcgrs.append(vcgr)

            # CSAR: Percentage of steps that attempted a CSV submission.
            # Steps are considered "submission attempt" if 'eval/submission_produced' is 1.
            submission_attempt_count = 0
            if "eval/submission_produced" in history.columns:
                submission_attempt_count = (
                    history["eval/submission_produced"] == 1
                ).sum()
            else:
                print(
                    f"Warning: 'eval/submission_produced' column not found in history for run '{run.name}'. Cannot calculate CSAR for this run."
                )

                pass
            csar = (
                (submission_attempt_count / denominator_steps_for_rates) * 100
                if denominator_steps_for_rates > 0
                else 0
            )
            print(
                f"  Submission Attempt Steps: {submission_attempt_count}/{total_logged_steps} (using {denominator_steps_for_rates} for % calculation)"
            )
            run_csars.append(csar)

            # Average Step Execution Time
            avg_exec_time = np.nan  # Default if column is missing or empty
            if (
                "exec/exec_time_s" in history.columns
                and not history["exec/exec_time_s"].empty
            ):
                avg_exec_time = history["exec/exec_time_s"].mean()
            else:
                print(
                    f"Warning: 'exec/exec_time_s' column not found or empty in history for run '{run.name}'. Cannot calculate average execution time."
                )
            run_avg_exec_times.append(avg_exec_time)
            print(avg_exec_time)

            # Average LLM Estimated Code Quality
            avg_code_quality = np.nan  # Default if column is missing or empty
            if (
                "code/estimated_quality" in history.columns
                and not history["code/estimated_quality"].empty
            ):
                # Ensure the column is numeric, coercing errors to NaN
                avg_code_quality = pd.to_numeric(
                    history["code/estimated_quality"], errors="coerce"
                ).mean()
            else:
                print(
                    f"Warning: 'code/estimated_quality' column not found or empty in history for run '{run.name}'. Cannot calculate average code quality."
                )
            run_avg_code_quality.append(avg_code_quality)

            #         # --- Retrieve Metrics from run.summary ---

            # Steps to First Working Code (WO)
            steps_to_wo = run.summary.get("steps_to_first_working_code", float("inf"))
            if steps_to_wo == float("inf"):
                runs_without_working_code += 1
                # Use replacement value for averaging, or handle separately later
                run_steps_to_wo.append(inf_steps_replacement)
            elif isinstance(steps_to_wo, (int, float)):
                run_steps_to_wo.append(steps_to_wo)
            else:
                print(
                    f"Warning: Unexpected value for 'steps_to_first_working_code' in summary for run '{run.name}': {steps_to_wo}. Treating as infinity."
                )
                runs_without_working_code += 1
                run_steps_to_wo.append(
                    inf_steps_replacement
                )  # Append replacement value

            print(f"WO_{run_steps_to_wo}")

        # --- Average Metrics Across Runs (Seeds) ---
        num_valid_runs = len(
            runs
        )  # Number of runs found and processed (those with history)

        # Use numpy.nanmean to ignore NaN values when averaging
        avg_vcgr = np.nanmean(run_vcgrs) if run_vcgrs else np.nan
        avg_csar = np.nanmean(run_csars) if run_csars else np.nan

        # Option 1: Average using replacement value (simplest)
        avg_steps_to_wo = np.nanmean(run_steps_to_wo) if run_steps_to_wo else np.nan
        # Option 2: Average only for runs that found working code (more accurate avg for *successful* runs)
        steps_wo_found = [s for s in run_steps_to_wo if s < inf_steps_replacement]
        avg_steps_to_wo_found_only = (
            np.mean(steps_wo_found) if steps_wo_found else np.nan
        )
        percentage_no_working_code = (
            (runs_without_working_code / num_valid_runs) * 100
            if num_valid_runs > 0
            else 0
        )

        avg_total_time = np.nanmean(run_total_times) if run_total_times else np.nan
        avg_avg_exec_time = (
            np.nanmean(run_avg_exec_times) if run_avg_exec_times else np.nan
        )

        avg_avg_code_quality = (
            np.nanmean(run_avg_code_quality) if run_avg_code_quality else np.nan
        )

        results = {
            "filters_used": filters,
            "num_runs_found": len(runs),  # Total runs found by filter
            "num_runs_processed": num_valid_runs,  # Runs that had history and were processed
            "average_metrics": {
                "VCGR (%)": round(avg_vcgr, 2) if not np.isnan(avg_vcgr) else None,
                "CSAR (%)": round(avg_csar, 2) if not np.isnan(avg_csar) else None,
                "Steps to First Working Code inf replaced": (
                    round(avg_steps_to_wo, 2) if not np.isnan(avg_steps_to_wo) else None
                ),
                "Steps to First Working Code (only for runs that found it)": (
                    round(avg_steps_to_wo_found_only, 2)
                    if not np.isnan(avg_steps_to_wo_found_only)
                    else None
                ),
                "Percentage of Runs with No Working Code (%)": round(
                    percentage_no_working_code, 2
                ),
                "Total Run Time (seconds)": (
                    round(avg_total_time, 2) if not np.isnan(avg_total_time) else None
                ),
                "Average Step Execution Time (seconds/step)": (
                    round(avg_avg_exec_time, 4)
                    if not np.isnan(avg_avg_exec_time)
                    else None
                ),
                "Average LLM Estimated Code Quality (0-10)": (
                    round(avg_avg_code_quality, 2)
                    if not np.isnan(avg_avg_code_quality)
                    else None
                ),
                "Aggregated Exception Type Frequencies": aggregated_exception_counts,
            },
            # Optional: Include individual run data for more detailed analysis
            "individual_run_data": {
                "VCGR": run_vcgrs,
                "CSAR": run_csars,
                "StepsToWO": run_steps_to_wo,
                "TotalTime": run_total_times,
                "AvgExecTime": run_avg_exec_times,
                "AvgCodeQuality": run_avg_code_quality,
            },
        }

        # Optional: Save results to a JSON file
        output_filename = f"logs/{run.name}/calculated_metrics_results.json"
        try:
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)
            print(f"\nResults also saved to {output_filename}")
        except Exception as e:
            print(f"Error saving results to {output_filename}: {e}", file=sys.stderr)
        return results

    except Exception as e:
        print(f"\nAn error occurred during metric calculation: {e}", file=sys.stderr)
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        return None


# --- Script Execution ---
if __name__ == "__main__":

    print("Starting empirical and informative metric calculation script...")

    # Call the function to calculate metrics
    metrics_results = calculate_empirical_metrics(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        filters=RUN_FILTERS,
        max_steps=MAX_STEPS_PER_ATTEMPT,
        inf_steps_replacement=INF_STEPS_REPLACEMENT,
    )

    if metrics_results:
        print("\n--- Calculated Metrics ---")
        # Use json.dumps with to handle potential NaN values nicely in output
        print(json.dumps(metrics_results, indent=4, sort_keys=True))

        # Optional: Save results to a JSON file
        output_filename = f"calculated_metrics_results.json"
        try:
            with open(output_filename, "w") as f:
                json.dump(metrics_results, f, indent=4, sort_keys=True)
            print(f"\nResults also saved to {output_filename}")
        except Exception as e:
            print(f"Error saving results to {output_filename}: {e}", file=sys.stderr)
    else:
        print("\nMetric calculation failed or no runs found matching the filters.")
