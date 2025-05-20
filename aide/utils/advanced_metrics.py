import os
import json
import pathlib
from collections import Counter, defaultdict
import sys
import numpy as np
from ..journal import journal2report
from . import config
from pathlib import Path
import time

try:
    cfg = config.load_cfg()
    task = config.load_task_desc(cfg=cfg)
    run_name = cfg.exp_name
except Exception as e:
    pass
# --- Configuration ---
BASE_RUN_DATA_DIR = "."
RUN_FOLDER_NAME = f"logs/{run_name}" or f"logs/run_"
JOURNAL_RELATIVE_PATH = f"journal.json"

BEST_CODE_RELATIVE_PATH = f"best_solution.py"

BEST_NODE_ID_RELATIVE_PATH = f"best_solution/node_id.txt"


def find_best_node_id():
    """Tries to find the best node ID from the cached file."""
    best_node_id_path = cfg.workspace_dir / cfg.exp_name / BEST_NODE_ID_RELATIVE_PATH
    if best_node_id_path.exists():
        try:
            with open(best_node_id_path, "r") as f:
                node_id = f.read().strip()
                if node_id:
                    try:
                        return int(node_id)
                    except ValueError:
                        print(
                            f"Warning: Could not parse best node ID from {best_node_id_path}",
                            file=sys.stderr,
                        )
                        return None
                else:
                    print(
                        f"Warning: Best node ID file {best_node_id_path} is empty.",
                        file=sys.stderr,
                    )
                    return None
        except Exception as e:
            print(
                f"Error reading best node ID file {best_node_id_path}: {e}",
                file=sys.stderr,
            )
            return None
    else:
        # If not found, the script might not be able to identify the best code reliably.
        print(
            f"Warning: Best node ID file not found at {best_node_id_path}. Cannot reliably identify best node's code.",
            file=sys.stderr,
        )
        return None


def generate_comprehensive_report(
    run_folder_path: Path, metrics: dict, journal=None
) -> str:
    """Generate a comprehensive report combining all metrics and information."""
    report_lines = []

    # Add header
    report_lines.append("# Run Report")
    report_lines.append(f"\n## Run Information")
    report_lines.append(f"- Run Name: {cfg.exp_name}")
    report_lines.append(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Add Advanced Metrics
    report_lines.append("\n## Advanced Metrics")
    if metrics:
        for key, value in metrics.items():
            report_lines.append(f"- {key}: {value}")

    # Add Empirical Metrics if available
    empirical_metrics_path = run_folder_path / "calculated_metrics_results.json"
    if empirical_metrics_path.exists():
        try:
            with open(empirical_metrics_path, "r") as f:
                empirical_metrics = json.load(f)
                report_lines.append("\n## Empirical Metrics")
                if "average_metrics" in empirical_metrics:
                    for key, value in empirical_metrics["average_metrics"].items():
                        report_lines.append(f"- {key}: {value}")
        except Exception as e:
            report_lines.append(f"\nError loading empirical metrics: {e}")

    # Add Journal Summary if available
    if journal:
        try:
            report_lines.append("\n## Journal Summary")
            report_lines.append(journal.generate_summary(include_code=False))
        except Exception as e:
            report_lines.append(f"\nError generating journal summary: {e}")

    return "\n".join(report_lines)


def calculate_advanced_metrics(run_folder_name: str, journal=None):
    """
    Calculates metrics based on the journal and best code files in a local run folder.

    Args:
        run_folder_name (str): name for a specific run.

    Returns:
        dict: A dictionary containing the calculated advanced metrics, or None if files are missing.
    """
    run_folder_path = Path(f"logs/{run_folder_name}")
    journal_path = Path(f"logs/{run_folder_name}") / JOURNAL_RELATIVE_PATH
    best_code_path = Path(f"logs/{run_folder_name}") / BEST_CODE_RELATIVE_PATH
    metrics = {}

    # --- Metric: LOC of Best Solution ---
    if best_code_path.exists():
        try:
            with open(best_code_path, "r") as f:
                code_content = f.read()
                # Count lines, including empty lines
                loc = code_content.count("\n") + 1 if code_content else 0
                metrics["LOC of Best Solution"] = loc
            print(f"Calculated LOC of Best Solution: {loc}")
        except Exception as e:
            print(
                f"Error reading best code file {best_code_path}: {e}", file=sys.stderr
            )
            metrics["LOC of Best Solution"] = None
    else:
        print(
            f"Best code file not found at {best_code_path}. Skipping LOC calculation.",
            file=sys.stderr,
        )
        metrics["LOC of Best Solution"] = None

    # --- Metrics from Journal Analysis ---
    if journal_path.exists():
        try:
            with open(str(journal_path), "r") as f:
                journal_data = json.load(f)

            # Each node dictionary is assumed to have 'id', 'parent_id' (can be null), 'is_buggy', 'stage_name', 'children' (list of IDs)
            nodes_list = journal_data.get("nodes", [])

            if not nodes_list:
                print(
                    f"Journal file {journal_path} is empty or has no 'nodes'. Cannot calculate journal metrics.",
                    file=sys.stderr,
                )
                return metrics  # Return metrics calculated so far (maybe just LOC)

            # Build a map from node ID to node data for easier access
            node_map = {node["id"]: node for node in nodes_list}

            # --- Metric: Search Depth ---

            child_to_parent = {}
            # Also build parent-to-children for branching factor
            parent_to_children = defaultdict(list)
            root_nodes = []

            for node in nodes_list:
                node_id = node.get("id")
                parent_id = node.get("parent_id")  # Assuming 'parent_id' key

                if node_id is None:
                    print(
                        f"Warning: Node without 'id' found in journal: {node}. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                if parent_id is not None:
                    child_to_parent[node_id] = parent_id
                    parent_to_children[parent_id].append(node_id)
                else:
                    root_nodes.append(node_id)

            if not root_nodes and nodes_list:
                print(
                    "Warning: No root nodes found in journal. Possibly malformed tree structure. Calculating depth from all nodes.",
                    file=sys.stderr,
                )
                # Fallback: If no root, just find longest parent chain starting from any node
                max_depth = 0
                for node_id in node_map:
                    current_depth = 0
                    current_id = node_id
                    visited = set()

                    while (
                        current_id is not None
                        and current_id in node_map
                        and current_id not in visited
                    ):
                        visited.add(current_id)
                        current_depth += 1
                        current_id = node_map[current_id].get("parent_id")
                    max_depth = max(max_depth, current_depth)

            else:
                # Standard tree traversal from roots
                max_depth = 0
                # Use a stack for iterative DFS or queue for iterative BFS
                stack = [
                    (root_id, 1) for root_id in root_nodes
                ]  # (node_id, current_depth)

                while stack:
                    current_id, current_depth = stack.pop()  # DFS

                    max_depth = max(max_depth, current_depth)

                    # Find children using the parent_to_children map
                    for child_id in parent_to_children.get(current_id, []):
                        # Ensure child exists in node_map before adding (robustness)
                        if child_id in node_map:
                            stack.append((child_id, current_depth + 1))
                        else:
                            print(
                                f"Warning: Child ID {child_id} of node {current_id} not found in node map.",
                                file=sys.stderr,
                            )

            metrics["Search Depth (Max Tree Depth)"] = max_depth
            print(f"Calculated Search Depth: {max_depth}")

            # --- Metric: Branching Factor ---
            total_children_count = sum(
                len(children) for children in parent_to_children.values()
            )
            nodes_with_children = len(
                parent_to_children
            )  # Count nodes that are parents

            # Handle division by zero if no nodes have children (e.g., flat structure)
            branching_factor = (
                total_children_count / nodes_with_children
                if nodes_with_children > 0
                else 0
            )
            metrics["Average Branching Factor"] = round(branching_factor, 2)
            print(
                f"Calculated Average Branching Factor: {metrics['Average Branching Factor']}"
            )

            # --- Metric: Buggy Node Debug Rate ---
            buggy_node_ids = {
                node["id"] for node in nodes_list if node.get("is_buggy", False)
            }
            debug_parent_ids = {
                node.get("parent_id")
                for node in nodes_list
                if node.get("stage_name") == "debug"
                and node.get("parent_id") is not None
            }

            # Count how many unique buggy node IDs are also parents of debug nodes
            buggy_parents_debugged = len(buggy_node_ids.intersection(debug_parent_ids))

            # Calculate rate: (Buggy parents debugged / Total unique buggy nodes) * 100%
            # Handle division by zero if no buggy nodes existed
            buggy_debug_rate = (
                (buggy_parents_debugged / len(buggy_node_ids)) * 100
                if len(buggy_node_ids) > 0
                else 0
            )
            metrics["Buggy Node Debug Rate (%)"] = round(buggy_debug_rate, 2)
            print(
                f"Calculated Buggy Node Debug Rate: {metrics['Buggy Node Debug Rate (%)']}"
            )
            print(
                f"  ({buggy_parents_debugged} out of {len(buggy_node_ids)} unique buggy nodes were debugged)"
            )

            # get a report
            if journal is not None:
                try:
                    final_report = journal2report(journal=journal, task_desc=task)
                    with open(file=run_folder_path / "report.md", mode="w") as f:
                        f.writelines(final_report)
                except Exception as e:
                    print(f"cant get the final report, skipping....")

            print("\n--- Advanced Metrics ---")
            if metrics:
                # Use json.dumps with sort_keys for consistent output
                print(json.dumps(metrics, indent=4, sort_keys=True))

                # Optional: Save results to a JSON file within the run folder
                output_filename = run_folder_path / "advanced_metrics.json"
                try:
                    with open(output_filename, "w") as f:
                        json.dump(metrics, f, indent=4, sort_keys=True)
                    print(f"\nAdvanced metrics also saved to {output_filename}")
                except Exception as e:
                    print(
                        f"Error saving advanced metrics to {output_filename}: {e}",
                        file=sys.stderr,
                    )
            else:
                print("Could not calculate advanced metrics.")

            # Generate and save comprehensive report
            try:
                report = generate_comprehensive_report(
                    run_folder_path, metrics, journal
                )
                report_path = run_folder_path / "report.md"
                with open(report_path, "w") as f:
                    f.write(report)
                print(f"\nComprehensive report saved to {report_path}")
            except Exception as e:
                print(f"Error generating comprehensive report: {e}")

        except FileNotFoundError:
            print(f"Journal file not found at {journal_path}.", file=sys.stderr)
            # metrics remain as calculated so far (maybe just LOC)
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from journal file {journal_path}. File might be corrupted.",
                file=sys.stderr,
            )
            # metrics remain as calculated so far
        except Exception as e:
            print(
                f"An unexpected error occurred while processing journal file {journal_path}: {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc()
    else:
        print(
            f"Journal file not found at {journal_path}. Skipping journal metrics.",
            file=sys.stderr,
        )
        # metrics remain as calculated so far (maybe just LOC)
    return metrics


# --- Script Execution ---
if __name__ == "__main__":
    # Expect the run folder name as a command line argument
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <run_folder_name>", file=sys.stderr)
        print(f"Example: python {sys.argv[0]} hszjppdp", file=sys.stderr)
        sys.exit(1)  # Exit if incorrect number of arguments

    run_folder_name = sys.argv[1]
    run_folder_path = pathlib.Path(BASE_RUN_DATA_DIR) / "logs" / run_folder_name

    print(f"Analyzing run data in folder: {run_folder_path}")

    if not run_folder_path.exists() or not run_folder_path.is_dir():
        print(
            f"Error: Run data directory not found or is not a directory: {run_folder_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Call the function to calculate metrics
    advanced_metrics = calculate_advanced_metrics(run_folder_path)

    print("\n--- Advanced Metrics ---")
    if advanced_metrics:
        # Use json.dumps with sort_keys for consistent output
        print(json.dumps(advanced_metrics, indent=4, sort_keys=True))

        # Optional: Save results to a JSON file within the run folder
        output_filename = run_folder_path / "advanced_metrics.json"
        try:
            with open(output_filename, "w") as f:
                json.dump(advanced_metrics, f, indent=4, sort_keys=True)
            print(f"\nAdvanced metrics also saved to {output_filename}")
        except Exception as e:
            print(
                f"Error saving advanced metrics to {output_filename}: {e}",
                file=sys.stderr,
            )
    else:
        print("Could not calculate advanced metrics.")
