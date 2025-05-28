# aide/utils/metrics_calculator.py
import json
import logging
import pandas as pd
import numpy as np
import re
import time
import os
import shutil
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from omegaconf import OmegaConf
from typing import Optional, cast, List, Dict, Any
import sys
from . import copytree

from .config import load_cfg
import pandas as pd

# --- Configuration ---
cfg = load_cfg()
try:
    from radon.complexity import cc_visit
    RADON_AVAILABLE = True
    logger_radon = logging.getLogger("radon") 
    logger_radon.setLevel(logging.CRITICAL) 
except ImportError:
    RADON_AVAILABLE = False
    logger_radon = None # Define to avoid UnboundLocalError if radon is not available


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None #
    

try:
    from ..journal import Journal, Node, filter_journal, get_path_to_node
    from .config import Config # For type hinting config
    from . import serialize
    from .metric import MetricValue, WorstMetricValue 
except ImportError: 
    print("Warning: Could not perform relative imports for metrics_calculator.py. Ensure script is run as a module or PYTHONPATH is set.")
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) 
    from aide.journal import Journal, Node, filter_journal, get_path_to_node
    from aide.utils.config import Config
    from aide.utils import serialize
    from aide.utils.metric import MetricValue, WorstMetricValue

logger = logging.getLogger("aide.metrics_calculator")
if not logger.handlers: # Setup basic logging if run standalone
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Configuration Constants (can be overridden by args in __main__) ---
DEFAULT_WANDB_ENTITY = "asim_awad"
DEFAULT_WANDB_PROJECT = "MLE_BENCH_AIDE"
DEFAULT_MAX_API_RETRIES = 5
DEFAULT_API_RETRY_DELAY = 10

WANDB_ENTITY = "asim_awad"

WANDB_PROJECT = "MLE_BENCH_AIDE_VM"

WANDB_RUN_NAME = cfg.wandb.run_name

DOWNLOAD_DIR = "./logs"

FILE_FILTER_PATTERN = cfg.wandb.run_name  # solve me


def copy_best_solution_and_submission():
    workspaces_dir = os.path.join("workspaces", cfg.exp_name)
    logs_dir = Path(os.path.join("logs", cfg.exp_name))
    best_submission_dir = Path(os.path.join(workspaces_dir, "best_submission"))
    try:
        if os.path.exists(best_submission_dir):
            copytree(best_submission_dir, logs_dir, use_symlinks=False)
            
        shutil.copy(os.path.join(best_submission_dir, "submission.csv"), os.path.join(logs_dir, "submission.csv"))
        logger.info(f"Copied best_submission directory to {logs_dir}")
    except Exception as e:
        logger.error(f"Error copying best_submission directory to {logs_dir}: {e}", exc_info=True)


def save_logs_to_wandb():
    copy_best_solution_and_submission()

    logger.info("Saving logs directory to WandB")
    wandb.save(f"logs/{cfg.exp_name}/*", base_path="logs")  # Save log files


# --- Data Loading Functions ---
def load_run_config(run_logs_path: Path) -> Optional[Config]:
    config_path = run_logs_path / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return None
    try:
        cfg_loaded = OmegaConf.load(config_path)
        if not (hasattr(cfg_loaded, 'agent') and hasattr(cfg_loaded.agent, 'code')):
             logger.error(f"Loaded config from {config_path} does not appear to be a valid AIDE Config structure.")
             return None
        return cast(Config, cfg_loaded) 
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}", exc_info=True)
        return cfg

def load_run_journal(run_logs_path: Path) -> Optional[Journal]:
    journal_path = run_logs_path / "journal.json"
    if not journal_path.exists():
        logger.error(f"Journal file not found at {journal_path}")
        return None
    try:
        journal = serialize.load_json(journal_path, Journal)
        logger.info(f"Successfully loaded journal with {len(journal.nodes if journal and journal.nodes else [])} nodes from {journal_path}")
        return journal
    except Exception as e:
        logger.error(f"Error loading journal from {journal_path}: {e}", exc_info=True)
        return None

def load_best_solution_code(run_logs_path: Path) -> Optional[str]:
    # Check primary location first (saved by save_run directly into logs/{exp_name})
    code_path_primary = run_logs_path / "best_solution.py"
    # Check secondary location (staged for artifacts by WandbLogger)
    code_path_secondary = run_logs_path / "wandb_artifacts_final" / "best_solution_code" / "solution.py"
    # Check older/alternative staging location (from workspace directly copied)
    code_path_tertiary = run_logs_path / "best_solution_code" / "solution.py" # if 'best_solution' folder was copied
    
    code_path_to_try = None
    if code_path_primary.exists():
        code_path_to_try = code_path_primary
        logger.debug(f"Found best_solution.py at primary location: {code_path_to_try}")
    elif code_path_secondary.exists():
        code_path_to_try = code_path_secondary
        logger.debug(f"Found best_solution.py at artifact staging location: {code_path_to_try}")
    elif code_path_tertiary.exists():
        code_path_to_try = code_path_tertiary
        logger.debug(f"Found best_solution.py at alternative staging location: {code_path_to_try}")
    else:
        logger.warning(f"Best solution code file not found at expected locations in {run_logs_path}")
        return None
    try:
        with open(code_path_to_try, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading best solution code from {code_path_to_try}: {e}", exc_info=True)
        return None
logger = logging.getLogger("aide.metrics_calculator")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

cfg = load_cfg()

def calculate_loc_metric(code: Optional[str]) -> Optional[int]:
    return len(code.splitlines()) if code else None

def calculate_cyclomatic_complexity(code: Optional[str]) -> Optional[float]:
    try:
        from radon.complexity import cc_visit
    except ImportError:
        return None
    if not code:
        return None
    blocks = cc_visit(code)
    comps = [b.complexity for b in blocks if hasattr(b, 'complexity')]
    return float(np.mean(comps)) if comps else 0.0

def calculate_imported_modules(code: Optional[str]) -> Optional[List[str]]:
    if not code:
        return None
    imports = set()
    for line in code.splitlines():
        line=line.strip()
        if line.startswith("import "):
            for part in line.split("import",1)[1].split(","):
                imports.add(part.strip().split()[0])
        elif line.startswith("from "):
            imports.add(line.split()[1].split(".")[0])
    return sorted(imports)

def calculate_tree_and_journal_metrics(journal: Optional[Journal], run_cfg: Any) -> Dict[str, Any]:
    # **unchanged**: tally nodes, depths, branching, successes, times…
    # (you can copy your existing implementation here verbatim)
    # For brevity, assume it returns a dict `metrics`.
    metrics: Dict[str, Any] = {}
    # … your existing logic …
    return metrics

def aggregate_wandb_history_metrics(history_df: pd.DataFrame, steps: int) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if history_df is None or history_df.empty:
        return {"status": "no history.csv"}
    metrics["avg_exec_time_per_step_secs_wb"] = history_df.get('exec/exec_time_s', pd.Series()).mean()
    metrics["total_buggy_steps_wb"] = int(history_df.get('eval/is_buggy', pd.Series()).sum())
    if steps>0:
        good = (history_df['eval/is_buggy']==0).sum()
        metrics["vcgr_percent_wb"] = good/steps*100
        sub = (history_df.get('eval/submission_produced_this_step',pd.Series())==1).sum()
        metrics["csar_percent_wb"] = sub/steps*100
    metrics["total_effective_fixes_wb"] = int(history_df.get('eval/effective_fix_this_step',pd.Series()).sum())
    metrics["total_effective_reflection_fixes_wb"] = int(history_df.get('eval/effective_reflection_fix_this_step',pd.Series()).sum())
    # errors:
    if 'exec/exception_type' in history_df:
        errs = history_df['exec/exception_type'].replace("None", np.nan).dropna()
        cnt = errs.value_counts().to_dict()
        metrics["error_type_distribution_wb"] = cnt
        metrics["most_frequent_error_type_wb"] = next(iter(cnt)) if cnt else None
    return metrics

def extract_wandb_summary_metrics(summary_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "steps_to_first_wo_wb_summary": summary_dict.get("summary/steps_to_first_working_code"),
        "best_validation_metric_wb_summary": summary_dict.get("summary/best_validation_metric_overall"),
        "total_submissions_by_non_buggy_nodes_wb_summary": summary_dict.get("summary/total_submissions_by_non_buggy_nodes"),
        "total_gold_medals_achieved_wb_summary": summary_dict.get("summary/total_gold_medals_achieved"),
        "total_silver_medals_achieved_wb_summary": summary_dict.get("summary/total_silver_medals_achieved"),
        "total_bronze_medals_achieved_wb_summary": summary_dict.get("summary/total_bronze_medals_achieved"),
        "total_steps_above_median_wb_summary": summary_dict.get("summary/total_steps_above_median"),
        "avg_code_quality_non_buggy_nodes_wb_summary": summary_dict.get("summary/avg_code_quality_non_buggy_nodes"),
        "total_effective_fixes_in_run_wb_summary": summary_dict.get("summary/total_effective_fixes_in_run"),
        "total_effective_reflection_fixes_in_run_wb_summary": summary_dict.get("summary/total_effective_reflection_fixes_in_run"),
        "best_overall_solution_produced_submission_csv_wb_summary": summary_dict.get("summary/best_overall_solution_produced_submission_csv"),
        "num_working_code_steps_total_wb_summary": summary_dict.get("summary/num_working_code_steps_total"),
        "num_buggy_nodes_total_wb_summary": summary_dict.get("summary/num_buggy_nodes_total"),
    }

def generate_all_metrics(run_name: str, run_id_for_wandb: Optional[str]=None) -> Dict[str, Any]:
    logger.info(f"Generating metrics for run {run_name}")
    run_logs = Path("logs")/run_name
    report: Dict[str, Any] = {
        "run_name": run_name,
        "report_generation_timestamp": time.time()
    }

    # 1) config & journal & code
    run_cfg = load_run_config(run_logs)
    journal = load_run_journal(run_logs)
    code = load_best_solution_code(run_logs)

    report["config_summary"] = {
        "competition_name": getattr(run_cfg, "competition_name", None),
        "agent_strategy": getattr(run_cfg.agent, "ITS_Strategy", None),
        "coder_model": getattr(run_cfg.agent.code, "model", None),
        "planner_model": getattr(run_cfg.agent.code, "planner_model", None),
        "feedback_model": getattr(run_cfg.agent.feedback, "model", None),
        "agent_steps_configured": getattr(run_cfg.agent, "steps", None),
    }

    report["best_solution_code_metrics"] = {
        "loc": calculate_loc_metric(code),
        "cyclomatic_complexity_avg": calculate_cyclomatic_complexity(code),
        "imported_modules": calculate_imported_modules(code),
    }

    report["journal_analysis_metrics"] = calculate_tree_and_journal_metrics(journal, run_cfg)

    # 2) history.csv
    hist_path = run_logs/"history.csv"
    if hist_path.exists():
        df = pd.read_csv(hist_path)
        report["wandb_history_metrics"] = aggregate_wandb_history_metrics(df, run_cfg.agent.steps)
    else:
        report["wandb_history_metrics"] = {"status":"history.csv missing"}

    # 3) wandb_summary_metrics.json
    sum_path = run_logs/"wandb_summary_metrics.json"
    if sum_path.exists():
        summary = json.loads(sum_path.read_text())
        report["wandb_summary_metrics"] = extract_wandb_summary_metrics(summary)
    else:
        report["wandb_summary_metrics"] = {"status":"wandb_summary_metrics.json missing"}

    # 4) cross-check
    jm = report["journal_analysis_metrics"].get("best_metric_value_journal")
    sm = report["wandb_summary_metrics"].get("best_validation_metric_wb_summary")
    if jm is not None and sm is not None:
        try:
            report["cross_check_metrics"] = {
                "consistent": np.isclose(float(jm), float(sm))
            }
        except:
            report["cross_check_metrics"] = {"consistent": False}

    # 5) write out comprehensive report
    out = run_logs/"comprehensive_metrics_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote comprehensive_metrics_report.json to {out}")
    return report

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run_name")
    args = p.parse_args()
    generate_all_metrics(args.run_name)

# # aide/utils/metrics_calculator.py

# import json
# import logging
# import pandas as pd
# import numpy as np
# import re
# import time
# import os
# import shutil
# from pathlib import Path
# import argparse
# from collections import Counter, defaultdict
# from omegaconf import OmegaConf
# from typing import Optional, cast, List, Dict, Any
# import sys
# from . import copytree

# from .config import load_cfg
# import pandas as pd

# # --- Configuration ---
# cfg = load_cfg()
# try:
#     from radon.complexity import cc_visit
#     RADON_AVAILABLE = True
#     logger_radon = logging.getLogger("radon") 
#     logger_radon.setLevel(logging.CRITICAL) 
# except ImportError:
#     RADON_AVAILABLE = False
#     logger_radon = None # Define to avoid UnboundLocalError if radon is not available


# try:
#     import wandb
#     WANDB_AVAILABLE = True
# except ImportError:
#     WANDB_AVAILABLE = False
#     wandb = None #
    

# try:
#     from ..journal import Journal, Node, filter_journal, get_path_to_node
#     from .config import Config # For type hinting config
#     from . import serialize
#     from .metric import MetricValue, WorstMetricValue 
# except ImportError: 
#     print("Warning: Could not perform relative imports for metrics_calculator.py. Ensure script is run as a module or PYTHONPATH is set.")
#     sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) 
#     from aide.journal import Journal, Node, filter_journal, get_path_to_node
#     from aide.utils.config import Config
#     from aide.utils import serialize
#     from aide.utils.metric import MetricValue, WorstMetricValue

# logger = logging.getLogger("aide.metrics_calculator")
# if not logger.handlers: # Setup basic logging if run standalone
#     handler = logging.StreamHandler(sys.stdout)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)


# # --- Configuration Constants (can be overridden by args in __main__) ---
# DEFAULT_WANDB_ENTITY = "asim_awad"
# DEFAULT_WANDB_PROJECT = "MLE_BENCH_AIDE"
# DEFAULT_MAX_API_RETRIES = 5
# DEFAULT_API_RETRY_DELAY = 10

# WANDB_ENTITY = "asim_awad"

# WANDB_PROJECT = "MLE_BENCH_AIDE_VM"

# WANDB_RUN_NAME = cfg.wandb.run_name

# DOWNLOAD_DIR = "./logs"

# FILE_FILTER_PATTERN = cfg.wandb.run_name  # solve me



# def copy_best_solution_and_submission():
#     workspaces_dir = os.path.join("workspaces", cfg.exp_name)
#     logs_dir = Path(os.path.join("logs", cfg.exp_name))
#     best_submission_dir = Path(os.path.join(workspaces_dir, "best_submission"))

#     if os.path.exists(best_submission_dir):
#         copytree(best_submission_dir, logs_dir, use_symlinks=False)
        
#         shutil.copy(os.path.join(best_submission_dir, "submission.csv"), os.path.join(logs_dir, "submission.csv"))
#         print(f"Copied best_submission directory to {logs_dir}")
#     else:
#         print(f"best_submission directory not found in {workspaces_dir}")


# def save_logs_to_wandb():
#     copy_best_solution_and_submission()

#     print("Saving logs directory to WandB")
#     wandb.save(f"logs/{cfg.exp_name}/*", base_path="logs")  # Save log files


# # --- Data Loading Functions ---
# def load_run_config(run_logs_path: Path) -> Optional[Config]:
#     config_path = run_logs_path / "config.yaml"
#     if not config_path.exists():
#         logger.error(f"Config file not found at {config_path}")
#         return None
#     try:
#         cfg_loaded = OmegaConf.load(config_path)
#         if not (hasattr(cfg_loaded, 'agent') and hasattr(cfg_loaded.agent, 'code')):
#              logger.error(f"Loaded config from {config_path} does not appear to be a valid AIDE Config structure.")
#              return None
#         return cast(Config, cfg_loaded) 
#     except Exception as e:
#         logger.error(f"Error loading config from {config_path}: {e}", exc_info=True)
#         return None

# def load_run_journal(run_logs_path: Path) -> Optional[Journal]:
#     journal_path = run_logs_path / "journal.json"
#     if not journal_path.exists():
#         logger.error(f"Journal file not found at {journal_path}")
#         return None
#     try:
#         journal = serialize.load_json(journal_path, Journal)
#         logger.info(f"Successfully loaded journal with {len(journal.nodes if journal and journal.nodes else [])} nodes from {journal_path}")
#         return journal
#     except Exception as e:
#         logger.error(f"Error loading journal from {journal_path}: {e}", exc_info=True)
#         return None

# def load_best_solution_code(run_logs_path: Path) -> Optional[str]:
#     # Check primary location first (saved by save_run directly into logs/{exp_name})
#     code_path_primary = run_logs_path / "best_solution.py"
#     # Check secondary location (staged for artifacts by WandbLogger)
#     code_path_secondary = run_logs_path / "wandb_artifacts_final" / "best_solution_code" / "solution.py"
#     # Check older/alternative staging location (from workspace directly copied)
#     code_path_tertiary = run_logs_path / "best_solution_code" / "solution.py" # if 'best_solution' folder was copied
    
#     code_path_to_try = None
#     if code_path_primary.exists():
#         code_path_to_try = code_path_primary
#         logger.debug(f"Found best_solution.py at primary location: {code_path_to_try}")
#     elif code_path_secondary.exists():
#         code_path_to_try = code_path_secondary
#         logger.debug(f"Found best_solution.py at artifact staging location: {code_path_to_try}")
#     elif code_path_tertiary.exists():
#         code_path_to_try = code_path_tertiary
#         logger.debug(f"Found best_solution.py at alternative staging location: {code_path_to_try}")
#     else:
#         logger.warning(f"Best solution code file not found at expected locations in {run_logs_path}")
#         return None
#     try:
#         with open(code_path_to_try, "r", encoding="utf-8") as f:
#             return f.read()
#     except Exception as e:
#         logger.error(f"Error loading best solution code from {code_path_to_try}: {e}", exc_info=True)
#         return None

# def fetch_wandb_run_data(entity: str, project: str, run_identifier: str, 
#                          max_retries: int = DEFAULT_MAX_API_RETRIES, 
#                          retry_delay: int = DEFAULT_API_RETRY_DELAY) -> Optional[Dict[str, Any]]:
#     if not WANDB_AVAILABLE:
#         logger.warning("wandb library not available. Skipping W&B data fetching.")
#         return None
    
#     api = wandb.Api(timeout=30)
#     target_run_obj = None
    
#     logger.info(f"Attempting to fetch W&B run: {entity}/{project}/{run_identifier}")
#     for attempt in range(max_retries):
#         try:
#             target_run_obj = api.run(f"{entity}/{project}/{run_identifier}") # Assumes run_identifier is ID or path
#             logger.info(f"Successfully fetched W&B run '{target_run_obj.name}' (ID: {target_run_obj.id}) on attempt {attempt + 1}.")
#             break
#         except wandb.errors.CommError as e:
#             logger.warning(f"Attempt {attempt + 1}/{max_retries} to fetch W&B run '{run_identifier}' failed with CommError: {e}")
#             if "Could not find run" in str(e) and attempt == 0 and not re.match(r"^[a-zA-Z0-9]{8,}$", run_identifier):
#                 # If "not found" and it doesn't look like an ID, try display_name once
#                 logger.info(f"Run ID lookup failed. Attempting to find by display_name: {run_identifier}")
#                 try:
#                     runs = api.runs(f"{entity}/{project}", filters={"display_name": run_identifier})
#                     if runs:
#                         if len(runs) > 1: logger.warning(f"Multiple runs found for display_name '{run_identifier}'. Using most recent.")
#                         target_run_obj = sorted(runs, key=lambda r: r.createdAt, reverse=True)[0]
#                         logger.info(f"Successfully fetched W&B run '{target_run_obj.name}' (ID: {target_run_obj.id}) by display_name.")
#                         break 
#                 except Exception as e_disp:
#                     logger.warning(f"Attempt to fetch by display_name '{run_identifier}' also failed: {e_disp}")
#             if attempt < max_retries - 1: time.sleep(retry_delay)
#             else: logger.error(f"Failed to fetch W&B run '{run_identifier}' after {max_retries} attempts.")
#         except Exception as e_gen:
#              logger.warning(f"Attempt {attempt + 1}/{max_retries} to fetch W&B run '{run_identifier}' failed with generic API error: {e_gen}")
#              if attempt < max_retries - 1: time.sleep(retry_delay)
#              else: logger.error(f"Failed to fetch W&B run '{run_identifier}' after {max_retries} attempts due to generic API error.")

#     if not target_run_obj:
#         return None

#     try:
#         hist = target_run_obj.history(pandas=True)
#         hist.to_csv(f"{DOWNLOAD_DIR}/{cfg.exp_name}/history.csv", index=False)
#     except Exception as e:
#         print(f"the history is not retrieved")
#     try:
#         summary_dict = dict(target_run_obj.summary) 
#         config_dict = dict(target_run_obj.config)
#     except Exception as e:
#         logger.error(f"Error processing data for W&B run '{target_run_obj.id}': {e}", exc_info=True)
#         print(f"the summary is not retrieved")
#         return None
#     return {
#             "wandb_run_obj": target_run_obj, # For potential direct access if needed
#             "history_df": hist,
#             "summary_dict": summary_dict,
#             "config_dict": config_dict
#         }

# # --- Metric Calculation Functions ---
# def calculate_loc_metric(code_content: Optional[str]) -> Optional[int]:
#     if code_content is None: return None
#     return len(code_content.splitlines())

# def calculate_cyclomatic_complexity(code_content: Optional[str]) -> Optional[float]:
#     if not RADON_AVAILABLE or code_content is None:
#         if not RADON_AVAILABLE: logger.debug("Radon library not available, skipping cyclomatic complexity.")
#         return None
#     try:
#         blocks = cc_visit(code_content)
#         complexities = [block.complexity for block in blocks if hasattr(block, 'complexity')]
#         return np.mean(complexities) if complexities else 0.0
#     except Exception as e:
#         logger.warning(f"Could not calculate cyclomatic complexity for a code block: {e}")
#         return None

# def calculate_imported_modules(code_content: Optional[str]) -> Optional[List[str]]:
#     if code_content is None: return None
#     imports = set()
#     for line in code_content.splitlines():
#         line = line.strip()
#         if line.startswith("import "):
#             parts = line.split("import ")[1]
#             for part in parts.split(','):
#                 imports.add(part.split(" as ")[0].strip().split('.')[0])
#         elif line.startswith("from "):
#             match = re.match(r"from\s+([_A-Za-z0-9\.]+)\s+import", line)
#             if match: imports.add(match.group(1).split('.')[0])
#     return sorted(list(imports))

# def calculate_tree_and_journal_metrics(journal: Optional[Journal], run_cfg: Optional[Config]) -> Dict[str, Any]:
#     metrics: Dict[str, Any] = {} 
#     default_journal_metrics = {
#         "total_nodes": 0, "max_tree_depth": 0, "avg_branching_factor": 0.0,
#         "num_draft_nodes": 0, "num_improve_nodes": 0, "num_debug_nodes": 0,
#         "num_buggy_nodes": 0, "num_non_buggy_nodes": 0,
#         "debug_attempt_rate_on_buggy_nodes": 0.0, "successful_debug_rate_among_attempts": 0.0,
#         "avg_successful_debug_chain_length": 0.0,
#         "num_effective_debug_steps_journal": 0,
#         "num_effective_reflection_fixes_journal": 0,
#         "time_to_first_wo_secs_journal": None, "steps_to_first_wo_journal": None,
#         "time_to_best_solution_secs_journal": None, "steps_to_best_solution_journal": None,
#         "best_metric_value_journal": None, "best_metric_node_id_journal": None,
#         "avg_code_quality_all_nodes": None, "avg_code_quality_non_buggy_nodes": None,
#         "code_quality_of_best_node_journal": None
#     }
#     if not journal or not journal.nodes:
#         logger.warning("Journal is None or has no nodes. Returning default/empty journal metrics.")
#         return default_journal_metrics

#     nodes = journal.nodes
#     metrics["total_nodes"] = len(nodes)

#     max_depth_val = 0
#     if nodes:
#         node_depths = {} 
#         root_nodes_ids = [n.id for n in nodes if n.parent is None]
#         for root_id in root_nodes_ids:
#             queue = [(root_id, 1)]; visited_bfs = {root_id}; node_depths[root_id] = 1
#             max_depth_val = max(max_depth_val, 1)
#             head = 0
#             while head < len(queue):
#                 curr_id, depth = queue[head]; head += 1
#                 curr_node_obj = next((n_obj for n_obj in nodes if n_obj.id == curr_id), None)
#                 if curr_node_obj:
#                     for child_node_obj in curr_node_obj.children: 
#                         child_id = child_node_obj.id
#                         if child_id not in visited_bfs:
#                             visited_bfs.add(child_id); node_depths[child_id] = depth + 1
#                             max_depth_val = max(max_depth_val, depth + 1)
#                             queue.append((child_id, depth + 1))
#     metrics["max_tree_depth"] = max_depth_val
    
#     parent_child_counts = [len(n.children) for n in nodes if n.children] 
#     metrics["avg_branching_factor"] = np.mean(parent_child_counts) if parent_child_counts else 0.0

#     metrics["num_draft_nodes"] = len(journal.draft_nodes)
#     metrics["num_improve_nodes"] = len([n for n in nodes if n.stage_name == "improve"])
#     metrics["num_debug_nodes"] = len([n for n in nodes if n.stage_name == "debug"])
#     metrics["num_buggy_nodes"] = len(journal.buggy_nodes)
#     metrics["num_non_buggy_nodes"] = len(journal.good_nodes)

#     buggy_node_ids_set = {n.id for n in journal.buggy_nodes}
#     debug_nodes_list = [n for n in nodes if n.stage_name == "debug"] 
#     debug_attempts_on_buggy_nodes = 0; successful_debugs = 0
#     debug_chain_lengths_successful: List[int] = []

#     for dn in debug_nodes_list:
#         if dn.parent and dn.parent.id in buggy_node_ids_set: 
#             debug_attempts_on_buggy_nodes += 1
#             if not dn.is_buggy: 
#                 successful_debugs += 1
#                 path = get_path_to_node(journal, dn.id)
#                 chain_len = 0
#                 for node_id_in_path in reversed(path):
#                     node_in_path = next((n_obj for n_obj in journal.nodes if n_obj.id == node_id_in_path), None)
#                     if node_in_path and node_in_path.stage_name == "debug": chain_len +=1
#                     else: break
#                 debug_chain_lengths_successful.append(chain_len)

#     metrics["debug_attempt_rate_on_buggy_nodes"] = (debug_attempts_on_buggy_nodes / len(buggy_node_ids_set)) * 100 if buggy_node_ids_set else 0.0
#     metrics["successful_debug_rate_among_attempts"] = (successful_debugs / debug_attempts_on_buggy_nodes) * 100 if debug_attempts_on_buggy_nodes > 0 else 0.0
#     metrics["avg_successful_debug_chain_length"] = np.mean(debug_chain_lengths_successful) if debug_chain_lengths_successful else 0.0
    
#     metrics["num_effective_debug_steps_journal"] = sum(1 for n in nodes if hasattr(n, 'effective_debug_step') and n.effective_debug_step)
#     metrics["num_effective_reflection_fixes_journal"] = sum(1 for n in nodes if hasattr(n, 'effective_reflections') and n.effective_reflections)
    
#     first_wo_node_time_val = float('inf'); best_node_time_val = float('inf')
#     first_wo_node_step_val = -1; best_node_step_val = -1
#     best_metric_obj: Optional[MetricValue] = None 
#     best_metric_node_id_local_val: Optional[str] = None
#     run_start_time_val = min(n.ctime for n in nodes) if nodes else time.time()

#     for node in sorted(nodes, key=lambda n: n.step):
#         if not node.is_buggy:
#             if node.ctime < first_wo_node_time_val:
#                 first_wo_node_time_val = node.ctime
#                 first_wo_node_step_val = node.step
            
#             if node.metric is not None and not isinstance(node.metric, WorstMetricValue): 
#                 if best_metric_obj is None or node.metric > best_metric_obj: # MetricValue comparison
#                     best_metric_obj = node.metric
#                     best_node_time_val = node.ctime
#                     best_node_step_val = node.step
#                     best_metric_node_id_local_val = node.id

#     metrics["time_to_first_wo_secs_journal"] = (first_wo_node_time_val - run_start_time_val) if first_wo_node_time_val != float('inf') else None
#     metrics["steps_to_first_wo_journal"] = first_wo_node_step_val + 1 if first_wo_node_step_val != -1 else None
#     metrics["time_to_best_solution_secs_journal"] = (best_node_time_val - run_start_time_val) if best_node_time_val != float('inf') else None
#     metrics["steps_to_best_solution_journal"] = best_node_step_val + 1 if best_node_step_val != -1 else None
#     metrics["best_metric_value_journal"] = best_metric_obj.value if best_metric_obj and best_metric_obj.value is not None else None
#     metrics["best_metric_node_id_journal"] = best_metric_node_id_local_val

#     all_code_qualities = [n.code_quality for n in nodes if hasattr(n, 'code_quality') and n.code_quality is not None]
#     non_buggy_code_qualities = [n.code_quality for n in journal.good_nodes if hasattr(n, 'code_quality') and n.code_quality is not None]
#     metrics["avg_code_quality_all_nodes"] = np.mean(all_code_qualities) if all_code_qualities else None
#     metrics["avg_code_quality_non_buggy_nodes"] = np.mean(non_buggy_code_qualities) if non_buggy_code_qualities else None
#     if best_metric_node_id_local_val:
#         best_node_obj_val = next((n for n in nodes if n.id == best_metric_node_id_local_val), None)
#         metrics["code_quality_of_best_node_journal"] = best_node_obj_val.code_quality if best_node_obj_val and hasattr(best_node_obj_val, 'code_quality') else None
    
#     return metrics

# def aggregate_wandb_history_metrics(history_df: Optional[pd.DataFrame], agent_steps_config: int) -> Dict[str, Any]:
#     metrics: Dict[str, Any] = {}
#     default_history_metrics = {
#         "avg_exec_time_per_step_secs_wb": None, "total_buggy_steps_wb": None,
#         "vcgr_percent_wb": None, "csar_percent_wb": None,
#         "total_effective_fixes_wb": None, "total_effective_reflection_fixes_wb": None,
#         "error_type_distribution_wb": None, "most_frequent_error_type_wb": None
#     }
#     if history_df is None or history_df.empty:
#         logger.warning("W&B history_df is missing or empty. Returning default W&B history metrics.")
#         return default_history_metrics

#     metrics["avg_exec_time_per_step_secs_wb"] = history_df['exec/exec_time_s'].mean() if 'exec/exec_time_s' in history_df.columns else None
#     metrics["total_buggy_steps_wb"] = int(history_df['eval/is_buggy'].sum()) if 'eval/is_buggy' in history_df.columns else None
    
#     if 'eval/is_buggy' in history_df.columns and agent_steps_config > 0:
#         valid_code_count = (history_df["eval/is_buggy"] == 0).sum()
#         metrics["vcgr_percent_wb"] = (valid_code_count / agent_steps_config) * 100
#     else:
#         metrics["vcgr_percent_wb"] = 0.0 if agent_steps_config > 0 else None # Default to 0% if steps exist but no buggy info

#     # Key for submission produced by non-buggy code THIS STEP
#     submission_key = "eval/submission_produced_this_step" 
#     if submission_key in history_df.columns and agent_steps_config > 0:
#         submission_produced_count = (history_df[submission_key] == 1).sum()
#         metrics["csar_percent_wb"] = (submission_produced_count / agent_steps_config) * 100
#     else:
#         metrics["csar_percent_wb"] = 0.0 if agent_steps_config > 0 else None
        
#     metrics["total_effective_fixes_wb"] = int(history_df['eval/effective_fix_this_step'].sum()) if 'eval/effective_fix_this_step' in history_df.columns else None
#     metrics["total_effective_reflection_fixes_wb"] = int(history_df['eval/effective_reflection_fix_this_step'].sum()) if 'eval/effective_reflection_fix_this_step' in history_df.columns else None # Key name from agent step

#     if 'exec/exception_type' in history_df.columns:
#         error_series = history_df['exec/exception_type'].replace("None", np.nan).dropna()
#         if not error_series.empty:
#             error_counts = Counter(error_series.tolist())
#             metrics["error_type_distribution_wb"] = dict(error_counts)
#             metrics["most_frequent_error_type_wb"] = error_counts.most_common(1)[0][0] if error_counts else None
#         else:
#             metrics["error_type_distribution_wb"] = {}
#             metrics["most_frequent_error_type_wb"] = None
#     else:
#         metrics["error_type_distribution_wb"] = None
#         metrics["most_frequent_error_type_wb"] = None
        
#     return metrics



# def extract_wandb_summary_metrics(summary_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#     default_summary = {
#         "steps_to_first_wo_wb_summary": None,
#         "best_validation_metric_wb_summary": None,
#         "total_submissions_by_non_buggy_nodes_wb_summary": None,
#         "total_gold_medals_achieved_wb_summary": None,
#         "total_silver_medals_achieved_wb_summary": None,
#         "total_bronze_medals_achieved_wb_summary": None,
#         "total_steps_above_median_wb_summary": None,
#         "avg_code_quality_non_buggy_nodes_wb_summary": None,
#         "total_effective_fixes_in_run_wb_summary": None, # Mapped from "summary/total_effective_fixes_in_run"
#         "total_effective_reflection_fixes_in_run_wb_summary": None, # Mapped from "summary/total_effective_reflection_fixes_in_run"
#         "best_overall_solution_produced_submission_csv_wb_summary": None, # Mapped from "summary/best_overall_solution_produced_submission_csv"
#         # Add any other keys you expect to be in wandb.summary that you want in this report
#         "num_working_code_steps_total_wb_summary": None,
#         "num_buggy_nodes_total_wb_summary": None,
#     }
#     if summary_dict is None:
#         logger.warning("W&B summary_dict is missing. Returning default W&B summary metrics.")
#         return default_summary
    
#     # Extract with .get() to gracefully handle missing keys
#     return {
#         "steps_to_first_wo_wb_summary": summary_dict.get("summary/steps_to_first_working_code"),
#         "best_validation_metric_wb_summary": summary_dict.get("summary/best_validation_metric_overall"),
#         "total_submissions_by_non_buggy_nodes_wb_summary": summary_dict.get("summary/total_submissions_by_non_buggy_nodes"),
#         "total_gold_medals_achieved_wb_summary": summary_dict.get("summary/total_gold_medals_achieved"),
#         "total_silver_medals_achieved_wb_summary": summary_dict.get("summary/total_silver_medals_achieved"),
#         "total_bronze_medals_achieved_wb_summary": summary_dict.get("summary/total_bronze_medals_achieved"),
#         "total_steps_above_median_wb_summary": summary_dict.get("summary/total_steps_above_median"),
#         "avg_code_quality_non_buggy_nodes_wb_summary": summary_dict.get("summary/avg_code_quality_non_buggy_nodes"),
#         "total_effective_fixes_in_run_wb_summary": summary_dict.get("summary/total_effective_fixes_in_run"),
#         "total_effective_reflection_fixes_in_run_wb_summary": summary_dict.get("summary/total_effective_reflection_fixes_in_run"),
#         "best_overall_solution_produced_submission_csv_wb_summary": summary_dict.get("summary/best_overall_solution_produced_submission_csv"),
#         "num_working_code_steps_total_wb_summary": summary_dict.get("summary/num_working_code_steps_total"),
#         "num_buggy_nodes_total_wb_summary": summary_dict.get("summary/num_buggy_nodes_total"),
#     }

# # ... (rest of metrics_calculator.py) ...

# # --- Main Orchestration ---
# def generate_all_metrics(run_name: str, run_id_for_wandb: Optional[str] = None) -> Dict[str, Any]:
#     logger.info(f"Starting comprehensive metrics generation for run: {run_name}")
#     run_logs_path = Path("logs") / run_name
#     all_metrics_report: Dict[str, Any] = {"run_name": run_name, "report_generation_timestamp": time.time()}

#     run_cfg = load_run_config(run_logs_path)
#     journal = load_run_journal(run_logs_path)
#     best_code_content = load_best_solution_code(run_logs_path)

#     if run_cfg:
#         all_metrics_report["config_summary"] = {
#             "competition_name": run_cfg.competition_name,
#             "agent_strategy": run_cfg.agent.ITS_Strategy,
#             "coder_model": run_cfg.agent.code.model,
#             "planner_model": run_cfg.agent.code.planner_model if hasattr(run_cfg.agent.code, 'planner_model') else None,
#             "feedback_model": run_cfg.agent.feedback.model,
#             "agent_steps_configured": run_cfg.agent.steps
#         }
#     else:
#         all_metrics_report["config_summary"] = {"error": "Could not load run_cfg from local logs."}

#     all_metrics_report["best_solution_code_metrics"] = {
#         "loc": calculate_loc_metric(best_code_content),
#         "cyclomatic_complexity_avg": calculate_cyclomatic_complexity(best_code_content),
#         "imported_modules": calculate_imported_modules(best_code_content)
#     }

#     all_metrics_report["journal_analysis_metrics"] = calculate_tree_and_journal_metrics(journal, run_cfg)
    
#     wandb_data = None
#     if WANDB_AVAILABLE and run_cfg and hasattr(run_cfg, 'wandb') and run_cfg.wandb.enabled:
#         entity = run_cfg.wandb.entity or os.getenv("WANDB_ENTITY_OVERRIDE", DEFAULT_WANDB_ENTITY)
#         project = run_cfg.wandb.project or os.getenv("WANDB_PROJECT_OVERRIDE", DEFAULT_WANDB_PROJECT)
#         identifier_for_wandb = run_id_for_wandb if run_id_for_wandb else run_name
        
#         logger.info(f"Fetching W&B data for {entity}/{project}, identifier: {identifier_for_wandb}")
#         wandb_data = fetch_wandb_run_data(entity, project, identifier_for_wandb) 
    
#     if wandb_data and run_cfg: # run_cfg must exist to get agent_steps_config
#         all_metrics_report["wandb_history_metrics"] = aggregate_wandb_history_metrics(
#             wandb_data.get("history_df"), 
#             run_cfg.agent.steps 
#         )
#         all_metrics_report["wandb_summary_metrics"] = extract_wandb_summary_metrics(wandb_data.get("summary_dict"))
        
#         journal_best_metric = all_metrics_report["journal_analysis_metrics"].get("best_metric_value_journal")
#         wb_summary_best_metric = all_metrics_report["wandb_summary_metrics"].get("best_validation_metric_wb_summary")
#         if journal_best_metric is not None and wb_summary_best_metric is not None:
#             # Ensure both are float for np.isclose if they are numbers
#             try:
#                 jbm_float = float(journal_best_metric)
#                 wbm_float = float(wb_summary_best_metric)
#                 consistent = np.isclose(jbm_float, wbm_float)
#             except (ValueError, TypeError):
#                 consistent = "TypeMismatch" # Or False

#             all_metrics_report["cross_check_metrics"] = {
#                 "best_metric_journal_vs_wb_summary": {
#                     "journal_value": journal_best_metric,
#                     "wandb_summary_value": wb_summary_best_metric,
#                     "consistent": consistent
#                 }
#             }
#     else:
#         all_metrics_report["wandb_history_metrics"] = {"status": "W&B data not fetched or run_cfg not available."}
#         all_metrics_report["wandb_summary_metrics"] = {"status": "W&B data not fetched or run_cfg not available."}

#     logger.info("Comprehensive metrics generation complete.")
    
#     output_path = run_logs_path / "comprehensive_metrics_report.json"
#     try:
#         with open(output_path, "w", encoding="utf-8") as f:
#             class NpEncoder(json.JSONEncoder): 
#                 def default(self, obj):
#                     if isinstance(obj, np.integer): return int(obj)
#                     if isinstance(obj, np.floating): return float(obj)
#                     if isinstance(obj, np.ndarray): return obj.tolist()
#                     if isinstance(obj, (Path, OmegaConf)): return str(obj)
#                     if isinstance(obj, (MetricValue)): return obj.to_dict()
#                     return super(NpEncoder, self).default(obj)
#             json.dump(all_metrics_report, f, indent=4, cls=NpEncoder)
#         logger.info(f"Comprehensive metrics report saved to: {output_path}")
#     except Exception as e:
#         logger.error(f"Error saving metrics report to JSON: {e}", exc_info=True)

#     return all_metrics_report

# # --- Main Execution (for standalone script) ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate comprehensive metrics for an AIDE agent run.")
#     parser.add_argument("run_name", type=str, help="The name of the run (directory name in ./logs/)")
#     parser.add_argument("--run_id", type=str, help="Optional W&B run ID if known, for more precise W&B data fetching.")
#     parser.add_argument("--entity", type=str, default=DEFAULT_WANDB_ENTITY, help="W&B entity.")
#     parser.add_argument("--project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project.")

#     args = parser.parse_args()


#     final_metrics_report = generate_all_metrics(args.run_name, run_id_for_wandb=args.run_id)

#     console_output = { 
#         "Run Name": final_metrics_report.get("run_name"),
#         "Config Summary": final_metrics_report.get("config_summary", {}),
#         "Key Journal Metrics": {
#             k: final_metrics_report.get("journal_analysis_metrics", {}).get(k) for k in [
#                 "total_nodes", "best_metric_value_journal", "steps_to_first_wo_journal"
#             ]
#         },
#         "Key W&B History Metrics": {
#             k: final_metrics_report.get("wandb_history_metrics", {}).get(k) for k in [
#                 "vcgr_percent_wb", "csar_percent_wb", "avg_exec_time_per_step_secs_wb"
#             ]
#         },
#         "Key W&B Summary Metrics": {
#              k: final_metrics_report.get("wandb_summary_metrics", {}).get(k) for k in [
#                 "best_validation_metric_wb_summary", "steps_to_first_wo_wb_summary",
#                 "total_gold_medals_achieved_wb_summary", # Example of adding a medal count
#             ]
#         },
#         "Best Solution LOC": final_metrics_report.get("best_solution_code_metrics", {}).get("loc")
#     }
#     try:
#         import yaml 
#         print("\n--- Metrics Calculator Summary ---")
#         print(yaml.dump(console_output, sort_keys=False, indent=2, allow_unicode=True))
#     except ImportError:
#         print("\n--- Metrics Calculator Summary (JSON fallback) ---")
#         print(json.dumps(console_output, indent=2))