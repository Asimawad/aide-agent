# aide/run.py
import atexit
import logging
import shutil
import sys
import pandas as pd
import os
from pathlib import Path 
from .utils import load_benchmarks 
import time
from rich.console import Console
from rich.logging import RichHandler 
import wandb
console = Console() 


from .utils import copytree 
from aide.utils.metrics_calculator import generate_all_metrics
from .utils.wandb_logger import WandbLogger 
from .agent import Agent, PlannerAgent, CodeChainAgent, SelfConsistencyAgent, SelfDebugAgent
from .interpreter import Interpreter
from .journal import Journal, Node 
from omegaconf import OmegaConf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.tree import Tree 
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

class VerboseFilter(logging.Filter):
    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)

# journal_to_rich_tree and journal_to_string_tree remain the same
def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()
    def append_rec(node: Node, tree):
        if node.is_buggy: s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            metric_display = f"{node.metric.value:.3f}" if node.metric and node.metric.value is not None else "N/A"
            if node is best_node: s = f"[{style}green]● {metric_display} (best)"
            else: s = f"[{style}green]● {metric_display}"
        subtree = tree.add(s)
        for child in node.children: append_rec(child, subtree)
    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes: append_rec(n, tree)
    return tree

def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"
    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy: s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            markers = []
            if node is best_node: markers.append("best")
            marker_str = " & ".join(markers)
            metric_display = f"{node.metric.value:.3f}" if node.metric and node.metric.value is not None else "N/A"
            if marker_str: s = f"{indent}● {metric_display} ({marker_str}) (ID: {node.id})\n"
            else: s = f"{indent}● {metric_display} (ID: {node.id})\n"
        tree_str += s
        for child in node.children: append_rec(child, level + 1)
    for n in journal.draft_nodes: append_rec(n, 0)
    return tree_str

def run():
    cfg = load_cfg()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # --- Logger Setup ---
    logger = logging.getLogger("aide") 
    logger.setLevel(logging.DEBUG) 
    logger.handlers.clear() 
    logger.propagate = False 
    log_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log", mode='w')
    verbose_file_handler.setFormatter(formatter)
    verbose_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(verbose_file_handler)
    normal_file_handler = logging.FileHandler(cfg.log_dir / "aide.log", mode='w')
    normal_file_handler.setFormatter(formatter)
    normal_file_handler.setLevel(logging.INFO) 
    normal_file_handler.addFilter(VerboseFilter()) 
    logger.addHandler(normal_file_handler)
    rich_console_handler = RichHandler(
        console=console, level=getattr(logging, cfg.log_level.upper(), logging.INFO), 
        show_path=False, show_level=True, show_time=True, markup=True, rich_tracebacks=True,
        log_time_format="[%X]",
    )
    rich_console_handler.setFormatter(logging.Formatter("%(message)s")) 
    rich_console_handler.addFilter(VerboseFilter()) 
    logger.addHandler(rich_console_handler)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)
    logger.info(f"Logging initialized. verbose.log level: DEBUG, aide.log level: INFO, Console level: {cfg.log_level.upper()}.")
    logger.info(f'Starting run "{cfg.exp_name}"')
    # --- End Logger Setup ---

    competition_benchmarks = None
    try:
        competition_benchmarks = load_benchmarks(cfg.competition_name)
        logger.info(f"Loaded benchmarks for competition: {cfg.competition_name}")
    except KeyError:
        logger.warning(f"No benchmarks found for competition: {cfg.competition_name}. Medal/threshold logging might be affected.")
    except Exception as e:
        logger.error(f"Error loading benchmarks for {cfg.competition_name}: {e}", exc_info=True)

    wandb_logger_instance = WandbLogger(cfg, logger, competition_benchmarks) 
    wandb_logger_instance.init_wandb()

    task_desc = load_task_desc(cfg)

    with Status("[blue]Preparing agent workspace...", console=console):
        prep_agent_workspace(cfg)
    
    global_step = 0 
    def cleanup():
        if global_step == 0 and cfg.workspace_dir.exists(): 
            logger.info(f"Cleaning up workspace as no steps were run: {cfg.workspace_dir}")
            shutil.rmtree(cfg.workspace_dir)
    atexit.register(cleanup)
    
    journal = Journal()
    
    agent_instance_args = {
        "task_desc": task_desc, "cfg": cfg, "journal": journal,
        "wandb_logger": wandb_logger_instance, 
        "competition_benchmarks": competition_benchmarks,
    }
    if cfg.agent.ITS_Strategy == "planner":
        logger.info("Initializing PlannerAgent.")
        agent = PlannerAgent(**agent_instance_args)
    elif cfg.agent.ITS_Strategy == "codechain" or cfg.agent.ITS_Strategy == "codechain_v2" or cfg.agent.ITS_Strategy == "codechain_v3":
        logger.info("Initializing CodeChainAgent.")
        agent = CodeChainAgent(**agent_instance_args)
    elif cfg.agent.ITS_Strategy == "self-debug":
        logger.info("Initializing SelfDebugAgent.")
        agent = SelfDebugAgent(**agent_instance_args)
    elif cfg.agent.ITS_Strategy == "self-consistency":
        logger.info("Initializing SelfConsistencyAgent.")
        agent = SelfConsistencyAgent(**agent_instance_args)
    else:
        logger.info("Initializing Agent.")
        agent = Agent(**agent_instance_args)

    interpreter = Interpreter(cfg.workspace_dir, **OmegaConf.to_container(cfg.exec))
    
    def exec_callback(*args, **kwargs):
        logger.info("Interpreter: Executing code...") 
        res = interpreter.run(*args, **kwargs)
        logger.info("Interpreter: Code execution finished.") 
        return res

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console, transient=True 
        ) as progress_bar:
            main_task = progress_bar.add_task("Agent Steps", total=cfg.agent.steps)

            for i in range(cfg.agent.steps):
                current_step_num_display = i + 1
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} START ---")
                t0_step = time.time()
                
                # The agent.step method will internally call wandb_logger_instance.log_step_data
                agent.step(exec_callback=exec_callback, current_step_number=current_step_num_display)
                
                t1_step = time.time()
                logger.info(f"Step {current_step_num_display} processing took {t1_step - t0_step:.2f} seconds.")
                
                if (current_step_num_display % cfg.get('save_every_n_steps', 1)) == 0:
                    logger.info(f"Saving run state at step {current_step_num_display}")
                    save_run(cfg, journal) # This saves journal.json, config.yaml, tree_plot.html locally
                    
                global_step += 1
                progress_bar.update(main_task, advance=1)
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} END (Total time: {t1_step - t0_step:.2f}s) --- \n")
                console.rule(f"End of Step {current_step_num_display}")

        logger.info("All agent steps completed.")
        logger.info("Final solution tree structure:\n" + journal_to_string_tree(journal))
        save_run(cfg, journal) 

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (KeyboardInterrupt).")
        console.print("\n[bold yellow]Run interrupted by user. Saving current state...[/bold yellow]")
    except Exception as e:
        logger.error(f"An unhandled exception occurred during the run: {e}", exc_info=True)
        console.print(f"\n[bold red]An error occurred: {e}[/bold red]")

    finally:
        logger.info("Interpreter cleanup initiated.")
        interpreter.cleanup_session()


        active_wandb_run_id = None
        if wandb_logger_instance and wandb_logger_instance.wandb_run:
            try:
                active_wandb_run_id = wandb_logger_instance.wandb_run.id
                if active_wandb_run_id:
                    logger.info(f"Successfully obtained active W&B run ID for metrics calculation: {active_wandb_run_id}")
                else: # Should not happen if .id exists and is not None
                    logger.warning("W&B run object exists but ID is None or empty.")
            except AttributeError: 
                logger.warning("W&B run object (wandb_logger_instance.wandb_run) is None or lacks 'id' attribute. Cannot pass ID to metrics calculator.")
            except Exception as e_get_id: # Catch any other unexpected error
                logger.error(f"Unexpected error getting W&B run ID: {e_get_id}", exc_info=True)
        else:
            logger.info("WandbLogger or its run object is not available. Proceeding without W&B run ID for metrics calculation.")


        # --- Step 2: Finalize the W&B run ---
        logger.info("W&B finalization initiated.")
        if wandb_logger_instance: 
            wandb_logger_instance.finalize_run(journal) 

        try:
            logger.info("Calculating comprehensive metrics locally (before W&B finalization)...")
            generate_all_metrics(
                cfg.exp_name, 
                run_id_for_wandb=active_wandb_run_id # Pass the ID if available
            )
            # comprehensive_metrics_report.json is now in logs/{exp_name}/
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}", exc_info=True)
        
        # --- Step 3: Local Workspace Cleanup ---
        if global_step == 0 or global_step == cfg.agent.steps :
            try:
                if global_step == cfg.agent.steps and (cfg.workspace_dir / "input").exists():
                     logger.info(f"Cleaning up input directory in workspace: {cfg.workspace_dir / 'input'}")
                     shutil.rmtree(cfg.workspace_dir / "input")
                elif global_step == 0 and cfg.workspace_dir.exists():
                    logger.info(f"Cleaning up entire workspace as no steps completed: {cfg.workspace_dir}")
                    shutil.rmtree(cfg.workspace_dir)
                
                submission_history_dir_to_clean = Path(cfg.log_dir) / "submission_history_per_step"
                if submission_history_dir_to_clean.exists():
                    if global_step == cfg.agent.steps: 
                        logger.info(f"Cleaning up submission_history_per_step directory: {submission_history_dir_to_clean}")
                        shutil.rmtree(submission_history_dir_to_clean)
            except Exception as e_clean:
                logger.error(f"Error during final workspace/log cleanup: {e_clean}")
        logger.info(f"Run '{cfg.exp_name}' finished.")

if __name__ == "__main__":
    run()