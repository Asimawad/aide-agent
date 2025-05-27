import atexit
import logging
import shutil
import sys
import pandas as pd
import os
from .utils import load_benchmarks 
import time
from rich.console import Console
# RichHandler is now configured inside run.py for the main logger
# from rich.logging import RichHandler 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# Now import tensorflow, transformers, etc.
console = Console() 


from .utils.wandb_logger import WandbLogger 

from .agent import Agent, PlannerAgent
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
from rich.logging import RichHandler # Moved here

class VerboseFilter(logging.Filter):
    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)

# journal_to_rich_tree and journal_to_string_tree remain the same
def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            metric_display = f"{node.metric.value:.3f}" if node.metric and node.metric.value is not None else "N/A"
            if node is best_node:
                s = f"[{style}green]● {metric_display} (best)"
            else:
                s = f"[{style}green]● {metric_display}"
        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)
    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree

def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            metric_display = f"{node.metric.value:.3f}" if node.metric and node.metric.value is not None else "N/A"
            if marker_str:
                s = f"{indent}● {metric_display} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {metric_display} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)
    for n in journal.draft_nodes:
        append_rec(n, 0)
    return tree_str

def run():
    cfg = load_cfg()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Configure the 'aide' logger (used by most of the application)
    # This setup is now centralized here.
    logger = logging.getLogger("aide") 
    logger.setLevel(logging.DEBUG) # Capture all levels; handlers will filter
    logger.handlers.clear() # Remove any pre-existing handlers
    logger.propagate = False # Prevent duplication if root logger also has handlers

    log_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # File handler for all logs (verbose)
    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log", mode='w')
    verbose_file_handler.setFormatter(formatter)
    verbose_file_handler.setLevel(logging.DEBUG) # Log everything
    logger.addHandler(verbose_file_handler)

    # File handler for normal logs (INFO and above, no 'verbose' extra)
    normal_file_handler = logging.FileHandler(cfg.log_dir / "aide.log", mode='w')
    normal_file_handler.setFormatter(formatter)
    normal_file_handler.setLevel(logging.INFO) 
    normal_file_handler.addFilter(VerboseFilter()) # Filter out 'verbose=True' logs
    logger.addHandler(normal_file_handler)
    
    # Rich console handler (for pretty TTY output)
    rich_console_handler = RichHandler(
        console=console, # Use the global console object
        level=getattr(logging, cfg.log_level.upper(), logging.INFO), # Use level from config
        show_path=False, # Keep console output concise
        show_level=True, 
        show_time=True, # Or False if log_step provides it
        markup=True,
        rich_tracebacks=True,
        log_time_format="[%X]", # e.g., [14:30:59]
    )
    rich_console_handler.setFormatter(logging.Formatter("%(message)s")) # Let RichHandler format the message part
    rich_console_handler.addFilter(VerboseFilter()) # Also filter verbose from console if desired
    logger.addHandler(rich_console_handler)
    
    # Configure other loggers like httpx if needed
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING) # Keep it less noisy

    logger.info(f"Logging initialized. verbose.log level: DEBUG, aide.log level: INFO, Console level: {cfg.log_level.upper()}.")
    logger.debug("This is a verbose debug message, should only go to aide.verbose.log.")
    logger.info("This is an info message, should go to all logs and console (if INFO level).")
    logger.info("This is a verbose info message, should only go to aide.verbose.log.", extra={"verbose": True})


    logger.info(f'Starting run "{cfg.exp_name}"')

    # Load competition benchmarks early
    competition_benchmarks = None
    try:
        competition_benchmarks = load_benchmarks(cfg.competition_name)
        logger.info(f"Loaded benchmarks for competition: {cfg.competition_name}")
    except KeyError:
        logger.warning(f"No benchmarks found for competition: {cfg.competition_name}. Medal/threshold logging might be affected.")
    except Exception as e:
        logger.error(f"Error loading benchmarks for {cfg.competition_name}: {e}", exc_info=True)

    # Initialize WandbLogger (it will handle wandb.init)
    wandb_logger_instance = WandbLogger(cfg, logger, competition_benchmarks) # Pass benchmarks
    wandb_logger_instance.init_wandb() # This calls wandb.init() if enabled

    task_desc = load_task_desc(cfg)

    with Status("[blue]Preparing agent workspace...", console=console): # Use the global console
        prep_agent_workspace(cfg)
    
    global_step = 0 
    def cleanup():
        if global_step == 0 and cfg.workspace_dir.exists(): 
            logger.info(f"Cleaning up workspace as no steps were run: {cfg.workspace_dir}")
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)
    
    # competition_benchmarks = load_benchmarks(cfg.competition_name)
    logger.info(f"Loaded benchmarks for {cfg.competition_name}")

    journal = Journal()
    # Pass wandb_logger_instance and competition_benchmarks to Agent/PlannerAgent
    if cfg.agent.ITS_Strategy == "planner":
        logger.info("Initializing PlannerAgent.")
        agent = PlannerAgent(
            task_desc=task_desc, 
            cfg=cfg,
            journal=journal,
            wandb_logger=wandb_logger_instance, 
            competition_benchmarks=competition_benchmarks,
        )
    else:
        logger.info("Initializing Agent.")
        agent = Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            wandb_logger=wandb_logger_instance, 
            competition_benchmarks=competition_benchmarks,
        )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)
    )
    # global_step is already initialized

    def exec_callback(*args, **kwargs):
        logger.info("Interpreter: Executing code...") 
        res = interpreter.run(*args, **kwargs)
        logger.info("Interpreter: Code execution finished.") 
        return res

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console, # Use the global console
            transient=True 
        ) as progress_bar:
            main_task = progress_bar.add_task("Agent Steps", total=cfg.agent.steps)

            for i in range(cfg.agent.steps):
                current_step_num_display = i + 1
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} START ---")
                t0 = time.time()
                
                agent.step(exec_callback=exec_callback, current_step_number=current_step_num_display)
                
                t1 = time.time()
                logger.info(f"Step {current_step_num_display} processing took {t1 - t0:.2f} seconds.")
                
                if (current_step_num_display % cfg.get('save_every_n_steps', 1)) == 0:
                    logger.info(f"Saving run state at step {current_step_num_display}")
                    save_run(cfg, journal) 
                    
                global_step += 1
                progress_bar.update(main_task, advance=1)
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} END (Total time: {t1 - t0:.2f}s) --- \n")
                # Optional: Rich console rule for visual separation in TTY
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
        
        logger.info("W&B finalization initiated.")
        if wandb_logger_instance: # Check if it was initialized
            wandb_logger_instance.finalize_run(journal, competition_benchmarks) 
        
        # Workspace cleanup logic (remains the same)
        if global_step == 0 or global_step == cfg.agent.steps :
            try:
                if global_step == cfg.agent.steps and (cfg.workspace_dir / "input").exists():
                     logger.info(f"Cleaning up input directory in workspace: {cfg.workspace_dir / 'input'}")
                     shutil.rmtree(cfg.workspace_dir / "input")
                elif global_step == 0 and cfg.workspace_dir.exists():
                    logger.info(f"Cleaning up entire workspace as no steps completed: {cfg.workspace_dir}")
                    shutil.rmtree(cfg.workspace_dir)
            except Exception as e_clean:
                logger.error(f"Error during final workspace cleanup: {e_clean}")
        logger.info(f"Run '{cfg.exp_name}' finished.")


if __name__ == "__main__":
    run()