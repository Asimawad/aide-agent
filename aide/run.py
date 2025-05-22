import atexit
import logging
import shutil
import sys
import pandas as pd
import os
# from .utils import copytree, empirical_eval, advanced_metrics, load_benchmarks # Keep these
from .utils import empirical_eval, advanced_metrics, load_benchmarks # copytree is used locally by wandb_logger
from tqdm import tqdm
import time
from rich import print as rich_print # Alias to avoid conflict with print function
from rich.console import Console
from rich.logging import RichHandler # For console output

console = Console() # Keep your console instance

# Assuming wandb_logger.py will be created in aide/utils/
from .utils.wandb_logger import WandbLogger # We will create this

# Keep other imports from your original run.py
from . import backend
from .agent import Agent, PlannerAgent
from .interpreter import Interpreter
from .journal import Journal, Node  # Node might not be directly used here but good to have context
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.markdown import Markdown
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
# from .utils.wandb_retreival import get_wb_data, save_logs_to_wandb # This will be handled by WandbLogger


# VerboseFilter remains the same (from your original code)
class VerboseFilter(logging.Filter):
    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)

def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

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
            # support for multiple markers; atm only "best" is supported
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            if marker_str:
                s = f"{indent}● {node.metric.value:.3f} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {node.metric.value:.3f} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str

def run():
    cfg = load_cfg()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # --- Logging Setup ---
    # Get the 'aide' logger, which seems to be the main application logger.
    # Other modules (agent, backend, etc.) should use getLogger("aide") or getLogger("aide.module_name")
    logger = logging.getLogger("aide")
    logger.setLevel(logging.DEBUG) # Set logger to lowest level, handlers will filter
    logger.handlers.clear() # Clear any existing handlers from previous runs or imports
    logger.propagate = False # Prevent logs from going to the root logger if it has handlers

    log_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d]: %(message)s" # More detailed format
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 1. aide.verbose.log (DEBUG and up, all details)
    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log", mode='w')
    verbose_file_handler.setFormatter(formatter)
    verbose_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(verbose_file_handler)

    # 2. aide.log (INFO and up, standard operations)
    normal_file_handler = logging.FileHandler(cfg.log_dir / "aide.log", mode='w')
    normal_file_handler.setFormatter(formatter)
    normal_file_handler.setLevel(logging.INFO)
    normal_file_handler.addFilter(VerboseFilter()) # Filter out 'verbose=True' logs
    logger.addHandler(normal_file_handler)

    # 3. Terminal (Console) (INFO and up, Rich formatted for main events)
    # RichHandler for pretty console output
    # We can customize what RichHandler shows. `show_path=False` is often good.
    # `log_time_format` can customize time if `show_time=True`.
    # The `pretty_logging.py`'s `log_step` customizes console output, so this handler
    # might primarily be for non-step logs or if `log_step` is removed.
    # If `pretty_logging.log_step` is the primary way to show progress on console,
    # this RichHandler might be redundant or could be configured for errors only.
    # For now, let's assume `pretty_logging.log_step` is the main console status updater.
    # This console_handler will catch other INFO logs.
    rich_console_handler = RichHandler(
        console=console, # Your existing Rich Console instance
        level=logging.INFO,
        show_path=False,
        show_level=True, # Show INFO/WARNING/ERROR
        show_time=True,
        markup=True,
        rich_tracebacks=True,
        log_time_format="[%X]", # e.g., [10:30:55]
    )
    rich_console_handler.setFormatter(logging.Formatter("%(message)s")) # RichHandler handles styling
    rich_console_handler.addFilter(VerboseFilter()) # Also filter verbose from console
    logger.addHandler(rich_console_handler)

    # httpx logger configuration (from your original code)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING) # Keep it less noisy

    logger.info(f"Logging initialized. verbose.log level: DEBUG, aide.log level: INFO, Console level: INFO.")
    logger.debug("This is a verbose debug message, should only go to aide.verbose.log.")
    logger.info("This is an info message, should go to all logs and console.")
    logger.info("This is a verbose info message, should only go to aide.verbose.log.", extra={"verbose": True})


    logger.info(f'Starting run "{cfg.exp_name}"')

    # Initialize WandbLogger (we'll create this util next)
    wandb_logger_instance = WandbLogger(cfg, logger) # Pass your main app logger
    wandb_logger_instance.init_wandb()

    task_desc = load_task_desc(cfg)
    # task_desc_str = backend.compile_prompt_to_md(task_desc) # If needed for display

    with Status("[blue]Preparing agent workspace...", console=console):
        prep_agent_workspace(cfg)
    
    global_step = 0 # This seems to track number of agent steps executed in this run

    def cleanup():
        if global_step == 0 and cfg.workspace_dir.exists(): # only if no steps ran
            logger.info(f"Cleaning up workspace as no steps were run: {cfg.workspace_dir}")
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)
    
    competition_benchmarks = load_benchmarks(cfg.competition_name)
    logger.info(f"Loaded benchmarks for {cfg.competition_name}")

    journal = Journal()
    if cfg.agent.ITS_Strategy == "planner":
        logger.info("Initializing PlannerAgent.")
        agent = PlannerAgent(
            task_desc=task_desc, # task_desc is already a string or dict
            cfg=cfg,
            journal=journal,
            wandb_logger=wandb_logger_instance, # Pass the logger instance
            competition_benchmarks=competition_benchmarks,
        )
    else:
        logger.info("Initializing Agent.")
        agent = Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            wandb_logger=wandb_logger_instance, # Pass the logger instance
            competition_benchmarks=competition_benchmarks,
        )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)
    )
    # global_step = len(journal) # If loading a journal, this would be set. For new run, it's 0.

    # Rich Progress bar (from your original code - seems fine)
    # prog = Progress(...)
    # status_rich = Status("[green]Generating code...", console=console) # Renamed from status

    def exec_callback(*args, **kwargs):
        # status_rich.update("[magenta]Executing code...")
        logger.info("Interpreter: Executing code...") # Logged via logger
        res = interpreter.run(*args, **kwargs)
        # status_rich.update("[green]Generating code...")
        logger.info("Interpreter: Code execution finished.") # Logged via logger
        return res

    # The generate_live function for Rich display can be simplified if console output is primarily through logging
    # For now, keeping it as it might still be useful for a live updating panel.

    try:
        # Using rich.progress.Progress for the main loop
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console, # Use your main console
            transient=True # Clear progress bar on exit
        ) as progress_bar:
            main_task = progress_bar.add_task("Agent Steps", total=cfg.agent.steps)

            for i in range(cfg.agent.steps):
                current_step_num_display = i + 1
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} START ---")
                # progress_bar.update(main_task, description=f"Step {current_step_num_display}/{cfg.agent.steps}")
                
                agent.step(exec_callback=exec_callback, current_step_number=current_step_num_display)
                
                # Save run state periodically
                if (current_step_num_display % cfg.get('save_every_n_steps', 1)) == 0: # Example: save every step
                    logger.info(f"Saving run state at step {current_step_num_display}")
                    save_run(cfg, journal) # save_run needs access to cfg and journal

                global_step += 1 # Increment after successful step
                progress_bar.update(main_task, advance=1)
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} END ---")
                console.rule(f"Step {current_step_num_display} Summary") # Visual separator

        logger.info("All agent steps completed.")
        logger.info(journal_to_string_tree(journal)) # Log final tree structure
        save_run(cfg, journal) # Final save

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
        wandb_logger_instance.finalize_run(journal, competition_benchmarks) # Pass journal for summary
        
        # Your existing final cleanup for workspace
        if global_step == 0 or global_step == cfg.agent.steps :
            try:
                if global_step == cfg.agent.steps and (cfg.workspace_dir / "input").exists():
                     logger.info(f"Cleaning up input directory in workspace: {cfg.workspace_dir / 'input'}")
                     shutil.rmtree(cfg.workspace_dir / "input")
                elif global_step == 0 and cfg.workspace_dir.exists(): # Only if it's a fresh run with 0 steps
                    logger.info(f"Cleaning up entire workspace as no steps completed: {cfg.workspace_dir}")
                    shutil.rmtree(cfg.workspace_dir)
            except Exception as e_clean:
                logger.error(f"Error during final workspace cleanup: {e_clean}")
        logger.info(f"Run '{cfg.exp_name}' finished.")


if __name__ == "__main__":
    run()