import atexit
import logging
import shutil
import sys
import pandas as pd
import os
from .utils import load_benchmarks 
import time
from rich.console import Console
from rich.logging import RichHandler # For console output

console = Console() 

from .utils.wandb_logger import WandbLogger 

from .agent import Agent, PlannerAgent
from .interpreter import Interpreter
from .journal import Journal, Node  # Node might not be directly used here but good to have context
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

    logger = logging.getLogger("aide")
    logger.setLevel(logging.DEBUG) 
    logger.handlers.clear() 
    logger.propagate = False 

    log_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d]: %(message)s" # More detailed format
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)


    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log", mode='w')
    verbose_file_handler.setFormatter(formatter)
    verbose_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(verbose_file_handler)


    normal_file_handler = logging.FileHandler(cfg.log_dir / "aide.log", mode='w')
    normal_file_handler.setFormatter(formatter)
    normal_file_handler.setLevel(logging.INFO)
    normal_file_handler.addFilter(VerboseFilter()) # Filter out 'verbose=True' logs
    logger.addHandler(normal_file_handler)

    rich_console_handler = RichHandler(
        console=console,
        level=logging.INFO,
        show_path=False,
        show_level=True, # Show INFO/WARNING/ERROR
        show_time=True,
        markup=True,
        rich_tracebacks=True,
        log_time_format="[%X]",
    )
    rich_console_handler.setFormatter(logging.Formatter("%(message)s")) 
    rich_console_handler.addFilter(VerboseFilter()) 
    logger.addHandler(rich_console_handler)


    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING) # Keep it less noisy

    logger.info(f"Logging initialized. verbose.log level: DEBUG, aide.log level: INFO, Console level: INFO.")
    logger.debug("This is a verbose debug message, should only go to aide.verbose.log.")
    logger.info("This is an info message, should go to all logs and console.")
    logger.info("This is a verbose info message, should only go to aide.verbose.log.", extra={"verbose": True})


    logger.info(f'Starting run "{cfg.exp_name}"')


    wandb_logger_instance = WandbLogger(cfg, logger) 
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
    
    competition_benchmarks = load_benchmarks(cfg.competition_name)
    logger.info(f"Loaded benchmarks for {cfg.competition_name}")

    journal = Journal()
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
    global_step = len(journal) 

    # prog = Progress(...)
    # status_rich = Status("[green]Generating code...", console=console) 

    def exec_callback(*args, **kwargs):
        # status_rich.update("[magenta]Executing code...")
        logger.info("Interpreter: Executing code...") # Logged via logger
        res = interpreter.run(*args, **kwargs)
        # status_rich.update("[green]Generating code...")
        logger.info("Interpreter: Code execution finished.") # Logged via logger
        return res

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True 
        ) as progress_bar:
            main_task = progress_bar.add_task("Agent Steps", total=cfg.agent.steps)

            for i in range(cfg.agent.steps):
                current_step_num_display = i + 1
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} START ---")
                # progress_bar.update(main_task, description=f"Step {current_step_num_display}/{cfg.agent.steps}")
                t0 = time.time()
                agent.step(exec_callback=exec_callback, current_step_number=current_step_num_display)
                t1 = time.time()
                logger.info(f"Step {current_step_num_display} took {t1 - t0:.2f} seconds.")
                # Save run state periodically
                if (current_step_num_display % cfg.get('save_every_n_steps', 1)) == 0: # Example: save every step
                    logger.info(f"Saving run state at step {current_step_num_display}")
                    save_run(cfg, journal) 
                    
                global_step += 1
                
                progress_bar.update(main_task, advance=1)
                logger.info(f"--- Agent Step {current_step_num_display}/{cfg.agent.steps} END : took {t1 - t0:.2f} seconds --- \n")
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