import atexit
import logging
import shutil
import sys
import pandas as pd
import os
from .utils import copytree, empirical_eval, advanced_metrics, load_benchmarks
from tqdm import tqdm  # Import tqdm for progress bar
import time

os.environ["WANDB_API_KEY"] = "8ca0d241dd66f5a643d64a770d61ad066f937c48"

try:
    import wandb
except ImportError:
    wandb = None

from . import backend
from .agent import Agent, PlannerAgent
from .interpreter import Interpreter
from .journal import Journal, Node
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
from .utils.wandb_retreival import get_wb_data, save_logs_to_wandb


class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

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

    log_format = "[%(asctime)s] %(levelname)s: %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, cfg.log_level.upper()))

    # Clear any existing handlers to prevent duplicates
    root_logger.handlers.clear()

    # Configure httpx logger
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Configure aide logger
    logger = logging.getLogger("aide")
    logger.handlers.clear()  # Clear any existing handlers

    # Create handlers
    file_handler = logging.FileHandler(cfg.log_dir / "aide.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(getattr(logging, cfg.log_level.upper()))
    console_handler.addFilter(VerboseFilter())

    # Add handlers to aide logger
    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    # Set propagate to False to prevent duplicate logging
    logger.propagate = False
    logger.info(f'Starting run "{cfg.exp_name}"')

    wandb_run = None
    if wandb and cfg.wandb.enabled:
        try:
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name,
                # dir="./",
                config=OmegaConf.to_container(cfg, resolve=True),  # Log the config
                job_type="aide_run",
                tags=["aide-agent", cfg.agent.code.model],  # Example tags
            )

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            wandb_run = None

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)
    competition_benchmarks = load_benchmarks(cfg.competition_name)

    journal = Journal()
    if cfg.agent.ITS_Strategy == "planner":
        agent = PlannerAgent(
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            wandb_run=wandb_run,
            competition_benchmarks=competition_benchmarks,
        )
    else:
        # Default to the standard Agent
        agent = Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            wandb_run=wandb_run,
            competition_benchmarks=competition_benchmarks,
        )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
    )
    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def generate_live():
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"),
            prog,
            status,
        )

        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    try:
        # Wrap the loop with tqdm for a progress bar
        with tqdm(total=cfg.agent.steps, desc="Agent Steps", unit="step") as pbar:
            for i in range(cfg.agent.steps):
                t0 = time.time()
                agent.step(exec_callback=exec_callback, current_step_number=i + 1)
                t1 = time.time()
                print(
                    f"*****************Time taken for step {i+1}: {t1-t0} seconds***************"
                )
                save_run(cfg, journal)  # Save progress locally
                global_step += 1
                pbar.update(1)

            # on the last step, print the tree

        logger.info(journal_to_string_tree(journal))
        save_run(cfg, journal)

    finally:  # Add finally block - This block runs no matter what.
        interpreter.cleanup_session()

        # # Check if a W&B run was successfully started
        if wandb_run:

            logger.info("Finishing W&B Run...")
            try:
                wo_step = None
                no_of_csvs = 0
                buggy_nodes = 0
                avg_code_quality = 0
                gold_medals = 0
                silver_medals = 0
                bronze_medals = 0
                above_amedian = 0
                effective_debugs = 0

                for node in journal.nodes:
                    if not node.is_buggy:
                        wo_step = node.step if wo_step is None else wo_step
                        no_of_csvs += 1
                        avg_code_quality += node.code_quality
                        if (
                            node.metric.value
                            >= competition_benchmarks["median_threshold"]
                        ):
                            above_amedian += 1
                        if node.effective_debug_step:
                            effective_debugs += 1
                        if (
                            node.metric.value
                            >= competition_benchmarks["gold_threshold"]
                        ):
                            gold_medals += 1
                        elif (
                            node.metric.value
                            >= competition_benchmarks["silver_threshold"]
                        ):
                            silver_medals += 1
                        elif (
                            node.metric.value
                            >= competition_benchmarks["bronze_threshold"]
                        ):
                            bronze_medals += 1
                    else:
                        buggy_nodes += 1
                        avg_code_quality += node.code_quality

                avg_code_quality /= cfg.agent.steps
                wandb.summary["steps_to_first_working_code"] = (
                    wo_step if wo_step is not None else cfg.agent.steps + 10
                )
                wandb.summary["no_of_csvs"] = no_of_csvs
                wandb.summary["buggy_nodes"] = buggy_nodes
                wandb.summary["avg_code_quality"] = avg_code_quality
                wandb.summary["gold_medals"] = gold_medals
                wandb.summary["silver_medals"] = silver_medals
                wandb.summary["bronze_medals"] = bronze_medals
                wandb.summary["above_amedian"] = above_amedian
                wandb.summary["effective_debug_step"] = effective_debugs
            except Exception as e:
                print(f"Error during collecting data summary: {e}")
            best_node = journal.get_best_node()
            if best_node:
                wandb.summary["best_validation_metric"] = best_node.metric.value
                wandb.summary["best_node_id"] = best_node.id
                wandb.summary["best_node_step"] = best_node.step

            try:
                shutil.rmtree(cfg.workspace_dir / "input", ignore_errors=True)
                df = pd.DataFrame(wandb.summary._as_dict())
                df.to_csv(f"{cfg.log_dir}/summary.csv", index=False)
            except:
                pass
            try:
                empirical_eval.calculate_empirical_metrics(id=wandb_run.id)
            except Exception as e:
                print(f"calculating empirical metrics gone wrong with this error : {e}")
            try:
                advanced_metrics.calculate_advanced_metrics(cfg.exp_name, journal)
            except Exception as e:
                print(f"calculating advanced metrics gone wrong with this error : {e}")
            try:
                save_logs_to_wandb()
                wandb_run.finish()
                logger.info("W&B Run finished.")
            except Exception as e_finalization:
                logger.error(f"Error during W&B finalization: {e_finalization}")
                if wandb_run:
                    try:
                        wandb_run.finish()
                        logger.info("W&B Run finished after encountering an error.")
                    except Exception as e_finish_retry:
                        logger.error(f"Error during W&B finish retry: {e_finish_retry}")

        # Clean up workspace if this was the first step
        if global_step == 0 or global_step == cfg.agent.steps:
            try:
                if global_step == cfg.agent.steps:
                    shutil.rmtree(cfg.workspace_dir / "input")
            except:
                shutil.rmtree(cfg.workspace_dir)


if __name__ == "__main__":
    run()
