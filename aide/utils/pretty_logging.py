# utils/pretty_logging.py
from __future__ import annotations
import logging, pathlib, time
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ────────────────────────────────────────────────────────────
# 1) Root-logger: 2 handlers →   • run.log  (DEBUG, all noise)
#                                • Rich TTY (INFO+, coloured)
# ────────────────────────────────────────────────────────────
LOG_PATH = pathlib.Path("logs") / "run.log"  # flat file in CWD

file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)

rich_handler = RichHandler(
    # console=Console(theme=Theme({
    #     "logging.level.info":  "cyan",
    #     "logging.level.error": "bold red",
    # })),
    rich_tracebacks=True,
    markup=True,
    show_level=True,
    show_time=False,  # we print our own time stamp in log_step
    show_path=False,
)
rich_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, rich_handler],
    format="%(message)s",
    force=True,  # override any earlier basicConfig
)

logger = logging.getLogger("aide")

# ────────────────────────────────────────────────────────────
# 2) One-liner helper – call after every Agent.step()
# ────────────────────────────────────────────────────────────
_T0 = time.time()


def log_step(
    *,
    step: int,
    total: int,
    stage: str,
    is_buggy: bool,
    exec_time: float | None = None,
    metric: float | None = None,
) -> None:
    """
    Print a neat progress line to stdout **and** to the rich handler.
    Example:
        00:03  step 2/25  self-reflection  ✓  exec 1.07s  metric 0.8529
    """
    mm, ss = divmod(int(time.time() - _T0), 60)
    status = "[red]✗[/]" if is_buggy else "[green]✓[/]"
    exec_str = f"exec {exec_time:>.2f}s" if exec_time is not None else ""
    metric_str = f"metric {metric:>.4f}" if metric is not None else ""
    logger.debug(
        f"[dim]{mm:02d}:{ss:02d}[/]  "
        f"step {step}/{total:<2}  "
        f"{stage:<15} "
        f"{status}  "
        f"{exec_str}  {metric_str}".rstrip()
    )
