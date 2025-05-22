# aide/utils/pretty_logging.py
from __future__ import annotations
import logging
import time
from rich.console import Console # Console can be passed or obtained if needed
from rich.logging import RichHandler # Not setting up handlers here anymore
from rich.theme import Theme # Not setting up themes here

logger = logging.getLogger("aide") # Get the 'aide' logger configured in run.py
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
    Formats a progress line for the console.
    Uses the 'aide' logger, which is configured with RichHandler in run.py.
    """
    mm, ss = divmod(int(time.time() - _T0), 60)
    status_icon = "[red]✗[/]" if is_buggy else "[green]✓[/]" # Rich markup
    exec_str = f"exec {exec_time:>.2f}s" if exec_time is not None else ""
    metric_str = f"metric {metric:>.4f}" if metric is not None else ""
    
    # Log as INFO so it appears on console via RichHandler if configured for INFO
    # The RichHandler in run.py uses "%(message)s" format, so markup works.
    log_message = (
        f"[dim]{mm:02d}:{ss:02d}[/]  "
        f"Step {step}/{total:<2}  "
        f"{stage:<15} " # Stage name
        f"{status_icon}  " # Buggy status
        f"{exec_str}  {metric_str}".rstrip()
    )
    logger.info(log_message) # Log as INFO to be caught by console RichHandler

    # Also log a more detailed version to file logs if needed (e.g., with node ID)
    # This depends on what information `log_step` has access to.
    # For example, if it had node_id:
    # logger.debug(f"STEP_SUMMARY: Step {step}/{total}, Node {node_id}, Stage {stage}, Buggy {is_buggy}, Exec {exec_time}, Metric {metric}", extra={"verbose":True})