from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Optional

import numpy as np
from dataclasses_json import DataClassJsonMixin

import logging
logger = logging.getLogger(__name__)

@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):
    """
    Represents the value of a metric to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.
    """

    value: float | int | np.number | np.floating | np.ndarray | None
    maximize: bool | None = field(default=None, kw_only=True)
    # Add an optional source/name for debugging metric mismatches
    source_step: Optional[int] = field(default=None, kw_only=True) 
    competition_name: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.value is not None:
            # Ensure it's a basic numeric type before float conversion
            if not isinstance(self.value, (float, int, np.number, np.floating)):
                logger.warning(f"MetricValue received non-standard numeric type {type(self.value)}: {self.value}. Attempting conversion.")
                try:
                    self.value = float(self.value)
                except (ValueError, TypeError) as e:
                    logger.error(f"Could not convert metric value {self.value} to float: {e}. Setting to None.")
                    self.value = None
            else:
                self.value = float(self.value)

    def __gt__(self, other) -> bool:
        """True if self is a _better_ (not necessarily larger) metric value than other"""
        if self.value is None:
            return False
        if other.value is None:
            return True

        # Check for type mismatch
        if not isinstance(other, MetricValue):
            logger.warning(f"Attempting to compare MetricValue with incompatible type {type(other)}. Treating as not better.")
            return False # Or raise TypeError if strictness is preferred

        # Handle maximize flag consistency
        if self.maximize is not None and other.maximize is not None and self.maximize != other.maximize:
            logger.warning(
                f"Comparing metrics with different optimization directions! "
                f"Self (maximize={self.maximize}, value={self.value}, step={self.source_step}, comp={self.competition_name}) vs "
                f"Other (maximize={other.maximize}, value={other.value}, step={other.source_step}, comp={other.competition_name}). "
                f"This comparison might be meaningless. Defaulting to self not being better."
            )


            # For now, let's make it conservative.
            return False
        
        # Or, if this is critical, this could be a point to raise a non-halting warning and return a default.
        current_maximize_direction = self.maximize
        if current_maximize_direction is None and other.maximize is not None:
            current_maximize_direction = other.maximize
            logger.debug(f"Metric comparison using inferred maximize direction ({current_maximize_direction}) from 'other' metric.")
        elif current_maximize_direction is None and other.maximize is None:
            logger.error(f"Cannot compare metrics: both have undefined optimization direction. Returning False.")
            return False


        if self.value == other.value:
            return False

        comp = self.value > other.value
        return comp if current_maximize_direction else not comp

    def __eq__(self, other: Any) -> bool:
        return self.value == other.value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            opt_dir = "?"
        elif self.maximize:
            opt_dir = "↑"
        else:
            opt_dir = "↓"
        return f"Metric{opt_dir}({self.value_npsafe:.4f})"

    @property
    def is_worst(self):
        """True if the metric value is the worst possible value."""
        return self.value is None

    @property
    def value_npsafe(self):
        return self.value if self.value is not None else float("nan")


@dataclass
class WorstMetricValue(MetricValue):
    """
    Represents an invalid metric value, e.g. when the agent creates a buggy solution.
    Always compares worse than any valid metric value.
    """

    value: None = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
