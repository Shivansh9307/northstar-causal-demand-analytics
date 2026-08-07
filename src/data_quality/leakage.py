"""
Automated leakage checker (PROJECT_ARCHITECTURE.md §7).

The synthetic dataset contains columns that are *outputs* of the simulation, not
inputs available to a real retailer:

* `potential_demand_units` - latent demand before stockout censoring;
* `lost_sales_estimate_units` - derived from that latent demand;
* the anomaly labels - the simulator telling you which rows it perturbed;
* everything in `data/ground_truth/` - the answers the models are meant to recover.

Any of these in a feature set makes every downstream result meaningless, and the
failure is silent: metrics improve, so nothing looks wrong. §7 therefore requires
a checker that *fails the pipeline* rather than warning.

The check deliberately goes beyond exact name matching. A column renamed on the
way into a feature frame (`potential_demand`, `true_uplift`, `anomaly`) leaks
exactly as much as the original, so normalised, stem and prefix rules apply too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config  # noqa: E402

# Ground-truth join keys are legitimate features; the parameters beside them are not.
GROUND_TRUTH_JOIN_KEYS = {"sku_id", "category"}

# Substrings that identify a leaking column however it has been renamed.
FORBIDDEN_STEMS = ("potentialdemand", "lostsales", "anomaly", "realisedatt", "realizedatt")

# Ground-truth parameters follow a `true_*` naming convention.
FORBIDDEN_PREFIXES = ("true",)


class LeakageError(AssertionError):
    """Raised when a forbidden column reaches a model feature set."""


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def ground_truth_columns() -> Set[str]:
    """Column names present in data/ground_truth/, excluding legitimate join keys."""
    directory = config.path("ground_truth")
    columns: Set[str] = set()
    if not directory.exists():
        return columns
    for csv_path in directory.glob("*.csv"):
        header = pd.read_csv(csv_path, nrows=0)
        columns.update(header.columns)
    return columns - GROUND_TRUTH_JOIN_KEYS


def forbidden_column_set() -> Set[str]:
    """Every explicitly named column that must not be used for training."""
    return set(config.forbidden_columns()) | ground_truth_columns()


def find_leaks(columns: Iterable[str]) -> List[str]:
    """Return the subset of `columns` that would leak. Empty means safe."""
    explicit = {_normalise(c) for c in forbidden_column_set()}
    leaks: List[str] = []
    for column in columns:
        normalised = _normalise(column)
        if normalised in explicit:
            leaks.append(column)
        elif any(stem in normalised for stem in FORBIDDEN_STEMS):
            leaks.append(column)
        elif any(normalised.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            leaks.append(column)
    return leaks


def assert_no_leakage(columns: Sequence[str], context: str = "feature set") -> None:
    """Raise LeakageError if any column would leak simulation ground truth."""
    leaks = find_leaks(columns)
    if leaks:
        raise LeakageError(
            f"Leakage detected in {context}: {sorted(leaks)}. "
            "These are simulation outputs or ground-truth parameters, not features "
            "available to a real retailer. See PROJECT_ARCHITECTURE.md §7."
        )


def assert_frame_is_safe(frame: pd.DataFrame, context: str = "feature frame") -> None:
    """Convenience wrapper for a dataframe about to be used for training."""
    assert_no_leakage(list(frame.columns), context)


def safe_feature_columns(columns: Iterable[str]) -> List[str]:
    """Filter a column list down to those safe for training, preserving order."""
    leaks = set(find_leaks(columns))
    return [c for c in columns if c not in leaks]
