"""
Shared configuration access for PromoPulse.

`config/config.yaml` is the single source of truth for the seed, the dev/full
scale toggle, every path, and the leakage rules. Nothing downstream should
hardcode those values (PROJECT_ARCHITECTURE.md §6 Phase 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Scale:
    """Resolved dataset dimensions for the active mode."""

    n_stores: int
    n_skus: int
    start_date: str
    end_date: str


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed() -> int:
    return int(load_config()["seed"])


def full_mode() -> bool:
    return bool(load_config()["full_mode"])


def scale() -> Scale:
    cfg = load_config()
    values = cfg["scale"]["full" if full_mode() else "dev"]
    return Scale(
        n_stores=int(values["n_stores"]),
        n_skus=int(values["n_skus"]),
        start_date=str(values["start_date"]),
        end_date=str(values["end_date"]),
    )


def path(name: str) -> Path:
    """Resolve a configured path key to an absolute path under the project root."""
    paths = load_config()["paths"]
    if name not in paths:
        raise KeyError(f"Unknown path key {name!r}. Available: {sorted(paths)}")
    return PROJECT_ROOT / paths[name]


def forbidden_columns() -> List[str]:
    """Columns that must never enter a model feature set (§7)."""
    return list(load_config()["leakage"]["forbidden_columns"])


def forbidden_sources() -> List[str]:
    """Directories whose columns are validation-only (§7)."""
    return list(load_config()["leakage"]["forbidden_sources"])
