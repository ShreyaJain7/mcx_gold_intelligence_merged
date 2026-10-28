"""
Small I/O helpers so every role reads and writes frames the same way.

Parquet (via pyarrow) is the contract format. If no parquet engine is
installed, frames fall back to CSV next to the requested path, and readers
transparently pick up whichever exists - so a teammate missing ``pyarrow``
still gets a working pipeline instead of a stack trace.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _has_parquet_engine() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def save_frame(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` as parquet, or CSV if no parquet engine is available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet" and _has_parquet_engine():
        df.to_parquet(path)
        return path
    csv_path = path.with_suffix(".csv")
    if path.suffix == ".parquet":
        logger.warning("No parquet engine installed; writing %s instead", csv_path.name)
    df.to_csv(csv_path, index=df.index.name is not None)
    return csv_path


def load_frame(path: str | Path, index_col: str | None = None,
               parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a frame written by ``save_frame`` (parquet preferred, CSV fallback)."""
    path = Path(path)
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Neither {path} nor {csv_path} exists")
    df = pd.read_csv(csv_path, parse_dates=parse_dates or [])
    return df.set_index(index_col) if index_col else df


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj).date())
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _clean(obj: Any) -> Any:
    """Recursively replace NaN/inf floats with None so the JSON stays strict."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    """Write strict JSON (no NaN tokens) that any consumer can parse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, default=_json_default, allow_nan=False))
    return path
