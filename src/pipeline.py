"""
Batch pipeline and the public API between the roles.

    Person A (data)  -> data/processed/normalized_bhavcopy.parquet
    Person B (quant) -> this module: load -> normalise -> per-pair strategy
                        -> attribution -> data/outputs/...
    Person C (viz)   -> app/main.py calls ``prepare_market`` + ``run_pair``
                        directly (live, cached), or reads data/outputs/.

CLI::

    python -m src.pipeline                                  # all default pairs
    python -m src.pipeline --pairs GOLDM:GOLDPETAL --entry-z 2.5
    python -m src.pipeline --input path/to/file.parquet --output data/outputs

Outputs (per run)::

    data/outputs/summary.json                  run metadata + metrics per pair
    data/outputs/normalized_prices.parquet     active-contract ₹/g@999 per symbol
    data/outputs/contract_calendar.parquet     lifecycle calendar
    data/outputs/<A>-<B>/signals.parquet       pair frame, residual, z, liquidity
    data/outputs/<A>-<B>/daily_pnl.parquet     position, P&L, equity, returns
    data/outputs/<A>-<B>/trades.parquet        round-trip blotter
    data/outputs/<A>-<B>/benchmark.parquet     buy & hold + gold_return
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .attribution import AttributionResult, attribute_strategy
from .config import DEFAULT_PAIRS, CalendarConfig, StrategyConfig
from .lifecycle import build_contract_calendar
from .quant_engine import (
    StrategyResult,
    continuous_normalized_prices,
    current_status,
    load_bhavcopy,
    normalize_prices,
    run_strategy,
)
from .storage import save_frame, save_json

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/processed/normalized_bhavcopy.parquet")
DEFAULT_OUTPUT = Path("data/outputs")


@dataclass
class PairRun:
    """A strategy result bundled with its attribution and live status."""

    result: StrategyResult
    attribution: AttributionResult
    status: dict[str, Any]


def prepare_market(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load, validate and normalise Person A's file (single entry point)."""
    return normalize_prices(load_bhavcopy(path))


def available_pairs(df_norm: pd.DataFrame, min_days: int = 120,
                    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS) -> list[tuple[str, str]]:
    """Pairs whose legs both have at least ``min_days`` of history."""
    days = df_norm.groupby("symbol")["trade_date"].nunique()
    return [(a, b) for a, b in pairs if days.get(a, 0) >= min_days and days.get(b, 0) >= min_days]


def run_pair(df_norm: pd.DataFrame, leg_a: str, leg_b: str,
             cfg: StrategyConfig | None = None) -> PairRun:
    """Strategy + attribution + live status for one pair."""
    res = run_strategy(df_norm, leg_a, leg_b, cfg or StrategyConfig())
    return PairRun(res, attribute_strategy(res), current_status(res))


def _file_digest(path: Path) -> str | None:
    for p in (path, path.with_suffix(".csv")):
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return None


def run_all(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
            pairs: list[tuple[str, str]] | None = None,
            cfg: StrategyConfig | None = None) -> dict[str, Any]:
    """Run every pair and persist artefacts. Returns the summary dict."""
    cfg = cfg or StrategyConfig()
    df = prepare_market(input_path)
    pairs = pairs or available_pairs(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_frame(continuous_normalized_prices(df, cfg.roll_dte), output_dir / "normalized_prices.parquet")
    save_frame(build_contract_calendar(df, CalendarConfig()), output_dir / "contract_calendar.parquet")

    summary: dict[str, Any] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(input_path), "input_sha256_16": _file_digest(Path(input_path)),
        "data_start": df["trade_date"].min(), "data_end": df["trade_date"].max(),
        "config": {k: (dict(v) if isinstance(v, dict) else v)
                   for k, v in dataclasses.asdict(cfg).items()},
        "pairs": {},
    }
    for a, b in pairs:
        try:
            run = run_pair(df, a, b, cfg)
        except ValueError as exc:  # e.g. a leg with no active contracts
            logger.warning("Skipping %s-%s: %s", a, b, exc)
            continue
        res = run.result
        pdir = output_dir / res.pair_id
        save_frame(res.frame, pdir / "signals.parquet")
        save_frame(res.daily, pdir / "daily_pnl.parquet")
        save_frame(res.trades, pdir / "trades.parquet")
        save_frame(res.benchmark, pdir / "benchmark.parquet")
        status = {k: v for k, v in run.status.items() if k != "gates"}
        status["gates"] = run.status["gates"]
        summary["pairs"][res.pair_id] = {"trading": res.summary,
                                         "attribution": run.attribution.to_dict(),
                                         "status": status}
        att = run.attribution
        logger.info(f"{res.pair_id:<20} trades={res.summary['n_trades']:>3}  "
                    f"P&L=₹{res.summary['cumulative_pnl_inr']:>12,.0f}  "
                    f"Sharpe={att.sharpe_ratio:5.2f}  beta={att.beta:6.3f}  "
                    f"state={run.status['state']}")

    save_json(summary, output_dir / "summary.json")
    return summary


def _parse_pairs(values: list[str] | None) -> list[tuple[str, str]] | None:
    if not values:
        return None
    out = []
    for v in values:
        a, _, b = v.upper().partition(":")
        if not b:
            raise argparse.ArgumentTypeError(f"Pair must look like GOLDM:GOLDPETAL, got {v!r}")
        out.append((a, b))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="MCX gold relative-value pipeline")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--pairs", nargs="*", help="e.g. GOLDM:GOLDPETAL GOLDM:GOLDGUINEA")
    p.add_argument("--entry-z", type=float, default=StrategyConfig.entry_z)
    p.add_argument("--exit-z", type=float, default=StrategyConfig.exit_z)
    p.add_argument("--z-window", type=int, default=StrategyConfig.z_window)
    p.add_argument("--detrend-window", type=int, default=StrategyConfig.detrend_window)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = dataclasses.replace(StrategyConfig(), entry_z=args.entry_z, exit_z=args.exit_z,
                              z_window=args.z_window, detrend_window=args.detrend_window)
    run_all(args.input, args.output, _parse_pairs(args.pairs), cfg)


if __name__ == "__main__":
    main()
