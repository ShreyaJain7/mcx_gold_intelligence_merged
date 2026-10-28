"""
Synthetic MCX gold bhavcopy generator - for demos and tests only.

Produces a file with exactly Person A's schema so Roles B and C can build and
test before real data lands. The generated market has:

* a GBM spot gold path (₹/g at 999 purity) with cost-of-carry futures curves;
* two expiry calendars: GOLDM on the 5th of each month, the small contracts
  on the last business day (so nearest-expiry matching is exercised);
* mean-reverting, occasionally jumping premiums on GOLDGUINEA / GOLDPETAL /
  GOLDTEN versus GOLDM - the dislocations the strategy is meant to find;
* front-month-heavy volume/OI, thin days, and a short GOLDTEN history;
* deliberately dirty rows (GOLDGUINEA quoted per gram, duplicates, zero
  closes) so the validator and normaliser are exercised.

None of these numbers are market data. Do not draw conclusions from them.

Usage::

    python -m src.synthetic_data --output data/processed/normalized_bhavcopy.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, BMonthEnd

from .config import CONTRACT_SPECS
from .storage import save_frame

logger = logging.getLogger(__name__)


def _expiry_calendar(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.period_range(start.to_period("M"), (end + pd.DateOffset(months=4)).to_period("M"),
                             freq="M")
    out = []
    for m in months:
        if symbol == "GOLDM":
            d = pd.Timestamp(year=m.year, month=m.month, day=5)
            if d.weekday() >= 5:            # roll back to previous business day
                d = d - BDay(1)
        else:
            d = BMonthEnd().rollforward(pd.Timestamp(year=m.year, month=m.month, day=1))
        out.append(pd.Timestamp(d).normalize())
    return out


def generate_synthetic_bhavcopy(start: str = "2023-10-02", end: str = "2026-09-25",
                                goldten_start: str = "2025-06-02",
                                seed: int = 11) -> pd.DataFrame:
    """Return a synthetic bhavcopy matching ``quant_engine.REQUIRED_COLUMNS``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # --- spot gold, ₹/g at 999 ------------------------------------------------
    mu, sigma, s0 = 0.15, 0.13, 5900.0
    shocks = rng.normal((mu - 0.5 * sigma**2) / 252, sigma / np.sqrt(252), n)
    spot = s0 * np.exp(np.cumsum(shocks))
    carry = 0.065

    # --- per-symbol premium vs GOLDM (₹/g), OU + rare jumps -------------------
    def ou(theta: float, vol: float, jump_p: float, jump_size: float, level: float) -> np.ndarray:
        x = np.zeros(n)
        for t in range(1, n):
            jump = rng.normal(0, jump_size) if rng.random() < jump_p else 0.0
            x[t] = x[t - 1] - theta * x[t - 1] + rng.normal(0, vol) + jump
        return x + level

    premium = {
        "GOLDM": np.zeros(n),
        "GOLDGUINEA": ou(0.12, 2.5, 0.015, 30.0, 6.0),
        "GOLDPETAL": ou(0.15, 2.0, 0.020, 25.0, 4.0),
        "GOLDTEN": ou(0.10, 3.0, 0.015, 30.0, 2.0),
    }
    base_volume = {"GOLDM": 15000, "GOLDGUINEA": 2500, "GOLDPETAL": 20000, "GOLDTEN": 1500}
    listing_months = {"GOLDM": 4, "GOLDGUINEA": 3, "GOLDPETAL": 3, "GOLDTEN": 3}

    rows = []
    date_pos = {d: i for i, d in enumerate(dates)}
    for sym, spec in CONTRACT_SPECS.items():
        first_day = pd.Timestamp(goldten_start) if sym == "GOLDTEN" else dates[0]
        for exp in _expiry_calendar(sym, dates[0], dates[-1]):
            listing = exp - pd.DateOffset(months=listing_months[sym])
            live = dates[(dates >= max(listing, first_day)) & (dates <= exp)]
            for d in live:
                i = date_pos[d]
                tau = (exp - d).days / 365.0
                fair = spot[i] * np.exp(carry * tau) + premium[sym][i] + rng.normal(0, 1.0)
                # Convert ₹/g@999 back to the contract's quotation basis.
                quoted = fair * spec.quote_grams * (spec.quote_purity / 999.0)
                close = round(quoted / spec.tick_size) * spec.tick_size
                rows.append((d, sym, exp, close, i))

    df = pd.DataFrame(rows, columns=["trade_date", "symbol", "expiry_date", "close", "_i"])

    # --- volume / OI: front month dominates, dries up into expiry -------------
    df["dte"] = (df["expiry_date"] - df["trade_date"]).dt.days
    df["rank"] = df.groupby(["trade_date", "symbol"])["expiry_date"].rank(method="first") - 1
    weight = np.select([df["rank"] == 0, df["rank"] == 1], [1.0, 0.15], 0.03)
    weight = np.where(df["dte"] <= 3, weight * 0.2, weight)
    base = df["symbol"].map(base_volume).to_numpy(float)
    vol = base * weight * rng.lognormal(0, 0.5, len(df))
    thin = df["symbol"].isin(["GOLDPETAL", "GOLDGUINEA"]).to_numpy() & (rng.random(len(df)) < 0.03)
    vol = np.where(thin, vol * 0.02, vol)
    df["volume"] = np.maximum(vol.round(), 0).astype("int64")
    df["open_interest"] = (base * 3 * weight * rng.lognormal(0, 0.2, len(df))).round().astype("int64")
    df = df.drop(columns=["_i", "dte", "rank"])

    # --- dirty rows to exercise validation ------------------------------------
    guinea_idx = df.index[df["symbol"] == "GOLDGUINEA"]
    per_gram = rng.choice(guinea_idx, size=6, replace=False)
    df.loc[per_gram, "close"] = (df.loc[per_gram, "close"] / 8.0).round()
    dupes = df.sample(3, random_state=seed)
    zero = df.sample(2, random_state=seed + 1).assign(close=0.0)
    df = pd.concat([df, dupes, zero], ignore_index=True)

    df["close"] = df["close"].astype("float64")
    df["trade_date"] = df["trade_date"].astype("datetime64[ns]")
    df["expiry_date"] = df["expiry_date"].astype("datetime64[ns]")
    return df.sort_values(["trade_date", "symbol", "expiry_date"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="data/processed/normalized_bhavcopy.parquet")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = generate_synthetic_bhavcopy(seed=args.seed)
    path = save_frame(df, Path(args.output))
    logger.info("Wrote %d synthetic rows to %s", len(df), path)


if __name__ == "__main__":
    main()
