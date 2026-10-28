"""
Contract lifecycle calendar (shared by Role B checks and Role C's calendar view).

For every (symbol, expiry) seen in the bhavcopy we derive:

* ``listing_date``   - first trade date observed in the data. This is a
                       *proxy*: the official MCX listing date can be earlier
                       if the contract did not trade (or the history starts
                       later). The dashboard labels it accordingly.
* ``tender_start``   - start of the tender / delivery-intent window, taken as
                       ``tender_business_days`` business days before expiry
                       (weekends skipped; exchange holidays are NOT modelled).
* ``expiry_date``    - from the bhavcopy.
* ``status``         - Trading / Tender period / Expired, as of the latest
                       trade date in the file.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay

from .config import CalendarConfig


def build_contract_calendar(df: pd.DataFrame, cfg: CalendarConfig | None = None,
                            as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return one row per contract with lifecycle dates and liquidity context."""
    cfg = cfg or CalendarConfig()
    as_of = pd.Timestamp(as_of) if as_of is not None else df["trade_date"].max()

    g = df.groupby(["symbol", "expiry_date"])
    cal = g.agg(listing_date=("trade_date", "min"),
                last_trade_date=("trade_date", "max"),
                total_volume=("volume", "sum")).reset_index()
    # Latest open interest per contract (point-in-time: last row on or before as_of).
    last_rows = df[df["trade_date"] <= as_of].sort_values("trade_date") \
        .groupby(["symbol", "expiry_date"]).tail(1)
    cal = cal.merge(last_rows[["symbol", "expiry_date", "open_interest"]]
                    .rename(columns={"open_interest": "latest_oi"}),
                    on=["symbol", "expiry_date"], how="left")

    cal["tender_start"] = cal["expiry_date"] - BDay(cfg.tender_business_days)
    cal["days_to_expiry"] = (cal["expiry_date"] - as_of).dt.days

    def status(row: pd.Series) -> str:
        if row["expiry_date"] < as_of:
            return "Expired"
        if as_of >= row["tender_start"]:
            return "Tender period"
        return "Trading"

    cal["status"] = cal.apply(status, axis=1)
    horizon = as_of - pd.Timedelta(days=cfg.recently_expired_days)
    cal["show"] = cal["expiry_date"] >= horizon
    return cal.sort_values(["expiry_date", "symbol"]).reset_index(drop=True)


def calendar_segments(cal: pd.DataFrame) -> pd.DataFrame:
    """Long-form segments for a Gantt/timeline chart.

    Two bars per contract: "Trading" (listing -> tender start) and
    "Tender period" (tender start -> expiry).
    """
    rows = []
    for r in cal.itertuples(index=False):
        label = f"{r.symbol} {r.expiry_date:%b-%y}"
        trading_end = max(r.listing_date, r.tender_start)
        rows.append({"contract": label, "symbol": r.symbol, "phase": "Trading",
                     "start": r.listing_date, "end": trading_end, "status": r.status})
        rows.append({"contract": label, "symbol": r.symbol, "phase": "Tender period",
                     "start": trading_end, "end": r.expiry_date + pd.Timedelta(days=1),
                     "status": r.status})
    return pd.DataFrame(rows)
