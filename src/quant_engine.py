"""
Quant & signal engine (Role B).

Pipeline, in order:

1. ``load_bhavcopy`` / ``validate_bhavcopy`` - schema checks and hygiene on
   Person A's parquet.
2. ``normalize_prices`` - every quote converted to ₹/gram at 999 purity,
   including detection of GOLDGUINEA rows accidentally published per gram.
3. ``build_pair_frame`` - one row per trade date holding leg A's active
   contract and the leg-B contract whose expiry is *nearest* to it.
4. ``walk_forward_detrend`` - removes the mechanical term-structure component
   (roll-down / carry) from the raw spread with a rolling regression fitted
   strictly on past rows, leaving ``residual_spread``.
5. ``rolling_zscore`` - point-in-time z of the residual.
6. ``backtest_pair`` - event-driven simulation on the *actual held
   contracts* (no continuous-series roll artefacts), with MCX costs,
   liquidity gating and a full trade blotter.
7. ``benchmark_buy_and_hold`` - roll-adjusted front-month gold P&L and the
   daily "spot gold" return series used for attribution.

``run_strategy`` chains 3-7 for one pair and returns a ``StrategyResult``
that the dashboard and the batch pipeline both consume.

Look-ahead protection is structural, not a convention: every statistic used
at date t is computed from rows strictly before t (regression fit, z mean/std)
or from information known in advance (expiry calendar). ``tests/`` contains a
truncation test proving that appending future data never changes past signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CONTRACT_SPECS,
    SUPPORTED_SYMBOLS,
    CostModel,
    StrategyConfig,
)

logger = logging.getLogger(__name__)

#: Contract with Person A: the input parquet must provide exactly these columns.
REQUIRED_COLUMNS: dict[str, str] = {
    "trade_date": "datetime64[ns]",
    "symbol": "object",
    "expiry_date": "datetime64[ns]",
    "close": "float64",
    "volume": "int64",
    "open_interest": "int64",
}


# =========================================================================== #
# 1. Loading & validation                                                     #
# =========================================================================== #

class DataValidationError(ValueError):
    """Raised when the input bhavcopy cannot be used safely."""


def load_bhavcopy(path: str | Path) -> pd.DataFrame:
    """Read Person A's normalised bhavcopy and run ``validate_bhavcopy``.

    Parquet is the contract format; a CSV with the same columns is accepted
    as a convenience for environments without ``pyarrow``.
    """
    path = Path(path)
    if not path.exists():
        # Tolerate a CSV sibling written by the parquet->csv fallback.
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Bhavcopy not found at {path}")
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, parse_dates=["trade_date", "expiry_date"])
    return validate_bhavcopy(raw)


def validate_bhavcopy(raw: pd.DataFrame) -> pd.DataFrame:
    """Enforce the input schema and remove rows that would poison the model.

    Hygiene steps (each logged with a row count):
      * missing required columns -> hard error;
      * symbols upper-cased/stripped; unknown symbols dropped;
      * non-positive or missing closes dropped (bhavcopy prints 0 for no-trade);
      * rows traded after their own expiry dropped;
      * negative / missing volume and OI floored to 0;
      * exact duplicate keys (trade_date, symbol, expiry_date) de-duplicated,
        keeping the last occurrence (latest file revision wins).
    """
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise DataValidationError(f"Input is missing required columns: {sorted(missing)}")

    df = raw[list(REQUIRED_COLUMNS)].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize().astype("datetime64[ns]")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.normalize().astype("datetime64[ns]")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    for col in ("volume", "open_interest"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype("int64")

    n0 = len(df)
    unknown = ~df["symbol"].isin(SUPPORTED_SYMBOLS)
    if unknown.any():
        logger.warning("Dropping %d rows with unsupported symbols: %s",
                       unknown.sum(), sorted(df.loc[unknown, "symbol"].unique()))
    df = df[~unknown]

    bad_price = df["close"].isna() | (df["close"] <= 0)
    bad_dates = df["trade_date"].isna() | df["expiry_date"].isna() | (df["trade_date"] > df["expiry_date"])
    if bad_price.any():
        logger.warning("Dropping %d rows with missing/non-positive close", bad_price.sum())
    if bad_dates.any():
        logger.warning("Dropping %d rows with invalid or post-expiry dates", bad_dates.sum())
    df = df[~bad_price & ~bad_dates]

    key = ["trade_date", "symbol", "expiry_date"]
    dupes = df.duplicated(key, keep="last")
    if dupes.any():
        logger.warning("Dropping %d duplicate (trade_date, symbol, expiry_date) rows", dupes.sum())
    df = df[~dupes]

    if df.empty:
        raise DataValidationError("No usable rows remain after validation.")
    logger.info("Validated bhavcopy: %d -> %d rows", n0, len(df))
    return df.sort_values(key).reset_index(drop=True)


# =========================================================================== #
# 2. Price normalisation                                                      #
# =========================================================================== #

def normalize_prices(df: pd.DataFrame, tolerance: float = 0.25) -> pd.DataFrame:
    """Add ``norm_price`` (₹/gram at 999 purity) and ``days_to_expiry``.

    Conversion (see ``config.CONTRACT_SPECS``)::

        GOLDM      : (close / 10) * (999 / 995)
        GOLDGUINEA :  close / 8
        GOLDPETAL  :  close / 1
        GOLDTEN    :  close / 10

    Per-gram edge case
    ------------------
    Some vendor feeds publish GOLDGUINEA per gram instead of per 8 g. Dividing
    such a row by 8 would produce a price ~87% too low and a spurious
    "dislocation". For every symbol with ``alt_quote_grams`` we compare each
    candidate conversion against a same-day reference - the median ₹/g of the
    *other* symbols (forward-filled from earlier days only, so no look-ahead)
    - and switch to the alternative basis only when the default conversion is
    more than ``tolerance`` away from the reference *and* the alternative is
    within it. Corrected rows are flagged in ``quote_basis_fixed``.
    """
    out = df.copy()
    factor = out["symbol"].map({s: c.to_per_gram_999 for s, c in CONTRACT_SPECS.items()})
    out["norm_price"] = out["close"] * factor
    out["quote_basis_fixed"] = False

    for symbol, spec in CONTRACT_SPECS.items():
        if not spec.alt_quote_grams:
            continue
        mask = out["symbol"] == symbol
        if not mask.any():
            continue
        others = out.loc[~out["symbol"].eq(symbol)]
        if others.empty:
            logger.warning("No cross-symbol reference for %s; per-gram check skipped", symbol)
            continue
        ref = others.groupby("trade_date")["norm_price"].median().sort_index()
        # Reference on each of this symbol's dates, using only same-day-or-earlier data.
        ref_on = ref.reindex(out.loc[mask, "trade_date"].values, method="ffill").to_numpy()
        default = out.loc[mask, "norm_price"].to_numpy()
        purity_adj = CONTRACT_SPECS[symbol].to_per_gram_999 * spec.quote_grams  # = 999/purity
        close = out.loc[mask, "close"].to_numpy()

        with np.errstate(invalid="ignore", divide="ignore"):
            default_err = np.abs(np.log(default / ref_on))
            best = default.copy()
            best_err = default_err.copy()
            for grams in spec.alt_quote_grams:
                cand = close / grams * purity_adj
                cand_err = np.abs(np.log(cand / ref_on))
                better = (cand_err < best_err) & (cand_err <= np.log1p(tolerance)) \
                    & (default_err > np.log1p(tolerance))
                best = np.where(better, cand, best)
                best_err = np.where(better, cand_err, best_err)
        fixed = best != default
        if fixed.any():
            logger.warning("%s: re-based %d rows quoted on an alternative gram basis",
                           symbol, int(fixed.sum()))
        out.loc[mask, "norm_price"] = best
        out.loc[mask, "quote_basis_fixed"] = fixed
        # Re-express corrected rows on the standard quotation basis too: the
        # backtest marks P&L on ``close``, so leaving a per-gram close in place
        # would book a fake ~87% price move on the held contract.
        std_close = best / CONTRACT_SPECS[symbol].to_per_gram_999
        out.loc[mask, "close"] = np.where(fixed, std_close, close)

    out["close_raw"] = df["close"]

    out["days_to_expiry"] = (out["expiry_date"] - out["trade_date"]).dt.days.astype("int64")
    return out


# =========================================================================== #
# 3. Contract selection & pair construction                                   #
# =========================================================================== #

def active_contracts(df: pd.DataFrame, symbol: str, roll_dte: int) -> pd.DataFrame:
    """Front ("active") contract of ``symbol`` for every trade date.

    Active = nearest expiry with ``days_to_expiry > roll_dte``. Because the
    rule depends only on the (pre-announced) expiry calendar, it is
    look-ahead free.
    """
    sub = df[(df["symbol"] == symbol) & (df["days_to_expiry"] > roll_dte)]
    if sub.empty:
        return sub.set_index("trade_date")
    idx = sub.groupby("trade_date")["expiry_date"].idxmin()
    return sub.loc[idx].set_index("trade_date").sort_index()


def continuous_normalized_prices(df: pd.DataFrame, roll_dte: int,
                                 symbols: tuple[str, ...] = SUPPORTED_SYMBOLS) -> pd.DataFrame:
    """Wide frame (date x symbol) of active-contract ₹/g@999 prices, for charts."""
    cols = {}
    for s in symbols:
        act = active_contracts(df, s, roll_dte)
        if not act.empty:
            cols[s] = act["norm_price"]
    return pd.DataFrame(cols).sort_index()


def build_pair_frame(df: pd.DataFrame, leg_a: str, leg_b: str, roll_dte: int) -> pd.DataFrame:
    """Align leg A's active contract with the leg-B contract of nearest expiry.

    MCX gold minis and the small contracts do not share an expiry calendar
    (e.g. a 5th-of-month cycle vs. a month-end cycle), so identical expiries
    are not guaranteed. On each date we take A's active contract and, among
    B's contracts with DTE > ``roll_dte``, the one minimising
    |expiry_B - expiry_A| (ties -> earlier expiry). ``expiry_gap_days`` is
    carried forward into the term-structure regression.
    """
    a = active_contracts(df, leg_a, roll_dte)
    if a.empty:
        raise ValueError(f"No active contracts for leg A = {leg_a}")
    b_all = df[(df["symbol"] == leg_b) & (df["days_to_expiry"] > roll_dte)]
    if b_all.empty:
        raise ValueError(f"No active contracts for leg B = {leg_b}")

    keep = ["expiry_date", "close", "norm_price", "volume", "open_interest",
            "days_to_expiry", "quote_basis_fixed"]
    a = a[keep].add_suffix("_a").reset_index()
    b = b_all[["trade_date", *keep]]
    m = a.merge(b, on="trade_date", how="inner")
    m["abs_gap"] = (m["expiry_date"] - m["expiry_date_a"]).dt.days.abs()
    m = m.sort_values(["trade_date", "abs_gap", "expiry_date"])
    m = m.drop_duplicates("trade_date", keep="first")
    m = m.rename(columns={c: f"{c}_b" for c in keep}).drop(columns="abs_gap")

    pair = m.set_index("trade_date").sort_index()
    pair["expiry_gap_days"] = (pair["expiry_date_b"] - pair["expiry_date_a"]).dt.days
    pair["dte"] = pair[["days_to_expiry_a", "days_to_expiry_b"]].min(axis=1)
    pair["spread"] = pair["norm_price_a"] - pair["norm_price_b"]
    pair["mid_price"] = 0.5 * (pair["norm_price_a"] + pair["norm_price_b"])
    pair.attrs.update(leg_a=leg_a, leg_b=leg_b)
    return pair


# =========================================================================== #
# 4. Walk-forward term-structure removal                                      #
# =========================================================================== #

def _design_matrix(pair: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    """Regressors explaining the *mechanical* part of the spread.

    * ``dte``    - days to expiry of leg A: roll-down as the contract ages
                   (this is the regression the problem statement asks for).
    * ``carry``  - mid_price x expiry_gap_days / 365: cost-of-carry between
                   two legs whose expiries differ. Its coefficient is the
                   implied annualised carry rate, so it absorbs the jump that
                   occurs when the nearest-expiry match switches cycles.
    * ``level``  - mid_price: absorbs any purity / quotation-basis premium
                   that scales with the gold price (e.g. if a purity
                   assumption in the spec is off, the bias grows with price
                   and would otherwise masquerade as a trending signal).
    """
    builders = {
        "dte": lambda p: p["days_to_expiry_a"].to_numpy(float),
        "carry": lambda p: (p["mid_price"] * p["expiry_gap_days"] / 365.0).to_numpy(float),
        "level": lambda p: p["mid_price"].to_numpy(float),
    }
    unknown = set(features) - set(builders)
    if unknown:
        raise ValueError(f"Unknown detrend features: {sorted(unknown)}")
    return np.column_stack([builders[f](pair) for f in features])


def walk_forward_detrend(pair: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Add ``fitted_spread`` and ``residual_spread`` using only past data.

    For each date t a ridge-stabilised OLS of spread on the features is fitted
    on rows [t - window, t - 1] (strictly before t) and evaluated at t. The
    residual at t is therefore a genuine out-of-sample forecast error: the
    model never sees the value it is judging.

    Features are standardised inside each window and a tiny ridge penalty
    keeps the solve stable when regressors are collinear over a short window
    (e.g. a constant expiry gap). The intercept is handled by centring.
    """
    out = pair.copy()
    X = _design_matrix(out, cfg.detrend_features)
    y = out["spread"].to_numpy(float)
    n, k = X.shape
    fitted = np.full(n, np.nan)
    coefs = np.full((n, k), np.nan)

    for t in range(n):
        lo = max(0, t - cfg.detrend_window)
        Xtr, ytr = X[lo:t], y[lo:t]
        ok = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
        if ok.sum() < cfg.detrend_min_obs or not np.isfinite(X[t]).all():
            continue
        Xtr, ytr = Xtr[ok], ytr[ok]
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (Xtr - mu) / sd
        ybar = ytr.mean()
        lam = cfg.detrend_ridge * len(ytr)
        beta = np.linalg.solve(Z.T @ Z + lam * np.eye(k), Z.T @ (ytr - ybar))
        fitted[t] = ybar + ((X[t] - mu) / sd) @ beta
        coefs[t] = beta / sd  # coefficients in original units, for diagnostics

    out["fitted_spread"] = fitted
    out["residual_spread"] = out["spread"] - out["fitted_spread"]
    for j, f in enumerate(cfg.detrend_features):
        out[f"coef_{f}"] = coefs[:, j]
    return out


# =========================================================================== #
# 5. Rolling z-score                                                          #
# =========================================================================== #

def rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.DataFrame:
    """Point-in-time z-score: ``(x_t - mean_{<t}) / std_{<t}``.

    ``shift(1)`` before rolling means today's value is compared with a
    distribution it is not part of - no same-bar contamination, no look-ahead.
    """
    hist = series.shift(1).rolling(window, min_periods=min_periods)
    mean, std = hist.mean(), hist.std(ddof=1)
    std = std.where(std > 1e-12)
    return pd.DataFrame({"rolling_mean": mean, "rolling_std": std,
                         "zscore": (series - mean) / std})


# =========================================================================== #
# 6. Liquidity screen & sizing                                                #
# =========================================================================== #

def leg_lots(symbol: str, target_grams: float) -> int:
    """Integer lots giving the closest gram exposure to ``target_grams``."""
    return max(1, int(round(target_grams / CONTRACT_SPECS[symbol].lot_grams)))


def liquidity_flags(pair: pd.DataFrame, leg_a: str, leg_b: str,
                    lots_a: int, lots_b: int, cfg: StrategyConfig) -> pd.Series:
    """True where *both* legs pass volume, OI and participation thresholds.

    Uses the signal day's own volume/OI (known at that close), never the
    execution day's, so the screen itself is point-in-time.
    """
    def leg_ok(sym: str, suffix: str, lots: int) -> pd.Series:
        vol = pair[f"volume_{suffix}"]
        oi = pair[f"open_interest_{suffix}"]
        return ((vol >= cfg.min_volume.get(sym, 0))
                & (oi >= cfg.min_open_interest.get(sym, 0))
                & (lots <= cfg.max_participation * vol))
    return leg_ok(leg_a, "a", lots_a) & leg_ok(leg_b, "b", lots_b)


# =========================================================================== #
# 7. Event-driven backtest on held contracts                                  #
# =========================================================================== #

@dataclass
class _Position:
    direction: int            # +1 long spread (buy A / sell B), -1 short spread
    expiry_a: pd.Timestamp
    expiry_b: pd.Timestamp
    mark_a: float             # last quoted price used for MTM, leg A
    mark_b: float
    entry_date: pd.Timestamp
    signal_date: pd.Timestamp
    entry_z: float
    entry_spread: float
    entry_index: int
    gross_pnl: float = 0.0
    costs: float = 0.0


@dataclass
class _PendingOrder:
    kind: str                 # "enter" | "exit"
    due_index: int
    signal_date: pd.Timestamp
    direction: int = 0
    expiry_a: pd.Timestamp | None = None
    expiry_b: pd.Timestamp | None = None
    z: float = np.nan
    reason: str = ""


class _PriceBook:
    """Forward-filled close lookup for any (symbol, expiry) on any date.

    Forward-fill only uses *past* prints, so marking a stale contract at its
    last traded price is look-ahead free.
    """

    def __init__(self, df: pd.DataFrame, symbols: tuple[str, ...], dates: pd.DatetimeIndex):
        sub = df[df["symbol"].isin(symbols)]
        self._px: dict[tuple[str, pd.Timestamp], pd.Series] = {}
        self._traded: dict[tuple[str, pd.Timestamp], pd.Series] = {}
        for (sym, exp), g in sub.groupby(["symbol", "expiry_date"]):
            s = g.set_index("trade_date")["close"].sort_index()
            full = s.reindex(s.index.union(dates)).ffill()
            self._px[(sym, exp)] = full
            self._traded[(sym, exp)] = pd.Series(True, index=s.index)

    def price(self, sym: str, exp: pd.Timestamp, date: pd.Timestamp) -> float:
        s = self._px.get((sym, pd.Timestamp(exp)))
        return float(s.get(date, np.nan)) if s is not None else np.nan

    def traded(self, sym: str, exp: pd.Timestamp, date: pd.Timestamp) -> bool:
        s = self._traded.get((sym, pd.Timestamp(exp)))
        return bool(s is not None and date in s.index)


def backtest_pair(frame: pd.DataFrame, df: pd.DataFrame, leg_a: str, leg_b: str,
                  cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the mean-reversion strategy on the pair frame.

    Rules
    -----
    Entry (decided at close t, executed at close t + execution_lag):
        z > +entry_z  -> short spread (sell A, buy B)
        z < -entry_z  -> long  spread (buy A, sell B)
        only if both legs pass the liquidity screen at t, DTE > exit_dte +
        entry_dte_buffer, and the same contract pair is still active on the
        execution day.
    Exit, z-driven (decided at t, executed at t + lag):
        |z| < exit_z                          -> "z_reverted"
        sign(z) flipped vs. entry             -> "z_sign_flip"
        |z| > stop_z in the adverse direction -> "z_stop"
    Exit, calendar-driven (known in advance -> executed at close t):
        min DTE of held legs <= exit_dte      -> "dte_exit"
        held pair no longer the active pair   -> "contract_roll"
        holding period >= max_holding_days    -> "max_holding"

    P&L is marked on the *specific contracts held* each day, so no
    continuous-contract roll gap ever enters the equity curve.

    Returns
    -------
    daily : DataFrame indexed by trade_date with position, gross_pnl, costs,
            net_pnl, equity, strategy_return and an ``event`` label.
    trades: one row per round trip (the blotter).
    """
    specs = CONTRACT_SPECS
    cm: CostModel = cfg.costs
    lots_a, lots_b = leg_lots(leg_a, cfg.target_grams), leg_lots(leg_b, cfg.target_grams)
    mult_a, mult_b = specs[leg_a].multiplier, specs[leg_b].multiplier
    dates = frame.index
    book = _PriceBook(df, (leg_a, leg_b), dates)

    z = frame["zscore"].to_numpy(float)
    liquid = frame["liquid"].to_numpy(bool)
    dte = frame["dte"].to_numpy(float)
    exp_a = frame["expiry_date_a"].to_numpy()
    exp_b = frame["expiry_date_b"].to_numpy()
    spread = frame["spread"].to_numpy(float)

    pos: _Position | None = None
    pending: _PendingOrder | None = None
    trades: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def leg_orders_cost(direction: int, opening: bool, pa: float, pb: float,
                        illiq_a: bool, illiq_b: bool) -> float:
        # Long spread opening = buy A / sell B; closing reverses the sides.
        buy_a = (direction > 0) == opening
        side_a, side_b = ("buy", "sell") if buy_a else ("sell", "buy")
        return (cm.order_cost(leg_a, pa, lots_a, side_a, illiq_a)
                + cm.order_cost(leg_b, pb, lots_b, side_b, illiq_b))

    def open_position(i: int, order: _PendingOrder) -> float:
        nonlocal pos
        t = dates[i]
        pa = book.price(leg_a, order.expiry_a, t)
        pb = book.price(leg_b, order.expiry_b, t)
        cost = leg_orders_cost(order.direction, True, pa, pb, False, False)
        pos = _Position(order.direction, order.expiry_a, order.expiry_b, pa, pb,
                        entry_date=t, signal_date=order.signal_date, entry_z=order.z,
                        entry_spread=spread[i], entry_index=i, costs=cost)
        return cost

    def close_position(i: int, reason: str, signal_z: float,
                       signal_date: pd.Timestamp) -> float:
        nonlocal pos
        assert pos is not None
        t = dates[i]
        pa = book.price(leg_a, pos.expiry_a, t)
        pb = book.price(leg_b, pos.expiry_b, t)
        # Exits are never blocked, but a stale / thin leg pays extra slippage.
        illiq_a = not book.traded(leg_a, pos.expiry_a, t) or not liquid[i]
        illiq_b = not book.traded(leg_b, pos.expiry_b, t) or not liquid[i]
        cost = leg_orders_cost(pos.direction, False, pa, pb, illiq_a, illiq_b)
        pos.costs += cost
        trades.append({
            "pair": f"{leg_a}-{leg_b}",
            "direction": "LONG_SPREAD" if pos.direction > 0 else "SHORT_SPREAD",
            "signal_date": pos.signal_date, "entry_date": pos.entry_date,
            "exit_signal_date": signal_date, "exit_date": t,
            "contract_a": f"{leg_a} {pd.Timestamp(pos.expiry_a):%d-%b-%Y}",
            "contract_b": f"{leg_b} {pd.Timestamp(pos.expiry_b):%d-%b-%Y}",
            "lots_a": lots_a, "lots_b": lots_b,
            "entry_z": pos.entry_z,
            "exit_signal_z": signal_z,   # z that triggered the exit (signal day)
            "exit_z": z[i],              # z on the execution day
            "entry_spread": pos.entry_spread, "exit_spread": spread[i],
            "holding_days": i - pos.entry_index,
            "gross_pnl": pos.gross_pnl, "costs": pos.costs,
            "net_pnl": pos.gross_pnl - pos.costs, "exit_reason": reason,
        })
        pos = None
        return cost

    for i, t in enumerate(dates):
        gross = cost = 0.0
        events: list[str] = []

        # 1) Mark-to-market the position carried into today.
        if pos is not None:
            pa = book.price(leg_a, pos.expiry_a, t)
            pb = book.price(leg_b, pos.expiry_b, t)
            if np.isfinite(pa) and np.isfinite(pb):
                gross = pos.direction * (lots_a * mult_a * (pa - pos.mark_a)
                                         - lots_b * mult_b * (pb - pos.mark_b))
                pos.mark_a, pos.mark_b = pa, pb
                pos.gross_pnl += gross

        # 2) Execute any signal-driven order that has come due.
        if pending is not None and i >= pending.due_index:
            order, pending = pending, None
            if order.kind == "exit" and pos is not None:
                cost += close_position(i, order.reason, order.z, order.signal_date)
                events.append(f"EXIT:{order.reason}")
            elif order.kind == "enter" and pos is None:
                same_pair = (exp_a[i] == order.expiry_a) and (exp_b[i] == order.expiry_b)
                if same_pair and dte[i] > cfg.exit_dte:
                    cost += open_position(i, order)
                    events.append("ENTER_LONG" if order.direction > 0 else "ENTER_SHORT")
                else:
                    events.append("ENTRY_CANCELLED_ROLL")

        # 3) Calendar-driven exits: known in advance, so executed today.
        if pos is not None:
            held_dte = min((pd.Timestamp(pos.expiry_a) - t).days,
                           (pd.Timestamp(pos.expiry_b) - t).days)
            reason = None
            if held_dte <= cfg.exit_dte:
                reason = "dte_exit"
            elif exp_a[i] != pos.expiry_a or exp_b[i] != pos.expiry_b:
                reason = "contract_roll"
            elif i - pos.entry_index >= cfg.max_holding_days:
                reason = "max_holding"
            if reason:
                cost += close_position(i, reason, z[i], t)
                events.append(f"EXIT:{reason}")
                pending = None

        # 4) Signal evaluation at today's close -> order for t + lag.
        zi = z[i]
        if np.isfinite(zi) and pending is None:
            if pos is not None:
                d = pos.direction
                reason = None
                if -d * zi < cfg.exit_z:  # reverted inside the band, or crossed zero
                    reason = "z_sign_flip" if np.sign(zi) == d else "z_reverted"
                elif cfg.stop_z is not None and -d * zi > cfg.stop_z:
                    reason = "z_stop"
                if reason:
                    pending = _PendingOrder("exit", i + cfg.execution_lag, t, reason=reason, z=zi)
            elif abs(zi) > cfg.entry_z and liquid[i] and dte[i] > cfg.exit_dte + cfg.entry_dte_buffer:
                pending = _PendingOrder("enter", i + cfg.execution_lag, t,
                                        direction=-int(np.sign(zi)),
                                        expiry_a=exp_a[i], expiry_b=exp_b[i], z=zi)
            elif abs(zi) > cfg.entry_z:
                events.append("SIGNAL_BLOCKED")

        # 4b) Zero-lag configuration executes immediately.
        if pending is not None and pending.due_index <= i:
            order, pending = pending, None
            if order.kind == "exit" and pos is not None:
                cost += close_position(i, order.reason, zi, t)
                events.append(f"EXIT:{order.reason}")
            elif order.kind == "enter" and pos is None:
                cost += open_position(i, order)
                events.append("ENTER_LONG" if order.direction > 0 else "ENTER_SHORT")

        rows.append({"trade_date": t,
                     "position": 0 if pos is None else pos.direction,
                     "gross_pnl": gross, "costs": cost,
                     "event": ";".join(events)})

    daily = pd.DataFrame(rows).set_index("trade_date")
    daily["net_pnl"] = daily["gross_pnl"] - daily["costs"]
    daily["equity"] = daily["net_pnl"].cumsum()
    daily["strategy_return"] = daily["net_pnl"] / cfg.capital_inr
    daily["drawdown"] = daily["equity"] - daily["equity"].cummax()
    trades_df = pd.DataFrame(trades)
    return daily, trades_df


# =========================================================================== #
# 8. Benchmark: roll-adjusted buy & hold gold                                 #
# =========================================================================== #

def benchmark_buy_and_hold(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Always-long front-month gold, same gram exposure as one strategy leg.

    Holds the active ``benchmark_symbol`` contract and rolls when it stops
    being active (paying round-trip roll costs). Returns daily ₹ P&L,
    cumulative P&L and ``gold_return`` - the daily return of the *held*
    contract, which is the spot-gold proxy used for alpha/beta attribution
    (roll gaps never contaminate it because each return is measured on one
    contract).
    """
    sym = cfg.benchmark_symbol
    spec = CONTRACT_SPECS[sym]
    lots = leg_lots(sym, cfg.target_grams)
    act = active_contracts(df, sym, cfg.roll_dte)
    dates = act.index
    book = _PriceBook(df, (sym,), dates)

    held_exp, mark = None, np.nan
    rows = []
    for t in dates:
        pnl = cost = 0.0
        ret = np.nan
        if held_exp is not None:
            px = book.price(sym, held_exp, t)
            if np.isfinite(px) and np.isfinite(mark):
                pnl = lots * spec.multiplier * (px - mark)
                ret = px / mark - 1.0
                mark = px
        new_exp = act.at[t, "expiry_date"]
        if held_exp is None or new_exp != held_exp:
            if held_exp is not None:  # roll: sell old, buy new
                cost += cfg.costs.order_cost(sym, mark, lots, "sell")
            held_exp = new_exp
            mark = book.price(sym, held_exp, t)
            cost += cfg.costs.order_cost(sym, mark, lots, "buy")
        rows.append({"trade_date": t, "bh_pnl": pnl - cost, "gold_return": ret,
                     "gold_norm_price": act.at[t, "norm_price"]})
    out = pd.DataFrame(rows).set_index("trade_date")
    out["bh_equity"] = out["bh_pnl"].cumsum()
    return out


# =========================================================================== #
# 9. Orchestration                                                            #
# =========================================================================== #

@dataclass
class StrategyResult:
    """Everything the dashboard needs for one pair."""

    leg_a: str
    leg_b: str
    config: StrategyConfig
    frame: pd.DataFrame          # pair frame + residual + z + liquidity
    daily: pd.DataFrame          # daily P&L / equity / returns
    trades: pd.DataFrame         # round-trip blotter
    benchmark: pd.DataFrame      # buy & hold + gold_return
    lots: tuple[int, int]
    data_end: pd.Timestamp       # latest trade_date anywhere in the input file
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        return f"{self.leg_a}-{self.leg_b}"


def run_strategy(df_norm: pd.DataFrame, leg_a: str, leg_b: str,
                 cfg: StrategyConfig | None = None) -> StrategyResult:
    """End-to-end run for one pair on an already *normalised* bhavcopy."""
    cfg = cfg or StrategyConfig()
    for leg in (leg_a, leg_b):
        if leg not in CONTRACT_SPECS:
            raise ValueError(f"Unsupported symbol {leg}")
    if leg_a == leg_b:
        raise ValueError("Pair legs must differ")

    frame = build_pair_frame(df_norm, leg_a, leg_b, cfg.roll_dte)
    frame = walk_forward_detrend(frame, cfg)
    frame = frame.join(rolling_zscore(frame["residual_spread"], cfg.z_window, cfg.z_min_periods))
    lots_a, lots_b = leg_lots(leg_a, cfg.target_grams), leg_lots(leg_b, cfg.target_grams)
    frame["liquid"] = liquidity_flags(frame, leg_a, leg_b, lots_a, lots_b, cfg)

    daily, trades = backtest_pair(frame, df_norm, leg_a, leg_b, cfg)
    bench = benchmark_buy_and_hold(df_norm, cfg)

    result = StrategyResult(leg_a, leg_b, cfg, frame, daily, trades, bench,
                            (lots_a, lots_b), data_end=df_norm["trade_date"].max())
    result.summary = summarize(result)
    return result


def summarize(res: StrategyResult) -> dict[str, Any]:
    """Headline trade statistics (attribution metrics live in attribution.py)."""
    t, d = res.trades, res.daily
    spec_a, spec_b = CONTRACT_SPECS[res.leg_a], CONTRACT_SPECS[res.leg_b]
    grams_a, grams_b = res.lots[0] * spec_a.lot_grams, res.lots[1] * spec_b.lot_grams
    n = len(t)
    return {
        "pair": res.pair_id,
        "as_of": str(d.index.max().date()) if len(d) else None,
        "first_date": str(d.index.min().date()) if len(d) else None,
        "n_days": int(len(d)),
        "n_trades": int(n),
        "hit_rate": float((t["net_pnl"] > 0).mean()) if n else float("nan"),
        "avg_holding_days": float(t["holding_days"].mean()) if n else float("nan"),
        "cumulative_pnl_inr": float(d["equity"].iloc[-1]) if len(d) else 0.0,
        "gross_pnl_inr": float(d["gross_pnl"].sum()),
        "total_costs_inr": float(d["costs"].sum()),
        "max_drawdown_inr": float(d["drawdown"].min()) if len(d) else 0.0,
        "time_in_market": float((d["position"] != 0).mean()) if len(d) else 0.0,
        "lots_a": res.lots[0], "lots_b": res.lots[1],
        "grams_a": grams_a, "grams_b": grams_b,
        "exit_reasons": t["exit_reason"].value_counts().to_dict() if n else {},
        "blocked_signals": int(d["event"].str.contains("SIGNAL_BLOCKED").sum()),
        "quote_basis_fixes": int(res.frame[["quote_basis_fixed_a", "quote_basis_fixed_b"]]
                                 .sum().sum()),
    }


def current_status(res: StrategyResult, max_staleness_days: int = 5) -> dict[str, Any]:
    """State for the quiet alert panel, evaluated on the latest row only.

    States
    ------
    WARMUP       - not enough history for a z-score yet.
    NEUTRAL      - |z| <= entry threshold: "No dislocation / market neutral".
    BLOCKED      - |z| beyond threshold but a gate fails (liquidity, DTE,
                   stale data) - shown quietly, never as an alert.
    OPPORTUNITY  - |z| beyond threshold and every gate passes.
    """
    cfg = res.config
    f = res.frame
    last = f.iloc[-1]
    as_of = f.index[-1]
    data_end = res.data_end  # a pair that stopped updating is stale vs. the whole file
    z = float(last["zscore"]) if pd.notna(last["zscore"]) else float("nan")
    gates = {
        "Z-score available": bool(np.isfinite(z)),
        f"|z| > {cfg.entry_z:g}": bool(np.isfinite(z) and abs(z) > cfg.entry_z),
        "Both legs liquid": bool(last["liquid"]),
        f"DTE > {cfg.exit_dte + cfg.entry_dte_buffer}": bool(last["dte"] > cfg.exit_dte + cfg.entry_dte_buffer),
        "Data fresh": bool((data_end - as_of).days <= max_staleness_days),
    }
    if not gates["Z-score available"]:
        state = "WARMUP"
    elif not gates[f"|z| > {cfg.entry_z:g}"]:
        state = "NEUTRAL"
    elif all(gates.values()):
        state = "OPPORTUNITY"
    else:
        state = "BLOCKED"
    direction = None
    if np.isfinite(z) and abs(z) > cfg.entry_z:
        direction = (f"SHORT spread: sell {res.leg_a}, buy {res.leg_b}" if z > 0
                     else f"LONG spread: buy {res.leg_a}, sell {res.leg_b}")
    return {
        "state": state, "as_of": as_of, "zscore": z,
        "residual_spread": float(last["residual_spread"]),
        "spread": float(last["spread"]), "dte": int(last["dte"]),
        "contract_a": f"{res.leg_a} {last['expiry_date_a']:%d-%b-%Y}",
        "contract_b": f"{res.leg_b} {last['expiry_date_b']:%d-%b-%Y}",
        "position": int(res.daily["position"].iloc[-1]),
        "direction": direction, "gates": gates,
    }
