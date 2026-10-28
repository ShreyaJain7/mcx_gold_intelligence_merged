"""
MCX Gold Intelligence - executive dashboard (Role C).

Run from the project root::

    streamlit run app/main.py

Data flow
---------
Person A's parquet  ->  src.pipeline.prepare_market()   (validate + normalise)
                    ->  src.pipeline.run_pair()          (signals, backtest, attribution)
                    ->  this app                         (render only; no quant logic here)

The app computes live (cached) so sidebar parameter changes are reflected
instantly; ``python -m src.pipeline`` produces the same numbers as files for
the submission / audit trail.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # make ``src`` importable
sys.path.insert(0, str(ROOT / "app"))   # make ``charts`` importable

import charts  # noqa: E402
from src.config import CONTRACT_SPECS, DEFAULT_PAIRS, CalendarConfig, StrategyConfig  # noqa: E402
from src.lifecycle import build_contract_calendar, calendar_segments  # noqa: E402
from src.pipeline import PairRun, available_pairs, prepare_market, run_pair  # noqa: E402
from src.quant_engine import continuous_normalized_prices  # noqa: E402
from src.storage import save_frame  # noqa: E402
from src.synthetic_data import generate_synthetic_bhavcopy  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("MCX_DATA_PATH",
                                   ROOT / "data/processed/normalized_bhavcopy.parquet"))

st.set_page_config(page_title="MCX Gold Intelligence", page_icon="🪙", layout="wide",
                   initial_sidebar_state="expanded")

# Dense, calm styling: tighter paddings, quiet metric cards, alert boxes.
st.markdown("""
<style>
:root { --gold:#c9a96e; --gold-bright:#e4c78c; --ink:#0b100f; --panel:#111816; --line:#2a342f; --muted:#9da79f; }
.stApp { background: radial-gradient(ellipse at 72% 0%, #20251e 0%, #0b100f 52%); color:#eee8dc; }
.block-container { padding-top:1.15rem; padding-bottom:1.8rem; max-width:1700px; }
[data-testid="stHeader"] { background:rgba(8,12,11,.88); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111613,#090d0c); border-right:1px solid #29322d; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--gold-bright); }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(28,35,31,.96),rgba(14,19,17,.96)); border:1px solid #343d35; border-radius:12px; padding:.8rem 1rem; box-shadow:0 8px 24px rgba(0,0,0,.12); }
[data-testid="stMetricLabel"] { color:#b7b9ac; font-size:.78rem; }
[data-testid="stMetricValue"] { color:#f1dfb8; font-family:Georgia,serif; font-size:1.65rem; }
[data-testid="stMetricDelta"] { font-size:.78rem; }
[data-testid="stPlotlyChart"] { background:#101613; border:1px solid #29342e; border-radius:12px; padding:.35rem .35rem .1rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#2b352f !important; border-radius:12px !important; }
.stButton>button { border:1px solid #8e774e; border-radius:9px; }
hr { border-color:#2a342f; }
.alert { border-radius:10px; padding:.75rem 1rem; margin:.2rem 0 .6rem; border:1px solid #303a33; background:#111815; }
.alert .head { font-weight:600; font-size:1.02rem; color:#e6d2a6; }
.alert .sub { color:#b5bcb4; font-size:.9rem; margin-top:.15rem; }
.alert.neutral { border-left:4px solid #7aa68a; }
.alert.quiet { border-left:4px solid #77817a; }
.alert.opp { border-left:4px solid #d4aa62; background:rgba(201,169,110,.09); }
.gate { display:inline-block; margin:.25rem .6rem 0 0; font-size:.85rem; color:#c9c9bd; }
.mono { font-variant-numeric:tabular-nums; }
.goldlens-kicker { color:#c9a96e; letter-spacing:.18em; text-transform:uppercase; font-size:.7rem; }
.goldlens-title { font-family:Georgia,'Times New Roman',serif; color:#f0dfbc; font-size:2.5rem; line-height:1.05; margin:.15rem 0 .35rem; }
.goldlens-subtitle { color:#aeb7ae; font-size:.94rem; margin-bottom:.8rem; }
div[data-testid="stDataFrame"] { border:1px solid #303a33; border-radius:9px; }
@media (max-width: 900px) { .goldlens-title {font-size:2rem;} .block-container {padding-left:1rem;padding-right:1rem;} }
</style>
""", unsafe_allow_html=True)


# =========================================================================== #
# Formatting helpers                                                          #
# =========================================================================== #

def fmt_inr(x: float, decimals: int = 0) -> str:
    """₹ with Indian digit grouping (12,34,567)."""
    if x is None or not np.isfinite(x):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    whole, _, frac = f"{x:.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if decimals else "")


def fmt_num(x: float, fmt: str = "{:.2f}") -> str:
    return "—" if x is None or not np.isfinite(x) else fmt.format(x)


# =========================================================================== #
# Cached data access (the only place the app touches the quant layer)         #
# =========================================================================== #

def _mtime(path: Path) -> float:
    for p in (path, path.with_suffix(".csv")):
        if p.exists():
            return p.stat().st_mtime
    return -1.0


@st.cache_data(show_spinner="Loading and normalising bhavcopy…")
def load_market(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache on change
    return prepare_market(path)


@st.cache_data(show_spinner="Running walk-forward signal engine…")
def compute_pair(path: str, mtime: float, leg_a: str, leg_b: str,
                 params: tuple[tuple[str, object], ...]) -> PairRun:
    cfg = dataclasses.replace(StrategyConfig(), **dict(params))
    return run_pair(load_market(path, mtime), leg_a, leg_b, cfg)


@st.cache_data(show_spinner=False)
def compute_prices(path: str, mtime: float, roll_dte: int) -> pd.DataFrame:
    return continuous_normalized_prices(load_market(path, mtime), roll_dte)


@st.cache_data(show_spinner="Scanning parameter grid…")
def sensitivity_grid(path: str, mtime: float, leg_a: str, leg_b: str,
                     params: tuple[tuple[str, object], ...]) -> pd.DataFrame:
    rows = []
    base = dict(params)
    for entry in (1.5, 2.0, 2.5, 3.0):
        for win in (40, 60, 90):
            p = {**base, "entry_z": entry, "z_window": win,
                 "z_min_periods": min(base.get("z_min_periods", 40), win)}
            run = compute_pair(path, mtime, leg_a, leg_b, tuple(sorted(p.items())))
            rows.append({"entry_z": entry, "z_window": win,
                         "Sharpe": run.attribution.sharpe_ratio,
                         "Net P&L (₹)": run.result.summary["cumulative_pnl_inr"],
                         "Trades": run.result.summary["n_trades"]})
    return pd.DataFrame(rows)


# =========================================================================== #
# Sidebar                                                                     #
# =========================================================================== #

with st.sidebar:
    st.markdown("<div style='font-family:Georgia,serif;font-size:1.65rem;color:#e4c78c'>✧ GoldLens</div><div style='font-size:.72rem;color:#9da79f;margin-bottom:1.2rem'>COMMODITY INTELLIGENCE</div>", unsafe_allow_html=True)
    st.header("Data")
    data_path = Path(st.text_input("Bhavcopy parquet (Person A output)", str(DEFAULT_DATA)))
    if _mtime(data_path) < 0:
        st.warning("File not found.")
        if st.button("Generate synthetic demo data", type="primary"):
            save_frame(generate_synthetic_bhavcopy(), data_path)
            st.rerun()
        st.stop()

    market = load_market(str(data_path), _mtime(data_path))
    pairs = available_pairs(market) or list(DEFAULT_PAIRS)
    st.header("Pair")
    pair_label = st.selectbox("Spread (A − B)", [f"{a} − {b}" for a, b in pairs])
    leg_a, leg_b = [s.strip() for s in pair_label.split("−")]

    st.header("Signal")
    d = StrategyConfig()
    entry_z = st.slider("Entry |z|", 1.0, 4.0, d.entry_z, 0.1)
    exit_z = st.slider("Exit |z|", 0.0, 1.5, d.exit_z, 0.1)
    z_window = st.slider("Z-score window (days)", 20, 180, d.z_window, 5)
    detrend_window = st.slider("Term-structure fit window (days)", 60, 250, d.detrend_window, 10)
    with st.expander("Execution & risk"):
        execution_lag = st.select_slider("Execution lag (days)", [0, 1, 2], d.execution_lag)
        exit_dte = st.number_input("Force exit at DTE ≤", 0, 10, d.exit_dte)
        max_hold = st.number_input("Max holding (days)", 1, 60, d.max_holding_days)
        target_grams = st.number_input("Grams per leg", 8, 2000, int(d.target_grams), 8)
        capital = st.number_input("Capital base (₹)", 100_000, 50_000_000,
                                  int(d.capital_inr), 100_000)
        participation = st.slider("Max % of daily volume", 1, 50, int(d.max_participation * 100))

    params = tuple(sorted({
        "entry_z": float(entry_z), "exit_z": float(exit_z), "z_window": int(z_window),
        "z_min_periods": int(min(d.z_min_periods, z_window)),
        "detrend_window": int(detrend_window),
        "detrend_min_obs": int(min(d.detrend_min_obs, detrend_window)),
        "execution_lag": int(execution_lag), "exit_dte": int(exit_dte),
        "max_holding_days": int(max_hold), "target_grams": float(target_grams),
        "capital_inr": float(capital), "max_participation": participation / 100.0,
    }.items()))
    st.caption(f"{len(market):,} rows · {market['trade_date'].min():%d %b %Y} → "
               f"{market['trade_date'].max():%d %b %Y}")

mtime = _mtime(data_path)
run = compute_pair(str(data_path), mtime, leg_a, leg_b, params)
res, att, status = run.result, run.attribution, run.status
cfg = res.config
summ = res.summary

# =========================================================================== #
# Header                                                                      #
# =========================================================================== #

st.markdown("<div class='goldlens-kicker'>Commodity intelligence platform · Demo workspace</div>"
            "<div class='goldlens-title'>GoldLens</div>"
            "<div class='goldlens-subtitle'>Real-time-style gold intelligence for smarter decisions</div>",
            unsafe_allow_html=True)
st.caption(f"MCX relative-value monitor · **{leg_a} − {leg_b}** spread in ₹/g at 999 purity · "
           f"as of {status['as_of']:%d %b %Y} · all figures net of modelled MCX costs")

# --------------------------------------------------------------------------- #
# 1. Executive summary cards                                                  #
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
last20 = res.daily["net_pnl"].tail(20).sum()
c1.metric("Cumulative P&L (net)", fmt_inr(summ["cumulative_pnl_inr"]),
          delta=f"{fmt_inr(last20)} last 20d", delta_color="normal",
          help=f"{summ['n_trades']} round trips · hit rate {fmt_num(summ['hit_rate'] * 100, '{:.0f}')}% · "
               f"costs {fmt_inr(summ['total_costs_inr'])} · max drawdown {fmt_inr(summ['max_drawdown_inr'])}")
c2.metric("Sharpe ratio (ann.)", fmt_num(att.sharpe_ratio),
          delta=f"vol {fmt_num(att.annualized_vol * 100, '{:.1f}')}% on ₹{capital / 1e5:,.0f}L",
          delta_color="off",
          help="Mean / std of daily net returns × √252, from the first day a signal could exist.")
c3.metric("Alpha (ann.)", fmt_num(att.alpha_annualized * 100, "{:+.2f}%"),
          delta=f"p = {fmt_num(att.alpha_pvalue, '{:.3f}')}", delta_color="off",
          help=f"Intercept of strategy_return ~ gold_return. {att.estimator}.")
c4.metric("Beta to spot gold", fmt_num(att.beta, "{:+.3f}"),
          delta=f"R² = {fmt_num(att.r_squared, '{:.3f}')}", delta_color="off",
          help="Slope vs. front-month GOLDM (roll-adjusted) daily return. ≈0 means market-neutral.")

# --------------------------------------------------------------------------- #
# 3. Quiet alert panel                                                        #
# --------------------------------------------------------------------------- #
def render_alert(s: dict) -> None:
    z = s["zscore"]
    gates = "".join(f"<span class='gate'>{'✓' if ok else '✗'} {name}</span>"
                    for name, ok in s["gates"].items())
    pos = {1: "long spread", -1: "short spread", 0: "flat"}[s["position"]]
    ctx = (f"<span class='mono'>z = {fmt_num(z, '{:+.2f}')}</span> "
           f"(entry ±{cfg.entry_z:g}) · {s['contract_a']} vs {s['contract_b']} · "
           f"DTE {s['dte']} · book: {pos}")
    if s["state"] == "OPPORTUNITY":
        html = (f"<div class='alert opp'><div class='head'>⚠ Statistical dislocation — "
                f"{s['direction']}</div><div class='sub'>{ctx}</div><div>{gates}</div></div>")
    elif s["state"] == "NEUTRAL":
        html = (f"<div class='alert neutral'><div class='head'>● No dislocation — market neutral"
                f"</div><div class='sub'>{ctx}</div></div>")
    elif s["state"] == "BLOCKED":
        html = (f"<div class='alert quiet'><div class='head'>○ Threshold crossed but not actionable"
                f"</div><div class='sub'>{ctx}</div><div>{gates}</div></div>")
    else:
        html = (f"<div class='alert quiet'><div class='head'>○ Warming up — not enough history "
                f"for a point-in-time z-score</div><div class='sub'>{ctx}</div></div>")
    st.markdown(html, unsafe_allow_html=True)


render_alert(status)

others = [p for p in pairs if p != (leg_a, leg_b)]
other_runs = {f"{a} − {b}": compute_pair(str(data_path), mtime, a, b, params) for a, b in others}
live_other = {k: r.status for k, r in other_runs.items() if r.status["state"] == "OPPORTUNITY"}
if live_other:
    for k, s in live_other.items():
        st.markdown(f"<div class='alert opp'><div class='head'>⚠ Also dislocated: {k}</div>"
                    f"<div class='sub'>z = {s['zscore']:+.2f} · {s['direction']}</div></div>",
                    unsafe_allow_html=True)
else:
    st.caption("Other pairs: " + " · ".join(
        f"{k} z {fmt_num(r.status['zscore'], '{:+.2f}')}" for k, r in other_runs.items())
        + " — all quiet.")

# --------------------------------------------------------------------------- #
# 2. Charts                                                                   #
# --------------------------------------------------------------------------- #
left, right = st.columns(2, gap="medium")
with left:
    prices = compute_prices(str(data_path), mtime, cfg.roll_dte)
    mode = st.radio("Price view", ["Levels", "Premium vs GOLDM"], horizontal=True,
                    label_visibility="collapsed")
    syms = ["GOLDM", "GOLDGUINEA", "GOLDPETAL"] + \
        (["GOLDTEN"] if st.checkbox("Include GOLDTEN (short history)", value=False) else [])
    st.plotly_chart(charts.normalized_price_chart(
        prices, syms, "premium" if mode.startswith("Premium") else "level"),
        width="stretch", theme="streamlit")
with right:
    st.plotly_chart(charts.residual_zscore_chart(res.frame, res.trades, cfg.entry_z, cfg.exit_z),
                    width="stretch", theme="streamlit")

left, right = st.columns(2, gap="medium")
with left:
    show_bh = st.checkbox("Show buy & hold gold", value=True)
    st.plotly_chart(charts.equity_curve_chart(res.daily, res.benchmark, show_bh),
                    width="stretch", theme="streamlit")
    bh_total = res.benchmark["bh_equity"].iloc[-1]
    st.caption(f"Same gram exposure ({summ['grams_a']:.0f} g). Buy & hold: {fmt_inr(bh_total)} "
               f"with full gold beta; strategy: {fmt_inr(summ['cumulative_pnl_inr'])} "
               f"with beta {fmt_num(att.beta, '{:+.3f}')}. The comparison is about risk, not size.")

# --------------------------------------------------------------------------- #
# 4. Contract lifecycle calendar                                              #
# --------------------------------------------------------------------------- #
with right:
    cal = build_contract_calendar(market, CalendarConfig())
    show_syms = st.multiselect("Contracts", list(CONTRACT_SPECS), default=[leg_a, leg_b],
                               label_visibility="collapsed")
    visible = cal[cal["show"] & cal["symbol"].isin(show_syms)]
    if visible.empty:
        st.info("No active or recently expired contracts for the selected symbols.")
    else:
        st.plotly_chart(charts.lifecycle_chart(calendar_segments(visible), status["as_of"]),
                        width="stretch", theme="streamlit")
        with st.expander("Calendar table"):
            st.dataframe(visible[["symbol", "expiry_date", "listing_date", "tender_start",
                                  "status", "days_to_expiry", "latest_oi"]]
                         .rename(columns={"listing_date": "first traded (listing proxy)"}),
                         hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Detail drawers                                                              #
# --------------------------------------------------------------------------- #
d1, d2, d3 = st.columns(3)
with d1.expander(f"Trade blotter ({summ['n_trades']})"):
    if res.trades.empty:
        st.write("No trades under current parameters.")
    else:
        st.dataframe(res.trades[["direction", "signal_date", "entry_date", "exit_date",
                                 "entry_z", "exit_signal_z", "holding_days", "gross_pnl",
                                 "costs", "net_pnl", "exit_reason"]]
                     .style.format({"entry_z": "{:+.2f}", "exit_signal_z": "{:+.2f}",
                                    "gross_pnl": "₹{:,.0f}", "costs": "₹{:,.0f}",
                                    "net_pnl": "₹{:,.0f}"}),
                     hide_index=True, width="stretch")
        st.download_button("Download CSV", res.trades.to_csv(index=False),
                           f"trades_{res.pair_id}.csv", "text/csv")
with d2.expander("Attribution detail"):
    st.dataframe(pd.DataFrame({
        "Estimate": [att.alpha_daily, att.alpha_annualized, att.beta, att.r_squared,
                     att.information_ratio, att.n_obs],
        "t-stat": [att.alpha_tstat, np.nan, att.beta_tstat, np.nan, np.nan, np.nan],
        "p-value": [att.alpha_pvalue, np.nan, att.beta_pvalue, np.nan, np.nan, np.nan],
    }, index=["Alpha (daily)", "Alpha (annualised)", "Beta", "R²", "Information ratio",
              "Observations"]), width="stretch")
    st.caption(att.estimator + (f" · {att.note}" if att.note else ""))
with d3.expander("Parameter sensitivity"):
    st.caption("Sharpe across entry thresholds × z-windows. A robust edge should not "
               "live on a single cell.")
    if st.button("Run 12-cell grid"):
        grid = sensitivity_grid(str(data_path), mtime, leg_a, leg_b, params)
        st.dataframe(grid.pivot(index="entry_z", columns="z_window", values="Sharpe")
                     .style.format("{:.2f}"), width="stretch")

# --------------------------------------------------------------------------- #
# 5. Honest limitations & model assumptions                                   #
# --------------------------------------------------------------------------- #
days_by_sym = market.groupby("symbol")["trade_date"].nunique()
ten_days = int(days_by_sym.get("GOLDTEN", 0))
cost_share = (summ["total_costs_inr"] / summ["gross_pnl_inr"] * 100
              if summ["gross_pnl_inr"] > 0 else float("nan"))
warmup = cfg.detrend_min_obs + cfg.z_min_periods

with st.container(border=True):
    st.subheader("Limitations & model assumptions")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"""
**Liquidity**
- Entries require both legs to clear volume/OI floors and keep our size ≤ {cfg.max_participation:.0%} of the day's volume. **{summ['blocked_signals']}** threshold crossings were blocked by these gates for this pair.
- This pair trades {summ['lots_a']} lot(s) of {leg_a} against {summ['lots_b']} lot(s) of {leg_b}. GOLDPETAL / GOLDGUINEA books are thin, and a large lot count can move the price more than the tick-slippage proxy assumes.
- Exits are never blocked; on thin days they pay {cfg.costs.illiquid_slippage_multiplier:g}× slippage instead.

**Settlement vs. execution friction**
- Bhavcopy `close` is used as the fill price. MCX daily settlement price can differ from the last traded price, and a real order fills at bid/ask, not at close.
- Costs modelled: ₹{cfg.costs.brokerage_per_order:g}/order brokerage, exchange charges, SEBI fee, GST, CTT (sell), stamp duty (buy), tick slippage. Costs consumed **{fmt_num(cost_share, '{:.0f}')}%** of gross P&L. Rates are indicative; confirm against current circulars.
- Positions exit at DTE ≤ {cfg.exit_dte}, which can fall inside the tender window (see calendar); holding into tender implies delivery obligations and extra margins not modelled here.
- Legs are sized to {summ['grams_a']:.0f} g vs {summ['grams_b']:.0f} g by integer lots; GOLDM's 995 fineness means the hedge is not exactly equal in fine-gold content.
""")
    with b:
        st.markdown(f"""
**Parameter sensitivity**
- Results depend on entry/exit z, the z window ({cfg.z_window}d) and the term-structure window ({cfg.detrend_window}d). Use the sensitivity grid above; a single strong cell is a warning sign.
- The first ~{warmup} trading days per pair are warm-up with no signals. All statistics are walk-forward, but the parameter defaults were chosen by hand, which is a mild form of selection bias.

**Short GOLDTEN history**
- GOLDTEN has **{ten_days}** trading days in this file versus {int(days_by_sym.max())} for the longest series. Its z-scores rest on few observations and few expiry cycles; treat GOLDTEN pairs as indicative only.

**Model assumptions**
- Prices normalised to ₹/g at 999 purity using the quotation bases and purities in `src/config.py`. The `level` regressor absorbs a mis-specified purity factor, but verify the specs against MCX.
- "Spot gold" = roll-adjusted front-month {cfg.benchmark_symbol}; futures carry a basis to physical spot.
- Listing dates are the first date seen in the data; tender windows are {CalendarConfig().tender_business_days} business days before expiry, with exchange holidays ignored.
- Returns use a fixed ₹{capital / 1e5:,.0f} lakh capital base; margin calls and funding costs are not simulated.
""")
