"""
Plotly figure builders for the dashboard (Role C).

Pure functions: DataFrames in, ``go.Figure`` out. No Streamlit calls here, so
every chart can be unit-tested or reused in a notebook / static report.

Colour rules
------------
* Colour follows the *entity*, never its rank: each symbol keeps the same hue
  in every chart, whatever subset is displayed.
* Buy/sell markers carry meaning by *shape* (triangle up / down) plus legend,
  not by colour alone.
* One y-axis per panel. Residual spread (₹/g) and z-score (unitless) are two
  stacked panels sharing the time axis - never a dual-axis chart.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fixed categorical order (validated CVD-safe palette, light-mode steps).
SYMBOL_COLORS: dict[str, str] = {
    "GOLDM": "#d8b678",       # blue
    "GOLDGUINEA": "#8fb2a3",  # orange
    "GOLDPETAL": "#c8a16a",   # aqua
    "GOLDTEN": "#d8d2b6",     # yellow
}
STRATEGY_COLOR = "#d4b06f"    # violet - the strategy is its own entity
BENCHMARK_COLOR = "#898781"   # muted grey - context, not the hero
LONG_COLOR = "#c8a16a"
SHORT_COLOR = "#e34948"
EXIT_COLOR = "#898781"
THRESHOLD_COLOR = "#c3c2b7"
PHASE_COLORS = {"Trading": "#86b6ef", "Tender period": "#8fb2a3"}

_BASE_LAYOUT = dict(
    margin=dict(l=8, r=8, t=30, b=6),
    height=280,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                title_text=""),
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=11, color="#d8d7ca"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111816",
)


def _layout(fig: go.Figure, title: str, **kw) -> go.Figure:
    fig.update_layout(**{**_BASE_LAYOUT, **kw})
    fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=14)))
    fig.update_xaxes(showgrid=False, color="#a8b0a8", linecolor="#303a33", tickfont=dict(color="#a8b0a8"))
    fig.update_yaxes(gridcolor="#26312b", gridwidth=1, zeroline=False, color="#a8b0a8", tickfont=dict(color="#a8b0a8"))
    return fig


# --------------------------------------------------------------------------- #
def normalized_price_chart(prices: pd.DataFrame, symbols: list[str],
                           mode: str = "level") -> go.Figure:
    """Active-contract prices in ₹/g at 999 purity.

    ``mode="level"`` plots raw normalised levels (the tracking view);
    ``mode="premium"`` plots each symbol minus GOLDM, which makes the small
    dislocations visible - at ~₹10k/g the levels overlap almost perfectly.
    """
    fig = go.Figure()
    base = prices.get("GOLDM")
    for sym in symbols:
        if sym not in prices:
            continue
        s = prices[sym].dropna()
        if mode == "premium":
            if sym == "GOLDM" or base is None:
                continue
            s = (s - base).dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=sym, mode="lines",
            line=dict(color=SYMBOL_COLORS.get(sym), width=2),
            hovertemplate=f"{sym}: ₹%{{y:,.1f}}/g<extra></extra>"))
    if mode == "premium":
        fig.add_hline(y=0, line=dict(color=THRESHOLD_COLOR, width=1))
        title, ytitle = "Premium vs GOLDM (₹/g @ 999)", "₹ per gram"
    else:
        title, ytitle = "Normalised price tracking (₹/g @ 999)", "₹ per gram"
    fig = _layout(fig, title)
    fig.update_yaxes(title_text=ytitle, tickprefix="₹", tickformat=",.0f")
    return fig


def residual_zscore_chart(frame: pd.DataFrame, trades: pd.DataFrame,
                          entry_z: float, exit_z: float) -> go.Figure:
    """Residual spread (top) and rolling z-score with trade markers (bottom)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.42, 0.58],
                        subplot_titles=("Residual spread (₹/g, term structure removed)",
                                        "Rolling z-score"))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["residual_spread"], name="Residual spread",
        mode="lines", line=dict(color=STRATEGY_COLOR, width=1.5), showlegend=False,
        hovertemplate="Residual: ₹%{y:,.2f}/g<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=THRESHOLD_COLOR, width=1), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["zscore"], name="z-score", mode="lines",
        line=dict(color="#52514e", width=1.5), showlegend=False,
        hovertemplate="z: %{y:.2f}<extra></extra>"), row=2, col=1)
    for lvl, dash in ((entry_z, "dash"), (-entry_z, "dash"), (exit_z, "dot"), (-exit_z, "dot")):
        fig.add_hline(y=lvl, line=dict(color=THRESHOLD_COLOR, width=1, dash=dash), row=2, col=1)

    if trades is not None and len(trades):
        z = frame["zscore"]
        for direction, name, symbol, color in (
                ("LONG_SPREAD", "Buy spread (long A / short B)", "triangle-up", LONG_COLOR),
                ("SHORT_SPREAD", "Sell spread (short A / long B)", "triangle-down", SHORT_COLOR)):
            t = trades[trades["direction"] == direction]
            if t.empty:
                continue
            fig.add_trace(go.Scatter(
                x=t["entry_date"], y=z.reindex(t["entry_date"]).values, name=name,
                mode="markers", marker=dict(symbol=symbol, size=11, color=color,
                                            line=dict(width=1.5, color="white")),
                customdata=t[["entry_z", "net_pnl"]].values,
                hovertemplate=(f"{name}<br>signal z %{{customdata[0]:.2f}}"
                               "<br>trade net ₹%{customdata[1]:,.0f}<extra></extra>")),
                row=2, col=1)
        fig.add_trace(go.Scatter(
            x=trades["exit_date"], y=z.reindex(trades["exit_date"]).values, name="Exit",
            mode="markers", marker=dict(symbol="circle-open", size=9, color=EXIT_COLOR,
                                        line=dict(width=2)),
            customdata=trades[["exit_reason"]].values,
            hovertemplate="Exit: %{customdata[0]}<extra></extra>"), row=2, col=1)

    fig = _layout(fig, "", height=420, hovermode="x")
    fig.update_yaxes(title_text="₹/g", row=1, col=1)
    fig.update_yaxes(title_text="z", row=2, col=1)
    fig.update_annotations(font_size=13, x=0, xanchor="left")
    return fig


def equity_curve_chart(daily: pd.DataFrame, bench: pd.DataFrame,
                       show_benchmark: bool = True) -> go.Figure:
    """Cumulative net ₹ P&L of the strategy vs. buy & hold gold (same grams)."""
    fig = go.Figure()
    if show_benchmark:
        b = bench["bh_equity"].reindex(daily.index).ffill()
        fig.add_trace(go.Scatter(
            x=b.index, y=b.values, name="Buy & hold gold (GOLDM, same grams)",
            mode="lines", line=dict(color=BENCHMARK_COLOR, width=2),
            hovertemplate="Buy & hold: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["equity"], name="Strategy (net of costs)",
        mode="lines", line=dict(color=STRATEGY_COLOR, width=2.5),
        hovertemplate="Strategy: ₹%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=THRESHOLD_COLOR, width=1))
    fig = _layout(fig, "Equity curve: strategy vs. raw gold buy & hold")
    fig.update_yaxes(title_text="Cumulative P&L (₹)", tickprefix="₹", tickformat=",.0f")
    return fig


def lifecycle_chart(segments: pd.DataFrame, as_of: pd.Timestamp) -> go.Figure:
    """Gantt of trading and tender windows per contract, with an as-of line."""
    fig = go.Figure()
    order = list(dict.fromkeys(segments.sort_values("end")["contract"]))
    for phase, color in PHASE_COLORS.items():
        s = segments[segments["phase"] == phase]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            base=s["start"], x=(s["end"] - s["start"]).dt.total_seconds() * 1000,
            y=s["contract"], orientation="h", name=phase,
            marker=dict(color=color, line=dict(width=0)),
            customdata=s[["start", "end", "status"]].astype(str).values,
            hovertemplate=("%{y}<br>" + phase + ": %{customdata[0]} → %{customdata[1]}"
                           "<br>Status: %{customdata[2]}<extra></extra>")))
    fig.add_vline(x=as_of, line=dict(color="#52514e", width=1.5, dash="dash"))
    fig.add_annotation(x=as_of, y=1.02, yref="paper", text="as of", showarrow=False,
                       font=dict(size=11, color="#52514e"))
    fig = _layout(fig, "Contract lifecycle: trading → tender → expiry",
                  barmode="overlay", hovermode="closest",
                  height=max(300, 22 * len(order) + 90))
    fig.update_xaxes(type="date")
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], title_text="")
    return fig
