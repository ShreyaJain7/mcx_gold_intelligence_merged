"""
Alpha / beta separation (Role B).

Regresses daily strategy returns on daily spot-gold returns::

    strategy_return_t = alpha + beta * gold_return_t + eps_t

A genuine relative-value (spread) strategy should show beta ~ 0: its P&L
should come from the *dislocation* closing, not from gold going up. A
significant beta means the "hedge" leaks directional exposure (e.g. lot
rounding leaving unequal grams per leg, or the 995-vs-999 fine-gold content).

Standard errors are Newey-West (HAC). Daily P&L of a strategy that holds
positions for several days is autocorrelated; plain OLS errors would overstate
the significance of alpha.

``statsmodels`` is the production path. If it cannot be imported, a small
numpy implementation of the same OLS + Newey-West estimator runs instead so
the pipeline never hard-fails; the result records which estimator was used.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    sm = None
    _HAS_STATSMODELS = False
    logger.warning("statsmodels not installed - using numpy OLS/Newey-West fallback")


@dataclass(frozen=True)
class AttributionResult:
    """Output of ``compute_attribution`` (all returns are simple, not log)."""

    alpha_daily: float
    alpha_annualized: float
    beta: float
    r_squared: float
    alpha_tstat: float
    beta_tstat: float
    alpha_pvalue: float
    beta_pvalue: float
    sharpe_ratio: float          # annualised, strategy returns (rf = 0: spread book is margin-funded)
    annualized_return: float
    annualized_vol: float
    information_ratio: float     # alpha / residual vol, annualised
    n_obs: int
    estimator: str               # "statsmodels OLS (HAC)" or "numpy OLS (HAC) fallback"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (mean / std * sqrt(periods)), rf = 0.

    Includes flat days: the ratio describes the strategy as actually run, not
    just its time in the market.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")


def _numpy_ols_hac(y: np.ndarray, x: np.ndarray, lags: int) -> dict:
    """OLS with Newey-West (Bartlett kernel) covariance - statsmodels fallback."""
    from scipy import stats  # scipy ships with every numpy stack we target

    X = np.column_stack([np.ones_like(x), x])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = b / se
    p = 2 * stats.norm.sf(np.abs(t))  # large-sample normal, as statsmodels HAC
    ss_res = float(u @ u)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"params": b, "tvalues": t, "pvalues": p, "resid": u,
            "rsquared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def compute_attribution(strategy_returns: pd.Series, gold_returns: pd.Series,
                        periods_per_year: int = 252, hac_lags: int = 5,
                        min_obs: int = 30) -> AttributionResult:
    """Run ``strategy_return ~ const + gold_return`` and report alpha/beta.

    Parameters
    ----------
    strategy_returns:
        Daily net strategy returns (₹ P&L / capital base), indexed by date.
    gold_returns:
        Daily spot-gold proxy returns, indexed by date.
    periods_per_year:
        Annualisation factor (MCX trades ~250-255 days a year).
    hac_lags:
        Newey-West lag truncation; should cover the typical holding period.
    min_obs:
        Below this many overlapping days, return NaNs with an explanatory note
        rather than a meaningless regression.
    """
    data = pd.concat({"y": strategy_returns, "x": gold_returns}, axis=1, join="inner").dropna()
    n = len(data)
    sharpe = sharpe_ratio(data["y"], periods_per_year) if n else float("nan")
    ann_ret = float(data["y"].mean() * periods_per_year) if n else float("nan")
    ann_vol = float(data["y"].std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else float("nan")

    nan = float("nan")
    if n < min_obs or data["x"].std() == 0 or data["y"].std() == 0:
        return AttributionResult(nan, nan, nan, nan, nan, nan, nan, nan, sharpe, ann_ret,
                                 ann_vol, nan, n, "n/a",
                                 note=f"Insufficient / degenerate data ({n} overlapping days)")

    y, x = data["y"].to_numpy(float), data["x"].to_numpy(float)
    if _HAS_STATSMODELS:
        X = sm.add_constant(x, has_constant="add")
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        params, tvals, pvals = fit.params, fit.tvalues, fit.pvalues
        resid, r2 = fit.resid, float(fit.rsquared)
        estimator = f"statsmodels OLS, Newey-West HAC ({hac_lags} lags)"
    else:
        fit = _numpy_ols_hac(y, x, hac_lags)
        params, tvals, pvals = fit["params"], fit["tvalues"], fit["pvalues"]
        resid, r2 = fit["resid"], fit["rsquared"]
        estimator = f"numpy OLS, Newey-West HAC ({hac_lags} lags) - statsmodels fallback"

    resid_sd = float(np.std(resid, ddof=2))
    info_ratio = (float(params[0]) / resid_sd * math.sqrt(periods_per_year)
                  if resid_sd > 0 else nan)
    return AttributionResult(
        alpha_daily=float(params[0]),
        alpha_annualized=float(params[0]) * periods_per_year,
        beta=float(params[1]),
        r_squared=r2,
        alpha_tstat=float(tvals[0]), beta_tstat=float(tvals[1]),
        alpha_pvalue=float(pvals[0]), beta_pvalue=float(pvals[1]),
        sharpe_ratio=sharpe, annualized_return=ann_ret, annualized_vol=ann_vol,
        information_ratio=info_ratio, n_obs=n, estimator=estimator,
    )


def attribute_strategy(result) -> AttributionResult:
    """Convenience wrapper for a ``quant_engine.StrategyResult``."""
    cfg = result.config
    # Evaluate from the first day a signal *could* exist: the warm-up period is
    # flat by construction and would dilute Sharpe / alpha if included.
    live = result.frame["zscore"].first_valid_index()
    if live is None:
        return compute_attribution(pd.Series(dtype=float), pd.Series(dtype=float))
    return compute_attribution(result.daily.loc[live:, "strategy_return"],
                               result.benchmark["gold_return"],
                               periods_per_year=cfg.periods_per_year,
                               hac_lags=cfg.hac_lags)
