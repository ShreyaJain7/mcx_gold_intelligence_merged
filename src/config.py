"""
Central configuration for the MCX Gold Intelligence prototype.

Everything tunable lives here so the quant engine, the batch pipeline and the
Streamlit dashboard all read *one* source of truth. The dataclasses are frozen:
to change a parameter, build a new object with ``dataclasses.replace`` (the
dashboard does this from its sidebar), which keeps runs reproducible and
cache-friendly.

IMPORTANT - indicative values
-----------------------------
Contract specifications, tick sizes and statutory charges below are
*indicative* defaults for a hackathon prototype. MCX and SEBI revise these via
circulars; confirm every number against the current MCX contract
specification pages and your broker's charge sheet before relying on P&L
figures. Purity figures follow the problem statement as given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# --------------------------------------------------------------------------- #
# Contract specifications                                                     #
# --------------------------------------------------------------------------- #

#: Purity every price is normalised to (parts per thousand fine gold).
TARGET_PURITY: float = 999.0


@dataclass(frozen=True)
class ContractSpec:
    """Static description of one MCX gold futures product.

    Attributes
    ----------
    symbol:
        MCX symbol as it appears in the bhavcopy.
    quote_grams:
        Grams of metal the quoted price refers to (price is "₹ per quote_grams").
    quote_purity:
        Purity (per mille) the quoted price refers to.
    lot_grams:
        Grams of metal in one traded lot.
    tick_size:
        Minimum price move, in *quoted* units (₹ per quote_grams).
    alt_quote_grams:
        Alternative quotation bases seen in dirty data (e.g. a vendor that
        publishes GOLDGUINEA per gram instead of per 8 g). The normaliser
        tests these against a same-day cross-symbol reference and picks the
        one that is economically consistent.
    """

    symbol: str
    quote_grams: float
    quote_purity: float
    lot_grams: float
    tick_size: float
    alt_quote_grams: tuple[float, ...] = ()

    @property
    def multiplier(self) -> float:
        """₹ P&L per lot for a 1-unit move in the *quoted* price."""
        return self.lot_grams / self.quote_grams

    @property
    def to_per_gram_999(self) -> float:
        """Factor that converts a quoted price to ₹/gram at 999 purity."""
        return (1.0 / self.quote_grams) * (TARGET_PURITY / self.quote_purity)


CONTRACT_SPECS: Mapping[str, ContractSpec] = {
    # ₹ per 10 g at 995 purity, 100 g lot  -> (close / 10) * (999 / 995)
    "GOLDM": ContractSpec("GOLDM", quote_grams=10.0, quote_purity=995.0,
                          lot_grams=100.0, tick_size=1.0),
    # ₹ per 8 g at 999 purity, 8 g lot     -> close / 8  (per-gram edge case handled)
    "GOLDGUINEA": ContractSpec("GOLDGUINEA", quote_grams=8.0, quote_purity=999.0,
                               lot_grams=8.0, tick_size=1.0, alt_quote_grams=(1.0,)),
    # ₹ per 1 g at 999 purity, 1 g lot     -> close / 1
    "GOLDPETAL": ContractSpec("GOLDPETAL", quote_grams=1.0, quote_purity=999.0,
                              lot_grams=1.0, tick_size=1.0),
    # ₹ per 10 g at 999 purity, 10 g lot   -> close / 10
    "GOLDTEN": ContractSpec("GOLDTEN", quote_grams=10.0, quote_purity=999.0,
                            lot_grams=10.0, tick_size=1.0),
}

SUPPORTED_SYMBOLS: tuple[str, ...] = tuple(CONTRACT_SPECS)

#: Pairs the batch pipeline evaluates by default (leg A, leg B).
#: spread = norm_price_A - norm_price_B; "long spread" = buy A / sell B.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("GOLDM", "GOLDPETAL"),
    ("GOLDM", "GOLDGUINEA"),
    ("GOLDGUINEA", "GOLDPETAL"),
    ("GOLDTEN", "GOLDM"),
)


# --------------------------------------------------------------------------- #
# Transaction cost model                                                      #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CostModel:
    """Per-order MCX futures cost model (all rates are fractions of notional).

    Components per order, per leg:
      * flat brokerage per executed order (discount-broker style);
      * exchange transaction charge on turnover;
      * SEBI turnover fee;
      * GST on (brokerage + exchange charge + SEBI fee);
      * Commodity Transaction Tax (CTT) on the *sell* side of non-agri futures;
      * stamp duty on the *buy* side;
      * slippage proxy = N ticks x tick size x multiplier x lots,
        scaled up when the contract fails the liquidity screen.
    """

    brokerage_per_order: float = 20.0          # ₹ flat per order
    exchange_txn_rate: float = 0.000021        # ~₹2.10 per lakh turnover (indicative)
    sebi_fee_rate: float = 0.000001            # ₹10 per crore
    gst_rate: float = 0.18                     # on brokerage + exchange + SEBI
    ctt_sell_rate: float = 0.0001              # 0.01% CTT, sell side, non-agri futures
    stamp_duty_buy_rate: float = 0.00002       # 0.002% stamp duty, buy side
    slippage_ticks: Mapping[str, float] = field(default_factory=lambda: {
        "GOLDM": 1.0, "GOLDGUINEA": 2.0, "GOLDPETAL": 2.0, "GOLDTEN": 2.0,
    })
    illiquid_slippage_multiplier: float = 3.0  # forced trades on thin days cost more

    def order_cost(self, symbol: str, quoted_price: float, lots: int,
                   side: str, illiquid: bool = False) -> float:
        """All-in ₹ cost of one order.

        Parameters
        ----------
        symbol: MCX symbol of the leg.
        quoted_price: execution price in the contract's quoted units.
        lots: absolute number of lots (must be > 0).
        side: ``"buy"`` or ``"sell"``.
        illiquid: apply the illiquid-day slippage multiplier.
        """
        if lots <= 0:
            return 0.0
        spec = CONTRACT_SPECS[symbol]
        notional = quoted_price * spec.multiplier * lots
        exchange = notional * self.exchange_txn_rate
        sebi = notional * self.sebi_fee_rate
        gst = (self.brokerage_per_order + exchange + sebi) * self.gst_rate
        ctt = notional * self.ctt_sell_rate if side == "sell" else 0.0
        stamp = notional * self.stamp_duty_buy_rate if side == "buy" else 0.0
        ticks = self.slippage_ticks.get(symbol, 1.0)
        if illiquid:
            ticks *= self.illiquid_slippage_multiplier
        slippage = ticks * spec.tick_size * spec.multiplier * lots
        return self.brokerage_per_order + exchange + sebi + gst + ctt + stamp + slippage


# --------------------------------------------------------------------------- #
# Strategy / signal configuration                                             #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StrategyConfig:
    """All knobs of the relative-value strategy.

    Look-ahead discipline (enforced in ``quant_engine``):
      * the term-structure regression at date t is fitted on rows < t only;
      * the z-score at t uses mean/std of residuals < t only;
      * z-driven orders decided at the close of t execute at the close of
        t + ``execution_lag`` (default 1 day);
      * calendar-driven exits (DTE, contract roll, max holding period) are
        known in advance and therefore execute at the close of the same day.
    """

    # --- contract selection ----------------------------------------------
    roll_dte: int = 2                 # a contract stops being "active" once DTE <= this
    # --- term-structure removal (walk-forward regression) ----------------
    detrend_window: int = 120         # trading days in each rolling fit
    detrend_min_obs: int = 60         # minimum rows before the first fit
    detrend_features: tuple[str, ...] = ("dte", "carry", "level")
    detrend_ridge: float = 1e-3       # tiny ridge penalty for numerical stability
    # --- z-score ---------------------------------------------------------
    z_window: int = 60                # rolling window for mean / std of residual
    z_min_periods: int = 40
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = 4.0        # hard stop if the dislocation keeps widening
    # --- risk / lifecycle ------------------------------------------------
    exit_dte: int = 2                 # force flat when either held leg has DTE <= this
    entry_dte_buffer: int = 5         # no new entries unless DTE > exit_dte + buffer
    max_holding_days: int = 15        # trading days
    execution_lag: int = 1            # days between signal close and execution close
    # --- sizing ----------------------------------------------------------
    target_grams: float = 200.0       # gram exposure per leg (200 g = LCM of 100 g and 8 g lots)
    capital_inr: float = 1_000_000.0  # capital base used to turn ₹ P&L into returns
    # --- liquidity screen -----------------------------------------------
    min_volume: Mapping[str, int] = field(default_factory=lambda: {
        "GOLDM": 500, "GOLDGUINEA": 200, "GOLDPETAL": 1000, "GOLDTEN": 100,
    })
    min_open_interest: Mapping[str, int] = field(default_factory=lambda: {
        "GOLDM": 1000, "GOLDGUINEA": 300, "GOLDPETAL": 2000, "GOLDTEN": 200,
    })
    max_participation: float = 0.10   # our lots must be <= 10% of that day's volume
    # --- benchmark -------------------------------------------------------
    benchmark_symbol: str = "GOLDM"   # front-month, roll-adjusted spot-gold proxy
    periods_per_year: int = 252
    hac_lags: int = 5                 # Newey-West lags for the alpha/beta regression
    # --- costs -----------------------------------------------------------
    costs: CostModel = field(default_factory=CostModel)


@dataclass(frozen=True)
class CalendarConfig:
    """Contract lifecycle calendar settings."""

    tender_business_days: int = 5     # tender period = last N business days to expiry
    recently_expired_days: int = 45   # keep just-expired contracts visible for context
