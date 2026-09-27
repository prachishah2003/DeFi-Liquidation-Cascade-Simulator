"""
Liquidator economics: when does a liquidation actually happen?

Every cascade model in this project up to now assumed liquidations are free
and instant -- the moment a position's health factor drops below 1, it gets
liquidated in full. That assumption is doing a lot of hidden work, and it is
wrong in the direction that matters.

A liquidation is a trade somebody chooses to make. The liquidator repays part
of the borrower's debt, seizes collateral worth `1 + liquidation_bonus` times
what they repaid, and sells that collateral to get their money back. They do
this only if it clears a profit:

    proceeds   = what the seized collateral actually fetches when SOLD into
                 the pool -- not its mark-to-market value. On a thin book, a
                 5% bonus is wiped out by 6% of slippage.
    costs      = debt repaid + gas + flash-loan fee (most liquidator bots
                 borrow the repayment rather than holding inventory)
    profit     = proceeds - costs

When profit goes negative the rational liquidator does nothing. The position
stays open and underwater, and if its collateral is worth less than its debt,
the shortfall is the PROTOCOL's loss: bad debt.

This flips the story the rest of the project tells. Cascades do not run until
liquidity runs out; they run until liquidating stops paying. What accumulates
past that point is not more liquidations, it is bad debt -- which is the
number a risk team actually cares about, and the number the model could not
previously produce at all.

Gas figures below are order-of-magnitude defaults, not measured values --
every one of them is a constructor argument, and `LiquidatorEconomics` is
meant to be swept, not trusted.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                     # avoids a circular import at runtime
    from cascade_sim import Pool


# Aave v3's flash-loan premium is 9 bps at the time of writing; liquidator
# bots overwhelmingly fund the repayment this way rather than holding
# inventory, so it is on by default. Set to 0 to model an inventoried
# liquidator (a market maker repaying from its own balance sheet).
DEFAULT_FLASH_LOAN_FEE = 0.0009

# A liquidationCall plus the DEX swap to unwind the seized collateral.
# Aave v3's own call is roughly 250-400k; the swap adds 100-200k depending on
# how many ticks it crosses. 450k is a round mid-estimate.
DEFAULT_GAS_UNITS = 450_000


@dataclass
class LiquidatorEconomics:
    """The liquidator's cost structure. Sweep these rather than believing them."""
    gas_price_gwei: float = 25.0
    gas_units: int = DEFAULT_GAS_UNITS
    flash_loan_fee: float = DEFAULT_FLASH_LOAN_FEE
    min_profit_usd: float = 25.0        # below this, a bot doesn't bid
    enabled: bool = True                # False reproduces the old free-and-instant model

    def gas_cost_usd(self, eth_price_usd: float) -> float:
        return self.gas_units * self.gas_price_gwei * 1e-9 * eth_price_usd


@dataclass
class LiquidationQuote:
    """A liquidation the liquidator has priced but not yet executed."""
    profitable: bool
    proceeds_usd: float          # what the collateral actually fetches when sold
    repay_usd: float
    gas_usd: float
    flash_loan_usd: float
    profit_usd: float
    slippage_pct: float          # realized execution shortfall vs. mark price
    reason: str                  # why it was rejected, or "profitable"

    @property
    def rejected(self) -> bool:
        return not self.profitable


def quote_liquidation(pool: "Optional[Pool]", collateral_asset: str,
                      seize_qty: float, mark_price: float, repay_usd: float,
                      econ: LiquidatorEconomics, eth_price_usd: float
                      ) -> LiquidationQuote:
    """Price a single liquidation without executing it.

    `pool` is quoted, not mutated -- the liquidator simulates the swap before
    deciding, exactly as a real bot does. With no pool for this asset we fall
    back to the mark price, which assumes zero slippage and therefore
    OVERSTATES profitability; that is stated rather than hidden.
    """
    gas_usd = econ.gas_cost_usd(eth_price_usd)
    flash_usd = repay_usd * econ.flash_loan_fee
    mark_value = seize_qty * mark_price

    if pool is not None:
        quote = pool.quote_sell(collateral_asset, seize_qty)
        proceeds = quote.amount_out
        if quote.ran_dry:
            # can't even sell the whole lot -- price what actually fills
            proceeds = quote.amount_out
    else:
        proceeds = mark_value

    slippage_pct = (100.0 * (mark_value - proceeds) / mark_value) if mark_value > 0 else 0.0
    profit = proceeds - repay_usd - gas_usd - flash_usd

    if not econ.enabled:
        return LiquidationQuote(True, proceeds, repay_usd, gas_usd, flash_usd,
                                profit, slippage_pct, "economics disabled")
    if profit < econ.min_profit_usd:
        if proceeds < repay_usd:
            reason = "slippage exceeds the liquidation bonus"
        elif profit < 0:
            reason = "gas and fees exceed the bonus"
        else:
            reason = f"profit ${profit:,.2f} below the ${econ.min_profit_usd:,.0f} floor"
        return LiquidationQuote(False, proceeds, repay_usd, gas_usd, flash_usd,
                                profit, slippage_pct, reason)

    return LiquidationQuote(True, proceeds, repay_usd, gas_usd, flash_usd,
                            profit, slippage_pct, "profitable")


def position_bad_debt(collateral_value_usd: float, debt_value_usd: float) -> float:
    """The protocol's loss on one position: debt that its collateral can no
    longer cover. Zero for any position that is still over-collateralized."""
    return max(0.0, debt_value_usd - collateral_value_usd)


#: How many fractions of a liquidation to price when the full one is refused.
#: Profit is not monotone in size -- gas makes tiny slices unprofitable and
#: slippage makes large ones unprofitable -- so the viable region is an
#: interval and a plain bisection can miss it. A coarse scan finds the
#: interval, then a few bisection steps sharpen its upper edge.
PARTIAL_SCAN_POINTS = 8
PARTIAL_REFINE_STEPS = 6


def largest_profitable_fraction(proceeds_at, repay_usd_full, econ,
                                eth_price_usd: float) -> float:
    """The biggest slice of a liquidation that still pays for itself.

    `proceeds_at(f)` prices the unwind of the collateral seized when repaying
    fraction `f` of the full close-factor amount. Returns the largest `f` in
    (0, 1] whose profit clears `econ.min_profit_usd`, or 0.0 if none does.

    WHY THIS EXISTS. Liquidation was previously all-or-nothing: price the full
    close-factor repayment, and if it did not clear its costs, walk away and
    leave the entire position open. Real liquidators do not behave that way.
    Slippage grows faster than linearly in size, so when the whole lot is
    unprofitable a smaller slice of the same position frequently is not -- the
    bot takes what pays and leaves the rest. Refusing the whole position
    instead overstates bad debt, and overstates it worst exactly where this
    model is most interesting: thin books under stress.

    Choosing the LARGEST viable slice rather than the most profitable one is
    deliberate. A single liquidator maximises their own profit, but they are
    not alone: whatever margin the first one leaves, the next bids for. With
    competition the position is worked down until no profitable slice is left,
    which is the upper edge of the viable interval, not its interior peak.
    That is also the conservative choice for the protocol's loss, which is the
    number this model exists to produce.

    Covered by test_partial_liquidation.py.
    """
    if repay_usd_full <= 0:
        return 0.0

    gas_usd = econ.gas_cost_usd(eth_price_usd)

    def clears(f: float) -> bool:
        repay = repay_usd_full * f
        profit = (proceeds_at(f) - repay - gas_usd
                  - repay * econ.flash_loan_fee)
        return profit >= econ.min_profit_usd

    if clears(1.0):
        return 1.0

    # Coarse scan downward for the largest viable point.
    best = 0.0
    step = 1.0 / PARTIAL_SCAN_POINTS
    for i in range(PARTIAL_SCAN_POINTS - 1, 0, -1):
        f = i * step
        if clears(f):
            best = f
            break
    if best <= 0.0:
        return 0.0

    # Sharpen the upper edge between the viable point and the failing one
    # above it, so the slice taken is as large as competition would drive it.
    lo, hi = best, min(1.0, best + step)
    for _ in range(PARTIAL_REFINE_STEPS):
        mid = (lo + hi) / 2.0
        if clears(mid):
            lo = mid
        else:
            hi = mid
    return lo
