"""A liquidator takes the slice that pays, not all-or-nothing.

Liquidation used to be a single yes/no on the full close-factor repayment: if
the whole lot did not clear its costs, the liquidator walked away and the
entire position stayed open. Slippage grows faster than linearly in size, so
when the full amount is unprofitable a smaller slice of the same position
frequently is not. Refusing the whole position overstates bad debt, and
overstates it worst exactly where this model is most interesting — thin books
under stress.

These tests drive `largest_profitable_fraction` with synthetic proceeds
curves, so the economics are isolated from any particular book.
"""

import pytest

from liquidator import LiquidatorEconomics, largest_profitable_fraction


def econ(min_profit=25.0, gas_gwei=0.0, flash=0.0):
    return LiquidatorEconomics(gas_price_gwei=gas_gwei, flash_loan_fee=flash,
                               min_profit_usd=min_profit)


def test_full_liquidation_is_taken_when_it_pays():
    """No change in behaviour where the old code already worked."""
    repay = 10_000.0
    # 5% bonus, negligible slippage
    proceeds = lambda f: repay * f * 1.05
    assert largest_profitable_fraction(proceeds, repay, econ(), 2500.0) == 1.0


def test_nothing_is_taken_when_no_size_pays():
    """A position underwater at every size must still be refused outright."""
    repay = 10_000.0
    # collateral fetches less than the debt at any size
    proceeds = lambda f: repay * f * 0.80
    assert largest_profitable_fraction(proceeds, repay, econ(), 2500.0) == 0.0


def test_partial_slice_is_taken_when_the_whole_lot_is_not():
    """The bug in one test: full unprofitable, half profitable.

    Proceeds carry a 5% bonus less a slippage term quadratic in size. At f=1
    slippage swamps the bonus; at small f it does not.
    """
    repay = 10_000.0

    def proceeds(f):
        gross = repay * f * 1.05
        slippage = repay * 0.20 * f * f      # quadratic in trade size
        return gross - slippage

    # the old behaviour: full size loses money
    assert proceeds(1.0) - repay < 0

    fraction = largest_profitable_fraction(proceeds, repay, econ(), 2500.0)
    assert 0.0 < fraction < 1.0
    # and the slice actually taken is profitable
    assert proceeds(fraction) - repay * fraction >= 25.0


def test_the_largest_viable_slice_is_taken_not_the_most_profitable():
    """Competition drives the size to the edge of viability, not its peak.

    Profit peaks near f=0.26 for this curve but stays viable well past it;
    the model should take the larger slice, because whatever margin the first
    liquidator leaves the next one bids for.
    """
    repay = 10_000.0

    def proceeds(f):
        return repay * f * 1.05 - repay * 0.20 * f * f

    def profit(f):
        return proceeds(f) - repay * f

    fraction = largest_profitable_fraction(proceeds, repay, econ(), 2500.0)
    peak = max((profit(i / 100.0), i / 100.0) for i in range(1, 101))[1]
    assert fraction > peak, (
        f"took f={fraction:.3f}, no larger than the profit-maximising "
        f"f={peak:.3f} — that is a monopolist, not a competitive market")


def test_gas_makes_tiny_slices_unviable():
    """The viable region is an interval, not a tail — gas is fixed per call.

    This is why the search scans rather than bisecting from zero: profit is
    not monotone in size.
    """
    repay = 1_000_000.0

    def proceeds(f):
        return repay * f * 1.05 - repay * 0.20 * f * f

    def profit(f):
        return proceeds(f) - repay * f

    e = econ(min_profit=25.0, gas_gwei=200.0)   # ~$225 of gas at 2500/ETH
    gas = e.gas_cost_usd(2500.0)
    assert gas > 100.0

    # A 0.1% slice earns less than the gas it burns, so the viable region has
    # a LOWER edge as well as an upper one -- it is an interval. That is why
    # the search scans for a viable point instead of bisecting up from zero,
    # which assumes monotonicity that does not hold here.
    assert profit(0.001) - gas < e.min_profit_usd
    assert profit(0.20) - gas > e.min_profit_usd

    fraction = largest_profitable_fraction(proceeds, repay, e, 2500.0)
    assert fraction > 0.20


def test_zero_repayment_is_refused():
    assert largest_profitable_fraction(lambda f: 1e9, 0.0, econ(), 2500.0) == 0.0


def test_result_is_a_valid_fraction():
    """Whatever the curve, the answer must be usable as a multiplier."""
    repay = 10_000.0
    for slope in (0.5, 0.9, 1.0, 1.05, 1.5):
        for curve in (0.0, 0.05, 0.2, 0.6):
            f = largest_profitable_fraction(
                lambda x, s=slope, c=curve: repay * x * s - repay * c * x * x,
                repay, econ(), 2500.0)
            assert 0.0 <= f <= 1.0
