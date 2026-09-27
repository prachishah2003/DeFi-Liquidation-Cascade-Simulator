"""Regression tests for the truncated-book artifacts.

These cover the two bugs behind the non-monotonic bad-debt result -- bad debt
came out at $373.28M for a -30% shock and $32.34M for a -45% one, and neither
figure was real. Both bugs live in what the swap engine counts as tradeable
book, and both produce the same symptom: a large, discontinuous price move
bought with a negligible amount of flow.

The synthetic book below reproduces the live weETH/WETH 5bp shape, which is
what the model actually hit: a couple of genuinely deep ranges followed by a
long tail of ranges holding effectively nothing but spanning a wide price band.
Crossing an empty range costs no input, so without a cut the price free-falls
across the tail and comes to rest somewhere with no economic meaning.
"""

import pytest

from uniswap_v3_math import (TickSegment, swap_multi_tick, max_input_available,
                             usable_segments)


def truncated_book():
    """Two real ranges, then a long near-empty tail. Mirrors the live shape.

    Measured on the weETH/WETH 5bp book: two ranges at L=4,790,207, then ~85
    ranges between L=0.03 and L=0.66, together spanning about 24% of price.
    """
    segments = []
    # deep, genuinely populated ranges just below the current price
    top = 1.05
    for _ in range(2):
        segments.append(TickSegment(sqrt_p_lower=top - 0.005,
                                    sqrt_p_upper=top,
                                    liquidity=4_790_207.0))
        top -= 0.005
    # the dead tail: wide in price, essentially empty
    for _ in range(85):
        segments.append(TickSegment(sqrt_p_lower=top - 0.005,
                                    sqrt_p_upper=top,
                                    liquidity=0.03))
        top -= 0.005
    return segments, 1.05


def deep_book():
    """A uniformly populated book, as a control -- nothing should be dropped."""
    segments = []
    top = 1.05
    for _ in range(40):
        segments.append(TickSegment(sqrt_p_lower=top - 0.005,
                                    sqrt_p_upper=top, liquidity=1_000_000.0))
        top -= 0.005
    return segments, 1.05


def test_selling_exactly_capacity_does_not_free_fall():
    """The cliff. Capacity and just under it must price the same.

    Before the fix, on the live book, selling 99.9% of `sell_capacity` moved
    the weETH/WETH ratio -0.19% and selling 100% of it moved it -24.41% -- the
    last 0.04 of 4,337 weETH crossing 84 further ticks. Cascade flow saturates
    at capacity, so whether a scenario landed on the cliff or a few tokens
    short of it decided whether every weETH-collateralised account was marked
    down a quarter.
    """
    segments, sqrt_p = truncated_book()
    capacity = max_input_available(sqrt_p, segments, zero_for_one=True,
                                   fee_bps=5.0)
    assert capacity > 0

    nearly = swap_multi_tick(sqrt_p, segments, capacity * 0.999,
                             zero_for_one=True, fee_bps=5.0)
    full = swap_multi_tick(sqrt_p, segments, capacity,
                           zero_for_one=True, fee_bps=5.0)

    near_move = nearly.new_sqrt_p ** 2 / sqrt_p ** 2 - 1
    full_move = full.new_sqrt_p ** 2 / sqrt_p ** 2 - 1

    # the last 0.1% of capacity may not move price by more than another 0.1%
    assert abs(full_move - near_move) < 1e-3, (
        f"discontinuity at capacity: {near_move:+.4%} at 99.9% of capacity "
        f"vs {full_move:+.4%} at 100%")


def test_repeated_capacity_sales_do_not_ratchet():
    """The ratchet. Consuming the book must not redefine what counts as book.

    The first attempt at fixing the cliff measured 'real depth' relative to
    whatever lay ahead of the CURRENT price. Once a sale had consumed the real
    ranges, the dead tail was all that lay ahead, so the tail became the book
    and the next sale promoted the next sliver of it. Three successive
    full-capacity sales on the live leg gave -0.19%, then -5.72%, then -24.41%
    -- and the third filled ZERO tokens while moving the price 19 points.
    """
    segments, sqrt_p = truncated_book()
    current = sqrt_p
    start = sqrt_p

    for sale in range(5):
        capacity = max_input_available(current, segments, zero_for_one=True,
                                       fee_bps=5.0)
        if capacity <= 0:
            break
        result = swap_multi_tick(current, segments, capacity,
                                 zero_for_one=True, fee_bps=5.0)
        # a sale that fills nothing must not move the price at all
        if result.amount_in_filled <= 0:
            assert result.new_sqrt_p == current, (
                f"sale {sale} filled nothing but moved price "
                f"{result.new_sqrt_p ** 2 / current ** 2 - 1:+.2%}")
        current = result.new_sqrt_p

    total_move = current ** 2 / start ** 2 - 1
    # the two real ranges span ~2% of price; the tail another ~22%. Walking
    # into the tail at all is the failure this guards.
    assert total_move > -0.05, (
        f"repeated sales ratcheted the price {total_move:+.2%} -- into the "
        f"tail the truncation was supposed to exclude")


def test_tail_is_dropped_and_flagged():
    """`usable_segments` keeps the real ranges and reports the truncation."""
    segments, sqrt_p = truncated_book()
    kept, truncated = usable_segments(sqrt_p, segments, zero_for_one=True)
    assert truncated is True
    assert len(kept) < len(segments)
    # every dropped range must be one of the near-empty ones
    assert all(seg.liquidity > 1.0 for seg in kept)


def test_uniform_book_is_not_truncated():
    """Control: a book with no dead tail must be kept whole.

    This is the guard against over-correcting. An earlier fix for a DIFFERENT
    bug stopped the walk at the first zero-liquidity range, which made an
    rETH/WETH pool with usable depth and one empty range at index 149 report a
    capacity of zero -- failing every rETH liquidation and manufacturing bad
    debt out of nothing. Truncation must not resurrect that.
    """
    segments, sqrt_p = deep_book()
    kept, truncated = usable_segments(sqrt_p, segments, zero_for_one=True)
    assert truncated is False
    assert len(kept) == len(segments)


def test_interior_gap_is_still_crossed_free():
    """An empty range with real depth beyond it must not end the book."""
    segments, sqrt_p = deep_book()
    # hollow out one range in the middle
    segments[10] = TickSegment(sqrt_p_lower=segments[10].sqrt_p_lower,
                               sqrt_p_upper=segments[10].sqrt_p_upper,
                               liquidity=0.0)
    kept, _ = usable_segments(sqrt_p, segments, zero_for_one=True)
    # depth beyond the gap is still reachable
    assert len(kept) > 11
    capacity = max_input_available(sqrt_p, segments, zero_for_one=True,
                                   fee_bps=5.0)
    beyond = sum(s.liquidity for s in segments[11:])
    assert capacity > 0 and beyond > 0


def test_impact_is_monotone_in_trade_size():
    """More selling may never move the price less. Cheap invariant, real teeth.

    Both bugs above violated this, and it is the property that made bad debt
    non-monotonic in shock size downstream.
    """
    segments, sqrt_p = truncated_book()
    capacity = max_input_available(sqrt_p, segments, zero_for_one=True,
                                   fee_bps=5.0)
    previous = sqrt_p
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.5, 2.0):
        result = swap_multi_tick(sqrt_p, segments, capacity * frac,
                                 zero_for_one=True, fee_bps=5.0)
        assert result.new_sqrt_p <= previous + 1e-12, (
            f"selling {frac:.0%} of capacity moved price LESS than a smaller "
            f"trade did")
        previous = result.new_sqrt_p


def test_overfilled_sale_reports_data_exhaustion():
    """Asking for more than the book holds must say so, not silently fill."""
    segments, sqrt_p = truncated_book()
    capacity = max_input_available(sqrt_p, segments, zero_for_one=True,
                                   fee_bps=5.0)
    result = swap_multi_tick(sqrt_p, segments, capacity * 3,
                             zero_for_one=True, fee_bps=5.0)
    assert result.ran_dry is True
    assert result.amount_in_filled < capacity * 3
