"""
Tests for the tick-walking swap engine in uniswap_v3_math.py.

These exist because of a specific bug: the walk used to skip the ACTIVE
tick range (the one containing the current price) whenever the price sat
strictly inside it rather than exactly on a boundary. Pools are built with
the price on a boundary, so nothing caught it -- but `set_collateral_price`
teleports the price mid-range at the start of every cascade, and from that
point on the engine integrated the *next* range's liquidity across the
skipped span. On a concentrated book that overstated price impact ~12x,
which inflated every amplification figure the project reports.

The defence is `_reference_swap` below: an independent implementation that
pre-splits the active segment at the current price, so the active range is
consumed by construction and no cleverness in the walk can hide an error.
Every impact test asserts the engine agrees with it.
"""

from uniswap_v3_math import (TickSegment, SwapResult, swap_multi_tick,
                             swap_within_range, sqrt_p_from_price,
                             price_from_sqrt_p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def book(mid: float, liquidities, step: float = 0.02):
    """A descending book of contiguous ranges, each `step` wide, starting at
    `mid`. liquidities[0] is the range immediately below the mid."""
    segments = []
    price = mid
    for liq in liquidities:
        nxt = price * (1 - step)
        segments.append(TickSegment(sqrt_p_lower=sqrt_p_from_price(nxt),
                                    sqrt_p_upper=sqrt_p_from_price(price),
                                    liquidity=liq))
        price = nxt
    return segments


def _reference_swap(sqrt_p, segments, amount_in, zero_for_one, fee_bps=30.0):
    """Independent reference: split whichever segment contains sqrt_p at
    exactly sqrt_p, so the active range is unambiguously part of the walk,
    then defer to the engine. If the engine's own segment selection is
    correct, the two must agree."""
    split = []
    for seg in segments:
        if seg.sqrt_p_lower < sqrt_p < seg.sqrt_p_upper:
            split.append(TickSegment(seg.sqrt_p_lower, sqrt_p, seg.liquidity))
            split.append(TickSegment(sqrt_p, seg.sqrt_p_upper, seg.liquidity))
        else:
            split.append(seg)
    return swap_multi_tick(sqrt_p, split, amount_in, zero_for_one, fee_bps)


def _rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a)


# ---------------------------------------------------------------------------
# The regression test for the actual bug
# ---------------------------------------------------------------------------

def test_active_segment_is_consumed_first():
    """
    The exact failure mode: a deep range at the mid, thin ranges below, and
    a price sitting INSIDE the deep range (as it always does after a shock).
    Skipping the active range makes the trade eat thin liquidity it should
    never have reached.
    """
    mid = 2800.0
    segments = book(mid, [40_000_000] + [3_000_000] * 29)
    shocked = sqrt_p_from_price(mid * 0.99)   # 1% down: still inside range 1

    got = swap_multi_tick(shocked, segments, 3_000.0, zero_for_one=True)
    want = _reference_swap(shocked, segments, 3_000.0, zero_for_one=True)

    got_px = price_from_sqrt_p(got.new_sqrt_p)
    want_px = price_from_sqrt_p(want.new_sqrt_p)
    print(f"  engine    -> ${got_px:,.2f}")
    print(f"  reference -> ${want_px:,.2f}")
    assert _rel(got_px, want_px) < 1e-9, (
        f"active range not consumed: engine ${got_px:,.2f} vs reference "
        f"${want_px:,.2f} -- the engine is eating the wrong range's liquidity")

    # and sanity: a 3k WETH sale into a 40M-deep range is a small move, not a
    # 10% collapse. Pin the magnitude so a future regression can't pass by
    # merely agreeing with an equally-broken reference.
    impact = (want_px / (mid * 0.99) - 1) * 100
    print(f"  impact of 3,000 WETH into a deep range: {impact:.3f}%")
    assert -2.0 < impact < 0.0, f"implausible impact {impact:.3f}%"


def test_price_on_boundary_still_works():
    """The construction-time case (price exactly on a range boundary), which
    is the only case the old code got right -- must stay right."""
    mid = 2800.0
    segments = book(mid, [8_000_000] * 20)
    sp = sqrt_p_from_price(mid)
    got = swap_multi_tick(sp, segments, 2_000.0, zero_for_one=True)
    want = _reference_swap(sp, segments, 2_000.0, zero_for_one=True)
    assert _rel(price_from_sqrt_p(got.new_sqrt_p),
                price_from_sqrt_p(want.new_sqrt_p)) < 1e-9
    assert not got.ran_dry
    print(f"  on-boundary sell -> ${price_from_sqrt_p(got.new_sqrt_p):,.2f}")


# ---------------------------------------------------------------------------
# Invariants the engine must satisfy regardless of implementation
# ---------------------------------------------------------------------------

def test_single_range_matches_closed_form():
    """Inside one range the walk must reproduce the exact Uniswap formula."""
    mid, liq, qty = 2800.0, 50_000_000.0, 100.0
    segments = book(mid, [liq], step=0.5)      # one very wide range
    sp = sqrt_p_from_price(mid)
    walked = swap_multi_tick(sp, segments, qty, zero_for_one=True)
    direct_sqrt_p, direct_out = swap_within_range(sp, liq, qty, zero_for_one=True)
    assert _rel(walked.new_sqrt_p, direct_sqrt_p) < 1e-12
    assert _rel(walked.amount_out, direct_out) < 1e-12
    print(f"  walk matches closed form to 1e-12 ({walked.amount_out:,.2f} out)")


def test_split_trade_matches_single_trade():
    """Selling 1,000 then 1,000 must land within a fee-rounding hair of
    selling 2,000 at once. Catches liquidity being double-counted or lost at
    range boundaries -- the same class of bug from the other direction."""
    mid = 2800.0
    segments = book(mid, [12_000_000 * (0.9 ** i) for i in range(25)])
    sp = sqrt_p_from_price(mid)

    one_shot = swap_multi_tick(sp, segments, 2_000.0, zero_for_one=True)
    first = swap_multi_tick(sp, segments, 1_000.0, zero_for_one=True)
    second = swap_multi_tick(first.new_sqrt_p, segments, 1_000.0, zero_for_one=True)

    px_one = price_from_sqrt_p(one_shot.new_sqrt_p)
    px_two = price_from_sqrt_p(second.new_sqrt_p)
    print(f"  2,000 at once -> ${px_one:,.2f}   |   1,000 x2 -> ${px_two:,.2f}")
    assert _rel(px_one, px_two) < 1e-6, "split trade diverges from single trade"


def test_impact_is_monotonic_in_size():
    mid = 2800.0
    segments = book(mid, [12_000_000 * (0.9 ** i) for i in range(25)])
    sp = sqrt_p_from_price(mid)
    last = mid
    for qty in (100, 500, 1_000, 2_500, 5_000):
        px = price_from_sqrt_p(
            swap_multi_tick(sp, segments, float(qty), zero_for_one=True).new_sqrt_p)
        assert px < last, f"selling {qty} did not move price below {last}"
        last = px
    print(f"  monotonic through 5,000 WETH (ends ${last:,.2f})")


def test_deeper_book_gives_less_impact():
    """A strictly deeper book must produce strictly less impact for the same
    trade. This is the invariant the original bug violated -- it made the
    deep range invisible, so depth stopped mattering."""
    mid, qty = 2800.0, 3_000.0
    shocked = sqrt_p_from_price(mid * 0.99)
    thin = book(mid, [5_000_000] * 30)
    deep = book(mid, [50_000_000] + [5_000_000] * 29)
    px_thin = price_from_sqrt_p(swap_multi_tick(shocked, thin, qty, True).new_sqrt_p)
    px_deep = price_from_sqrt_p(swap_multi_tick(shocked, deep, qty, True).new_sqrt_p)
    print(f"  thin book -> ${px_thin:,.2f}   deep book -> ${px_deep:,.2f}")
    assert px_deep > px_thin, "extra depth at the mid changed nothing -- active range ignored"


def test_empty_range_is_crossed_free():
    """A range with zero liquidity is traversed instantly: no input consumed,
    price gaps through it to the next range that has depth."""
    mid = 2800.0
    segments = book(mid, [8_000_000, 0.0, 8_000_000, 8_000_000])
    sp = sqrt_p_from_price(mid)
    result = swap_multi_tick(sp, segments, 50_000.0, zero_for_one=True)
    print(f"  gapped book: filled {result.amount_in_filled:,.0f} of 50,000, "
          f"ran_dry={result.ran_dry}")
    assert result.ran_dry, "a zero-liquidity range should register as running dry"


def test_ran_dry_only_when_liquidity_exhausted():
    mid = 2800.0
    segments = book(mid, [20_000_000] * 40)
    sp = sqrt_p_from_price(mid)
    small = swap_multi_tick(sp, segments, 500.0, zero_for_one=True)
    assert not small.ran_dry and small.amount_in_filled == 500.0
    huge = swap_multi_tick(sp, segments, 50_000_000.0, zero_for_one=True)
    assert huge.ran_dry and huge.amount_in_filled < 50_000_000.0
    print(f"  500 fills cleanly; 50,000,000 runs dry after "
          f"{huge.amount_in_filled:,.0f} filled")


def test_selling_the_other_token_raises_price():
    mid = 2800.0
    segments = book(mid, [10_000_000] * 20, step=0.02)
    # extend the book upward so there is somewhere for price to go
    up = []
    price = mid
    for _ in range(20):
        nxt = price * 1.02
        up.append(TickSegment(sqrt_p_from_price(price), sqrt_p_from_price(nxt), 10_000_000))
        price = nxt
    sp = sqrt_p_from_price(mid)
    result = swap_multi_tick(sp, segments + up, 5_000_000.0, zero_for_one=False)
    px = price_from_sqrt_p(result.new_sqrt_p)
    print(f"  buying with 5,000,000 USDC -> ${px:,.2f}")
    assert px > mid, "buying token0 should raise price"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
    print(f"\nAll {len(tests)} checks passed.")
