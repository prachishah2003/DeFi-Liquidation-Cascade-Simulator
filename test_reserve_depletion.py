"""A pool cannot pay out the same tokens twice.

`Pool.sell` used to move only the price. The reserve cap
(`reserve_limited_capacity`) was therefore re-derived from the pool's
ORIGINAL balance on every call, so a venue could be sold into indefinitely:
the live wstETH/WETH leg reported a capacity of 1,591 wstETH, then 1,592,
then 1,594, for as many rounds as the cascade cared to run — each sale paying
out WETH the pool had already spent.

That is the "static book" limitation. Note what is and is not static here:
Uniswap's liquidity L genuinely does not change as price moves through a
range, so leaving the tick book alone is correct. What was wrong was treating
the payout token as inexhaustible.
"""

import pytest

from cascade_sim import Pool
from uniswap_v3_math import TickSegment


def make_pool(held_token1=100.0):
    """WETH/USDC-shaped pool with a deep book but a small USDC balance.

    The book is deliberately deeper than the reserves, so the reserve cap is
    the binding constraint and the test is about reserves rather than ticks.
    """
    segments = []
    top = 1.05
    for _ in range(60):
        segments.append(TickSegment(sqrt_p_lower=top - 0.005,
                                    sqrt_p_upper=top, liquidity=5_000_000.0))
        top -= 0.005
    return Pool(token0_symbol="WETH", token1_symbol="USDC", sqrt_p=1.05,
                segments=segments, fee_bps=30.0,
                tvl_token0=10.0, tvl_token1=held_token1)


def test_reserves_fall_when_collateral_is_sold():
    pool = make_pool()
    before0, before1 = pool.tvl_token0, pool.tvl_token1
    result = pool.sell("WETH", pool.sell_capacity("WETH") * 0.5)

    assert result.amount_in_filled > 0
    # the pool receives what it was sold and pays out the other token
    assert pool.tvl_token0 == pytest.approx(before0 + result.amount_in_filled)
    assert pool.tvl_token1 == pytest.approx(before1 - result.amount_out)
    assert pool.tvl_token1 < before1


def test_capacity_shrinks_after_a_sale():
    """The bug in one assertion: selling must reduce what can be sold next."""
    pool = make_pool()
    first = pool.sell_capacity("WETH")
    pool.sell("WETH", first)
    second = pool.sell_capacity("WETH")

    assert first > 0
    assert second < first, (
        f"capacity did not fall after a full-capacity sale: {first:,.2f} "
        f"then {second:,.2f} — the pool is paying out reserves it has spent")


def test_repeated_sales_exhaust_the_pool_rather_than_running_forever():
    """Total volume must be bounded by what the pool can actually pay out."""
    pool = make_pool(held_token1=100.0)
    total_in, total_out = 0.0, 0.0

    for _ in range(25):
        capacity = pool.sell_capacity("WETH")
        if capacity <= 1e-9:
            break
        result = pool.sell("WETH", capacity)
        if result.amount_in_filled <= 1e-12:
            break
        total_in += result.amount_in_filled
        total_out += result.amount_out
    else:
        pytest.fail("pool still had capacity after 25 full-capacity sales — "
                    "reserves are not depleting")

    # cannot have paid out more of token1 than it ever held
    assert total_out <= 100.0 + 1e-6, (
        f"paid out {total_out:,.4f} token1 from a pool holding 100.0")
    assert pool.tvl_token1 == pytest.approx(100.0 - total_out, abs=1e-6)
    assert total_in > 0


def test_quoting_does_not_move_reserves():
    """Only `sell` settles. A quote is a liquidator thinking, not trading."""
    pool = make_pool()
    before = (pool.tvl_token0, pool.tvl_token1, pool.sqrt_p)
    pool.quote_sell("WETH", pool.sell_capacity("WETH") * 0.5)
    assert (pool.tvl_token0, pool.tvl_token1, pool.sqrt_p) == before


def test_unknown_reserves_stay_unknown():
    """A pool whose holdings were never fetched must not invent them.

    `reserve_limited_capacity` returns inf in that case deliberately — absence
    of the constraint is not the constraint being satisfied — and selling must
    not turn that None into a number.
    """
    pool = make_pool()
    pool.tvl_token0 = None
    pool.tvl_token1 = None
    pool.sell("WETH", 1.0)
    assert pool.tvl_token0 is None
    assert pool.tvl_token1 is None
