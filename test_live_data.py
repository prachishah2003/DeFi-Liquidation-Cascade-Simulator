"""
Tests the pure data-transformation logic in live_data.py using synthetic
responses shaped exactly like what the Uniswap v3 subgraph returns -- this
does NOT hit the network, so it runs anywhere.

Two things are tested on purpose because both caused real, silent bugs
against the live pool:

1. Token ordering: Uniswap orders token0/token1 by ascending contract
   address, not by which one is "the collateral" you care about. The real
   USDC/WETH 0.3% pool has token0=USDC, token1=WETH.

2. Liquidity magnitude: the subgraph's `liquidity` field is in RAW
   (undecimaled) units -- astronomically large numbers. The first version
   of this code decimal-adjusted price but not liquidity, so a human-scale
   sell (50 WETH) against raw-scale liquidity produced a price move so
   tiny it was indistinguishable from float noise -- technically nonzero,
   so a loose "price went down" assertion passed, but economically zero.
   `test_realistic_liquidity_magnitude` below uses genuinely raw-scale
   liquidity and requires a *meaningful* (>0.01%) move, which is what
   should have caught this the first time.
"""

import math
from live_data import build_pool_from_ticks
from cascade_sim import Pool


def synthetic_pool_response(token0_symbol: str, token0_decimals: int,
                             token1_symbol: str, token1_decimals: int,
                             price_1_per_0: float, human_liquidity: float):
    """
    price_1_per_0  : how many token1 per 1 token0 (raw Uniswap convention)
    human_liquidity: liquidity expressed in human-readable-unit terms --
                      this gets converted to the RAW magnitude the real
                      subgraph would return, so the test exercises the
                      same conversion build_pool_from_ticks has to undo.
    """
    raw_price = price_1_per_0 / (10 ** (token0_decimals - token1_decimals))
    current_tick = int(math.log(raw_price) / math.log(1.0001))

    liquidity_adj = 10 ** ((token0_decimals + token1_decimals) / 2)
    raw_liquidity = human_liquidity * liquidity_adj

    pool = {
        "tick": str(current_tick),
        "sqrtPrice": "0",  # unused; build_pool_from_ticks recomputes from tick
        "liquidity": str(int(raw_liquidity)),
        "token0": {"symbol": token0_symbol, "decimals": str(token0_decimals)},
        "token1": {"symbol": token1_symbol, "decimals": str(token1_decimals)},
    }
    tick_spacing = 60
    ticks = []
    for i in range(1, 11):
        ticks.append({
            "tickIdx": str(current_tick - i * tick_spacing * 20),
            "liquidityNet": str(-int(raw_liquidity * 0.1) // i),
        })
    for i in range(1, 6):
        ticks.append({
            "tickIdx": str(current_tick + i * tick_spacing * 20),
            "liquidityNet": str(int(raw_liquidity * 0.15) // i),
        })
    return pool, ticks


def _check_weth_pool(pool: Pool, expected_price: float, sell_qty: float,
                      min_impact_pct: float):
    price_now = pool.collateral_price("WETH")
    print(f"  reconstructed WETH price: ${price_now:,.2f}  (expected ~{expected_price:,.0f})")
    assert abs(price_now - expected_price) / expected_price < 0.05, "price reconstruction is off"

    result = pool.sell("WETH", sell_qty)
    new_price = pool.collateral_price("WETH")
    impact_pct = (price_now - new_price) / price_now * 100
    print(f"  sell {sell_qty:g} WETH -> ${price_now:,.2f} -> ${new_price:,.2f} "
          f"({impact_pct:.4f}% impact), got {result.amount_out:,.2f} of the other token")
    assert impact_pct > min_impact_pct, (
        f"price impact ({impact_pct:.6f}%) is too small to be real -- "
        f"this is the failure mode of the liquidity-scaling bug"
    )


def test_weth_as_token1():
    """Real-world ordering: token0=USDC, token1=WETH."""
    print("Case 1: token0=USDC, token1=WETH (matches the real mainnet pool)")
    pool_data, ticks_data = synthetic_pool_response(
        "USDC", 6, "WETH", 18, price_1_per_0=1 / 2800.0, human_liquidity=8_000_000)
    pool = build_pool_from_ticks(pool_data, ticks_data)
    assert pool.token0_symbol == "USDC" and pool.token1_symbol == "WETH"
    _check_weth_pool(pool, expected_price=2800.0, sell_qty=50.0, min_impact_pct=0.01)


def test_weth_as_token0():
    """The other possible ordering."""
    print("\nCase 2: token0=WETH, token1=USDC")
    pool_data, ticks_data = synthetic_pool_response(
        "WETH", 18, "USDC", 6, price_1_per_0=2800.0, human_liquidity=8_000_000)
    pool = build_pool_from_ticks(pool_data, ticks_data)
    assert pool.token0_symbol == "WETH" and pool.token1_symbol == "USDC"
    _check_weth_pool(pool, expected_price=2800.0, sell_qty=50.0, min_impact_pct=0.01)


def test_realistic_liquidity_magnitude():
    """
    The regression test for the actual bug that shipped: raw liquidity in
    the tens-of-quintillions range (typical for a real top-tier pool),
    with a realistic multi-position liquidation-sized sell (500 WETH).
    Requires a clearly meaningful price impact -- this is what should have
    failed the first time instead of silently passing.
    """
    print("\nCase 3: realistic raw-magnitude liquidity (the actual bug)")
    pool_data, ticks_data = synthetic_pool_response(
        "USDC", 6, "WETH", 18, price_1_per_0=1 / 2800.0, human_liquidity=50_000_000)
    pool = build_pool_from_ticks(pool_data, ticks_data)
    print(f"  raw liquidity field value: {pool_data['liquidity']} "
          f"(this is the magnitude a real subgraph actually returns)")
    _check_weth_pool(pool, expected_price=2800.0, sell_qty=500.0, min_impact_pct=0.05)


if __name__ == "__main__":
    test_weth_as_token1()
    test_weth_as_token0()
    test_realistic_liquidity_magnitude()
    print("\nAll checks passed.")