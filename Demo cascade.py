"""
Demo: a -12% WETH shock triggers liquidations across Aave and Compound.
Liquidators dump seized WETH into a WETH/USDC pool, which pushes price down
further and triggers a second wave -- the endogenous cascade in action.

Swap in real numbers later:
  - position data: query Aave/Compound subgraphs for open positions
  - pool liquidity: query the Uniswap v3 subgraph's `ticks` for WETH/USDC,
    convert each tick's liquidityNet into TickSegment objects
"""

from uniswap_v3_math import TickSegment, sqrt_p_from_price
from cascade_sim import Position, Pool, run_cascade, summarize


def build_demo_pool(mid_price: float) -> Pool:
    """A simplified WETH/USDC pool: liquidity thins out the further price
    moves from the current mid-price, like a real concentrated-liquidity pool."""
    sqrt_p = sqrt_p_from_price(mid_price)
    segments = []
    # Build liquidity bands going downward (price falling as WETH is sold)
    band_width_pct = 0.02
    liquidity_at_mid = 8_000_000.0
    p = mid_price
    for i in range(15):
        p_next = p * (1 - band_width_pct)
        # liquidity decays the further from the current price (thinner tails)
        liq = liquidity_at_mid * (0.85 ** i)
        segments.append(TickSegment(sqrt_p_lower=sqrt_p_from_price(p_next),
                                     sqrt_p_upper=sqrt_p_from_price(p),
                                     liquidity=liq))
        p = p_next
    return Pool(token0_symbol="WETH", token1_symbol="USDC", sqrt_p=sqrt_p,
                segments=segments)


def build_demo_positions(mid_price: float) -> list:
    """
    A spread of positions across two protocols, each seeded to a specific
    *starting* health factor (all safely above 1.0) so the shock -- not a
    modeling bug -- is what pushes them underwater.
    """
    # (id, protocol, collateral_qty, threshold, target_starting_HF)
    specs = [
        ("aave-1", "aave", 500, 0.80, 1.08),
        ("aave-2", "aave", 300, 0.82, 1.15),
        ("aave-3", "aave", 150, 0.85, 1.30),
        ("comp-1", "compound", 800, 0.78, 1.05),
        ("comp-2", "compound", 220, 0.80, 1.20),
        ("comp-3", "compound", 90, 0.83, 1.40),
        ("aave-4", "aave", 1200, 0.79, 1.10),
    ]
    positions = []
    for pid, protocol, coll_qty, threshold, target_hf in specs:
        debt_qty = (coll_qty * mid_price * threshold) / target_hf
        positions.append(Position(pid, protocol, "WETH", coll_qty, "USDC",
                                   debt_qty, liquidation_threshold=threshold))
    return positions


if __name__ == "__main__":
    mid_price = 2_800.0  # WETH/USDC
    pools = {"WETH": build_demo_pool(mid_price)}
    positions = build_demo_positions(mid_price)

    print(f"Starting WETH price: ${mid_price:,.2f}")
    print("Positions before shock:")
    for p in positions:
        hf = (p.collateral_qty * mid_price * p.liquidation_threshold) / p.debt_qty
        print(f"  {p.position_id:8s} ({p.protocol:8s}) HF = {hf:.3f}")

    print("\nApplying -12% WETH shock and running cascade...\n")
    logs = run_cascade(positions, pools, initial_shock={"WETH": -0.12}, max_rounds=10)

    print("\n--- Summary ---")
    summary = summarize(logs)
    final_price = pools["WETH"].collateral_price("WETH")
    total_sold = logs[-1].cumulative_collateral_sold["WETH"] if logs else 0.0
    print(f"Rounds run: {len(logs)}")
    print(f"Accounts liquidated: {summary.accounts_liquidated} "
          f"(across {summary.liquidation_events} liquidation events -- one account "
          f"can be hit in several consecutive rounds)")
    print(f"Debt repaid: ${summary.debt_repaid_usd:,.0f}  |  "
          f"collateral seized: ${summary.collateral_seized_usd:,.0f}")
    print(f"WETH price: ${mid_price:,.2f} -> ${final_price:,.2f} "
          f"({(final_price/mid_price - 1)*100:.2f}% total move)")
    print(f"Total WETH sold by liquidators into the pool: {total_sold:,.1f} WETH")