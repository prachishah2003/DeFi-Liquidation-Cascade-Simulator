"""
Same live cascade as live_demo_cascade.py, but using the proper
multi-collateral/multi-debt model (multi_asset.py) instead of the v1
single-dominant-asset simplification.

Requires: export GRAPH_API_KEY=...
"""

import os
import sys

from live_data import fetch_aave_positions_multi, fetch_uniswap_pool, WETH_USDC_POOL, require_data_access
from cascade_sim import summarize
from multi_asset import multi_health_factor, run_cascade_multi

SHOCK_PCT = -0.12
MAX_AAVE_USERS = 500


def main():
    require_data_access()

    print("Fetching live WETH/USDC pool liquidity...")
    pool = fetch_uniswap_pool(WETH_USDC_POOL)
    print(f"  current price: ${pool.collateral_price('WETH'):,.2f} "
          f"({len(pool.segments)} liquidity segments loaded)")

    print("\nFetching live Aave v3 positions (full multi-asset, no collapsing)...")
    positions, static_prices = fetch_aave_positions_multi(first=MAX_AAVE_USERS)

    if not positions:
        print("No positions with both collateral and debt in this batch.")
        sys.exit(0)

    pools = {"WETH": pool}
    prices = dict(static_prices)
    prices["WETH"] = pool.collateral_price("WETH")

    n_multi = sum(1 for p in positions if len(p.collateral) > 1 or len(p.debt) > 1)
    print(f"\n{len(positions)} positions total, {n_multi} genuinely multi-asset "
          f"(more than one collateral or debt asset)")

    print(f"\nHealth factors at current prices (worst 15):")
    ranked = sorted(positions, key=lambda p: multi_health_factor(p, prices))[:15]
    underwater_already = 0
    for p in ranked:
        hf = multi_health_factor(p, prices)
        tag = ""
        if len(p.collateral) > 1 or len(p.debt) > 1:
            coll_str = "+".join(p.collateral.keys())
            debt_str = "+".join(p.debt.keys())
            tag = f"  [{coll_str} vs {debt_str}]"
        flag = "  <-- ALREADY UNDERWATER" if hf < 1.0 else ""
        if hf < 1.0:
            underwater_already += 1
        print(f"  {p.position_id:12s} HF = {hf:.3f}{flag}{tag}")
    if underwater_already:
        print(f"\n  ({underwater_already} position(s) already below HF 1.0)")

    print(f"\nApplying {SHOCK_PCT:+.0%} WETH shock (other assets unshocked) "
          f"and running cascade...\n")
    logs = run_cascade_multi(positions, pools, static_prices,
                              initial_shock={"WETH": SHOCK_PCT}, max_rounds=15)

    final_price = pool.collateral_price("WETH")
    price_now = prices["WETH"]
    summary = summarize(logs)
    multi_liquidated = sum(1 for p in positions
                            if p.liquidated_rounds and (len(p.collateral) > 1 or len(p.debt) > 1))

    print("\n--- Summary ---")
    print(f"Rounds run: {len(logs)}")
    print(f"Accounts liquidated: {summary.accounts_liquidated} "
          f"({multi_liquidated} of those were genuinely multi-asset), across "
          f"{summary.liquidation_events} liquidation events")
    print(f"Debt repaid: ${summary.debt_repaid_usd:,.0f}  |  "
          f"collateral seized: ${summary.collateral_seized_usd:,.0f}")
    print(f"WETH price: ${price_now:,.2f} -> ${final_price:,.2f} "
          f"({(final_price/price_now - 1)*100:.2f}% total move, "
          f"vs {SHOCK_PCT:.0%} shock applied)")


if __name__ == "__main__":
    main()