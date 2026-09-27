"""
Same cascade demo as demo_cascade.py, but pulling real data instead of
synthetic positions/pools.

Requires: export GRAPH_API_KEY=... (free key from thegraph.com/studio)

NOTE: I have not been able to run this against the live network myself --
this sandbox's egress allowlist doesn't include gateway.thegraph.com. Every
piece downstream of the fetch (health factors, the cascade loop, the swap
math) is already validated by demo_cascade.py and test_live_data.py; the
only genuinely untested part end-to-end is the live HTTP round trip itself.
If a query errors out, paste me the error -- subgraph field names do drift
between schema versions and it's usually a one-line fix.
"""

import os
import sys

from live_data import fetch_aave_positions, fetch_uniswap_pool, WETH_USDC_POOL
from cascade_sim import run_cascade, summarize

SHOCK_PCT = -0.12   # change this to test other shock sizes
MAX_AAVE_USERS = 500  # how many borrower accounts to pull; raise for a fuller picture


def main():
    if not os.environ.get("GRAPH_API_KEY"):
        print("Missing GRAPH_API_KEY. Get a free key at thegraph.com/studio, "
              "then: export GRAPH_API_KEY=your_key_here")
        sys.exit(1)

    print("Fetching live WETH/USDC pool liquidity...")
    try:
        pool = fetch_uniswap_pool(WETH_USDC_POOL)
    except Exception as e:
        print(f"Pool fetch failed: {e}")
        print("Check the pool address and your API key/subgraph access, then retry.")
        sys.exit(1)
    print(f"  current price: ${pool.collateral_price('WETH'):,.2f}  "
          f"({len(pool.segments)} liquidity segments loaded)")

    print("\nFetching live Aave v3 positions...")
    try:
        all_positions = fetch_aave_positions(first=MAX_AAVE_USERS)
    except Exception as e:
        print(f"Position fetch failed: {e}")
        print("Check your API key/subgraph access, then retry.")
        sys.exit(1)

    # v1 scope: only positions collateralized in WETH, since that's the only
    # pool we've wired up. Extend fetch_uniswap_pool calls per-asset to widen this.
    positions = [p for p in all_positions if p.collateral_asset == "WETH"]
    print(f"  {len(all_positions)} open borrow positions fetched, "
          f"{len(positions)} are WETH-collateralized (this run's scope)")

    if not positions:
        print("No WETH-collateralized positions in this batch -- try raising "
              "MAX_AAVE_USERS or re-run (subgraph pagination order isn't fixed).")
        sys.exit(0)

    pools = {"WETH": pool}
    price_now = pool.collateral_price("WETH")

    print(f"\nHealth factors at current price (${price_now:,.2f}):")
    underwater_already = 0
    for p in sorted(positions, key=lambda p: p.collateral_qty * price_now
                     * p.liquidation_threshold / max(p.debt_qty, 1e-9))[:15]:
        hf = (p.collateral_qty * price_now * p.liquidation_threshold) / p.debt_qty
        flag = "  <-- ALREADY UNDERWATER" if hf < 1.0 else ""
        if hf < 1.0:
            underwater_already += 1
        print(f"  {p.position_id:12s} HF = {hf:.3f}{flag}")
    if underwater_already:
        print(f"\n  ({underwater_already} position(s) already below HF 1.0 before "
              f"any shock -- real accounts pending liquidation right now)")

    print(f"\nApplying {SHOCK_PCT:+.0%} WETH shock and running cascade...\n")
    logs = run_cascade(positions, pools, initial_shock={"WETH": SHOCK_PCT}, max_rounds=15)

    print("\n--- Summary ---")
    final_price = pool.collateral_price("WETH")
    summary = summarize(logs)
    print(f"Rounds run: {len(logs)}")
    print(f"Accounts liquidated: {summary.accounts_liquidated} "
          f"({summary.liquidation_events} liquidation events)")
    print(f"Debt repaid: ${summary.debt_repaid_usd:,.0f}  |  "
          f"collateral seized: ${summary.collateral_seized_usd:,.0f}")
    print(f"WETH price: ${price_now:,.2f} -> ${final_price:,.2f} "
          f"({(final_price / price_now - 1) * 100:.2f}% total move, "
          f"vs {SHOCK_PCT:.0%} shock applied)")


if __name__ == "__main__":
    main()