"""
Same as live_demo_multi.py, but with a SECOND live pool (WBTC/USDC) wired
up alongside WETH/USDC. This is the difference between "tracking WBTC
exposure" (counted at a static price) and "modeling WBTC contagion" (its
price actually moves when liquidations sell into it, which can then push
OTHER accounts holding WBTC underwater too -- the real multi-asset cascade
the project was scoped toward from the start).

Anything held in an asset with no live pool here (wstETH, LINK, etc.) still
falls back to its static fetch-time price, same as before -- this just
narrows that gap for the two biggest collateral assets instead of closing
it completely (closing it fully would mean a live pool per asset, which is
a bigger lift than this session has room for).
"""

import os
import sys

from live_data import fetch_aave_positions_multi, fetch_uniswap_pool, \
    WETH_USDC_POOL, WBTC_USDC_POOL, require_data_access
from cascade_sim import summarize
from multi_asset import multi_health_factor, run_cascade_multi

SHOCK_PCT = -0.12
MAX_AAVE_USERS = 500


def main():
    require_data_access()

    print("Fetching live WETH/USDC pool...")
    weth_pool = fetch_uniswap_pool(WETH_USDC_POOL, tick_window=15000)
    print(f"  WETH price: ${weth_pool.collateral_price('WETH'):,.2f}")

    print("Fetching live WBTC/USDC pool...")
    try:
        wbtc_pool = fetch_uniswap_pool(WBTC_USDC_POOL, tick_window=15000)
        print(f"  WBTC price: ${wbtc_pool.collateral_price('WBTC'):,.2f}")
        pools = {"WETH": weth_pool, "WBTC": wbtc_pool}
    except Exception as e:
        print(f"  WBTC pool fetch failed: {e} -- continuing with WETH-only pricing.")
        pools = {"WETH": weth_pool}

    print("\nFetching live Aave v3 positions (multi-asset)...")
    positions, static_prices = fetch_aave_positions_multi(first=MAX_AAVE_USERS)
    if not positions:
        print("No positions fetched.")
        sys.exit(0)

    prices_now = dict(static_prices)
    for asset, pool in pools.items():
        prices_now[asset] = pool.collateral_price(asset)

    both_asset_accounts = [p for p in positions
                            if "WETH" in p.collateral and "WBTC" in p.collateral]
    print(f"\n{len(positions)} positions total, {len(both_asset_accounts)} hold "
          f"BOTH WETH and WBTC as collateral -- these are where cross-asset "
          f"contagion actually has somewhere to propagate through.")

    print(f"\nApplying {SHOCK_PCT:+.0%} WETH shock (WBTC untouched initially) "
          f"and running cascade...\n")
    logs = run_cascade_multi(positions, pools, static_prices,
                              initial_shock={"WETH": SHOCK_PCT}, max_rounds=20)

    summary = summarize(logs)
    final_weth = pools["WETH"].collateral_price("WETH")
    final_wbtc = pools["WBTC"].collateral_price("WBTC") if "WBTC" in pools else prices_now.get("WBTC")

    print("\n--- Summary ---")
    print(f"Rounds run: {len(logs)}")
    print(f"Accounts liquidated: {summary.accounts_liquidated} "
          f"({summary.liquidation_events} liquidation events)")
    print(f"Debt repaid: ${summary.debt_repaid_usd:,.0f}  |  "
          f"collateral seized: ${summary.collateral_seized_usd:,.0f}")
    print(f"WETH: ${prices_now['WETH']:,.2f} -> ${final_weth:,.2f} "
          f"({(final_weth/prices_now['WETH'] - 1)*100:.2f}%)")
    if "WBTC" in pools:
        wbtc_move = (final_wbtc / prices_now["WBTC"] - 1) * 100
        print(f"WBTC: ${prices_now['WBTC']:,.2f} -> ${final_wbtc:,.2f} "
              f"({wbtc_move:+.2f}%)  <- this moved even though only WETH was shocked "
              f"directly; any nonzero move here is pure cross-asset contagion, "
              f"via accounts that held both and got liquidated")


if __name__ == "__main__":
    main()