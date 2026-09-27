"""
Runs score_systemic_assets against live Aave v3 multi-asset positions --
answers "which collateral asset, if it crashed, would flip the most
accounts underwater?" using real position data.
"""

import os
import sys

from live_data import fetch_aave_positions_multi, fetch_uniswap_pool, WETH_USDC_POOL, require_data_access
from systematic_scoring import score_systemic_assets, print_ranking

SHOCK_PCT = -0.20


def main():
    require_data_access()

    print("Fetching live WETH/USDC pool (for a real current WETH price)...")
    pool = fetch_uniswap_pool(WETH_USDC_POOL)
    weth_price = pool.collateral_price("WETH")

    print("Fetching live Aave v3 positions (multi-asset)...")
    positions, static_prices = fetch_aave_positions_multi(first=500)
    if not positions:
        print("No positions fetched.")
        sys.exit(0)

    prices = dict(static_prices)
    prices["WETH"] = weth_price

    print(f"\n{len(positions)} multi-asset positions loaded")
    results = score_systemic_assets(positions, prices, shock_pct=SHOCK_PCT)
    print_ranking(results, shock_pct=SHOCK_PCT, top_n=20)

    print("\nNote: 'accounts_holding' counts every account that has this asset "
          "as ANY part of its collateral, not just accounts collateralized "
          "purely in it -- that's deliberate, since a diversified account's "
          "safety still partly depends on every asset it holds.")


if __name__ == "__main__":
    main()