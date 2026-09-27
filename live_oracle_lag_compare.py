"""
Runs the SAME live shock through both the instant-oracle cascade
(cascade_sim.run_cascade, used everywhere else in this project) and the
oracle-lag cascade (oracle_lag.run_cascade_oracle_lag), on an identical
deep-copied starting snapshot, so the only difference between the two
results is the oracle mechanism itself.
"""

import copy
import os
import sys

from live_data import fetch_aave_positions, fetch_uniswap_pool, WETH_USDC_POOL, require_data_access
from cascade_sim import run_cascade, summarize
from oracle_lag import run_cascade_oracle_lag

SHOCK_PCT = -0.12
DEVIATION_THRESHOLD = 0.005  # 0.5% -- realistic Chainlink major-pair value


def main():
    require_data_access()

    print("Fetching live pool + positions once (both runs use identical copies)...")
    base_pool = fetch_uniswap_pool(WETH_USDC_POOL, tick_window=15000)
    all_positions = fetch_aave_positions(first=500, verbose=False)
    base_positions = [p for p in all_positions if p.collateral_asset == "WETH"]
    starting_price = base_pool.collateral_price("WETH")
    print(f"  {len(base_positions)} WETH-collateralized positions, "
          f"starting price ${starting_price:,.2f}")

    print(f"\n=== Instant oracle (baseline, what every other script uses) ===")
    pos_a = copy.deepcopy(base_positions)
    pool_a = copy.deepcopy(base_pool)
    logs_a = run_cascade(pos_a, {"WETH": pool_a}, initial_shock={"WETH": SHOCK_PCT},
                          max_rounds=20, verbose=True)
    final_a = pool_a.collateral_price("WETH")

    print(f"\n=== Oracle lag ({DEVIATION_THRESHOLD:.1%} deviation threshold) ===")
    pos_b = copy.deepcopy(base_positions)
    pool_b = copy.deepcopy(base_pool)
    logs_b = run_cascade_oracle_lag(pos_b, {"WETH": pool_b},
                                     initial_shock={"WETH": SHOCK_PCT},
                                     deviation_threshold=DEVIATION_THRESHOLD,
                                     max_rounds=30, verbose=True)
    final_b = pool_b.collateral_price("WETH")

    sum_a, sum_b = summarize(logs_a), summarize(logs_b)
    liq_a, liq_b = sum_a.accounts_liquidated, sum_b.accounts_liquidated

    print("\n--- Comparison ---")
    print(f"{'':30s}{'Instant oracle':>18s}{'Oracle lag':>18s}")
    print(f"{'Rounds to resolve':30s}{len(logs_a):>18d}{len(logs_b):>18d}")
    print(f"{'Accounts liquidated':30s}{liq_a:>18d}{liq_b:>18d}")
    print(f"{'Liquidation events':30s}"
          f"{sum_a.liquidation_events:>18d}{sum_b.liquidation_events:>18d}")
    print(f"{'Debt repaid':30s}"
          f"{'$' + format(sum_a.debt_repaid_usd, ',.0f'):>18s}"
          f"{'$' + format(sum_b.debt_repaid_usd, ',.0f'):>18s}")
    print(f"{'Final WETH price':30s}{'$' + format(final_a, ',.2f'):>18s}"
          f"{'$' + format(final_b, ',.2f'):>18s}")
    print(f"{'Total realized move':30s}"
          f"{(final_a/starting_price-1)*100:>17.2f}%{(final_b/starting_price-1)*100:>17.2f}%")

    if liq_a != liq_b or abs(final_a - final_b) > 0.01:
        print("\nThe two runs diverge -- oracle lag measurably changed the outcome "
              "on this shock/data. That divergence IS the finding: it's the gap "
              "between what an instant-price model predicts and what the real "
              "on-chain mechanism (which reads a lagging oracle) would actually do.")
    else:
        print("\nThe two runs converged to the same result -- for this particular "
              "shock size and pool depth, round-to-round price moves were large "
              "enough to cross the deviation threshold every round anyway, so "
              "lag didn't change the outcome. Try a smaller shock or a deeper "
              "pool (smaller per-round price impact) to see the mechanism bite.")


if __name__ == "__main__":
    main()