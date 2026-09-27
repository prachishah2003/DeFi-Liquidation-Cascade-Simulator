"""
Cross-protocol liquidation cascade: Aave AND Compound positions, sharing
the SAME WETH/USDC pool. This is the actual contagion demonstration --
a WETH shock can trigger liquidations on Aave, whose AMM selling moves
price enough to trigger Compound liquidations too (or vice versa), each
protocol amplifying the other through the shared market they both sell into.

MakerDAO is intentionally NOT included here -- it uses a structurally
different vault/CDP model rather than the pooled-market pattern Aave and
Compound share, so it doesn't fit fetch_lending_positions as-is. Adding it
is a real next step, but a separate one (different query shape entirely,
not a parameter tweak) -- flagged here rather than guessed at blind.
"""

import os
import sys

from live_data import fetch_aave_positions, fetch_compound_positions, \
    fetch_uniswap_pool, WETH_USDC_POOL, require_data_access
from cascade_sim import run_cascade, summarize

SHOCK_PCT = -0.12
MAX_ACCOUNTS_PER_PROTOCOL = 500


def main():
    require_data_access()

    print("Fetching live WETH/USDC pool liquidity...")
    pool = fetch_uniswap_pool(WETH_USDC_POOL, tick_window=15000)
    price_now = pool.collateral_price("WETH")
    print(f"  current price: ${price_now:,.2f}")

    print("\nFetching live Aave v3 positions...")
    aave_all = fetch_aave_positions(first=MAX_ACCOUNTS_PER_PROTOCOL)
    aave_positions = [p for p in aave_all if p.collateral_asset == "WETH"]

    print("\nFetching live Compound v3 positions...")
    try:
        compound_all = fetch_compound_positions(first=MAX_ACCOUNTS_PER_PROTOCOL)
        compound_positions = [p for p in compound_all if p.collateral_asset == "WETH"]
    except Exception as e:
        print(f"  Compound fetch failed: {e}")
        print("  This is genuinely new/unverified against the live API (see module "
              "docstring) -- if this is a field-name error, same process as every "
              "other schema issue this session: paste the error, it's usually a "
              "one-line fix. Continuing with Aave-only for now.")
        compound_positions = []

    print(f"\n{len(aave_positions)} WETH-collateralized Aave positions, "
          f"{len(compound_positions)} WETH-collateralized Compound positions")

    combined = aave_positions + compound_positions
    if not combined:
        print("No positions from either protocol -- nothing to simulate.")
        sys.exit(0)

    pools = {"WETH": pool}

    print(f"\nWorst health factors across BOTH protocols (worst 15):")
    from cascade_sim import health_factor
    prices = {"WETH": price_now}
    ranked = sorted(combined, key=lambda p: health_factor(p, prices))[:15]
    for p in ranked:
        hf = health_factor(p, prices)
        flag = "  <-- ALREADY UNDERWATER" if hf < 1.0 else ""
        print(f"  {p.position_id:14s} [{p.protocol:8s}] HF = {hf:.3f}{flag}")
        if hf < 0.01:
            print(f"      ** SUSPICIOUS: HF this close to zero usually means a "
                  f"scaling bug, not real risk. Raw values: collateral_qty="
                  f"{p.collateral_qty:.6g} {p.collateral_asset}, debt_qty="
                  f"{p.debt_qty:.6g} {p.debt_asset}, threshold={p.liquidation_threshold}. "
                  f"A debt_qty many orders of magnitude too large is the most "
                  f"likely culprit -- compare against this account's other "
                  f"position sizes for a sanity check before trusting this one. **")

    print(f"\nApplying {SHOCK_PCT:+.0%} WETH shock and running the SHARED "
          f"cross-protocol cascade...\n")
    logs = run_cascade(combined, pools, initial_shock={"WETH": SHOCK_PCT}, max_rounds=20)

    final_price = pool.collateral_price("WETH")
    summary = summarize(logs)
    aave_liquidated = sum(1 for p in aave_positions if p.liquidated_rounds)
    compound_liquidated = sum(1 for p in compound_positions if p.liquidated_rounds)

    print("\n--- Summary ---")
    print(f"Rounds run: {len(logs)}")
    print(f"Liquidation events: {summary.liquidation_events} "
          f"(a position can be hit more than once across rounds -- the close "
          f"factor is 50% until HF drops below 0.95, so a partial liquidation "
          f"can leave a position still underwater)")
    print(f"Unique accounts liquidated: {summary.accounts_liquidated} "
          f"({aave_liquidated} Aave, {compound_liquidated} Compound)")
    print(f"Debt repaid: ${summary.debt_repaid_usd:,.0f}  |  "
          f"collateral seized: ${summary.collateral_seized_usd:,.0f}")
    print(f"WETH price: ${price_now:,.2f} -> ${final_price:,.2f} "
          f"({(final_price/price_now - 1)*100:.2f}% total move, "
          f"vs {SHOCK_PCT:.0%} shock applied)")

    if aave_liquidated and compound_liquidated:
        print("\nBoth protocols had liquidations in this run -- worth checking "
              "the round-by-round log above for whether one protocol's early "
              "liquidations are what pushed the other protocol's positions "
              "underwater (the actual contagion mechanism, not just coincidence).")


if __name__ == "__main__":
    main()