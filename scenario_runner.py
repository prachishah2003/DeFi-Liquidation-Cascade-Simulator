"""
Run correlated-shock and depeg scenarios against the live multi-asset book.

The single-asset path in this project models one collateral asset against
stablecoin debt priced at exactly $1.00. That is fine for studying the
cascade mechanism and useless for asking what a real crash does to a real
book, because real accounts hold four kinds of staked ether at once and the
dollar their debt is denominated in has broken before.

This runs the multi-asset engine, where every asset carries its own price,
so a stablecoin depeg can move the DEBT side -- which is the only way to see
that a depeg makes borrowers safer rather than riskier.

    export GRAPH_API_KEY=...
    python3 scenario_runner.py
"""

import copy
import os
import sys

from live_data import fetch_lending_positions_multi_with_coverage, \
    fetch_venue_router, fetch_lst_venues, AAVE_V3_SUBGRAPH_ID, \
    require_data_access
from multi_asset import run_cascade_multi, multi_health_factor, drop_unpriced
from cascade_sim import summarize
from liquidator import LiquidatorEconomics
from market_impact import ImpactDecay
from factor_model import default_scenarios, DEFAULT_ETH_BETAS

# MEASURED, not assumed: estimate_kappa.py finds ~0.00 reversion over the
# 14-36 seconds in which one wave of liquidations triggers the next. Impact
# does revert -- about 90% of it -- but over ten minutes or so, by which time
# a cascade has finished. Using a longer-horizon kappa here would materially
# understate risk, so the default is the cascade-speed figure.
KAPPA = 0.0
GAS_GWEI = 60.0


def run_scenario(positions, static_prices, venues, scenario, economics, decay):
    prices = dict(static_prices)
    for asset, venue in venues.items():
        prices[asset] = venue.collateral_price(asset)
    returns = scenario.asset_returns(prices.keys())
    shock = {asset: r for asset, r in returns.items() if abs(r) > 1e-12}

    # Deep-copy the WHOLE mapping in one call, never venue by venue. Every
    # LST venue holds a reference to the same WETH router, and a single
    # deepcopy preserves that sharing through its memo; copying each value
    # separately would give each LST its own private WETH book and silently
    # delete the contagion channel this models.
    pools = copy.deepcopy(venues)
    sim_positions = copy.deepcopy(positions)

    # Underwater at the SHOCKED prices, before any cascade -- i.e. what the
    # shock alone does. Computing it at current prices (as a first version of
    # this did) reports the same number for every scenario, which is a
    # giveaway that it is measuring nothing about the scenario.
    shocked_prices = {a: p * (1 + returns.get(a, 0.0)) for a, p in prices.items()}
    underwater_before = sum(1 for p in sim_positions
                            if not p.is_fully_liquidated()
                            and multi_health_factor(p, shocked_prices) < 1.0)
    logs = run_cascade_multi(sim_positions, pools, prices, shock,
                             max_rounds=100, verbose=False,
                             economics=economics, decay=decay)
    summary = summarize(logs)
    final = pools["WETH"].collateral_price("WETH")
    implied = prices["WETH"] * (1 + returns.get("WETH", 0.0))
    return {
        "scenario": scenario,
        "underwater_before": underwater_before,
        "summary": summary,
        "weth_final": final,
        "weth_implied": implied,
        "extra_move_pp": (final / implied - 1) * 100 if implied else 0.0,
    }


def main():
    require_data_access()

    print("Fetching venues and multi-asset Aave positions...")
    router = fetch_venue_router("WETH", verbose=False)
    # Two-hop venues for the staking tokens, all sharing the WETH router
    venues = {"WETH": router}
    venues.update(fetch_lst_venues(router, verbose=True))
    print(f"  price impact modelled for: {sorted(venues)}")

    positions, static_prices, coverage = \
        fetch_lending_positions_multi_with_coverage(
            AAVE_V3_SUBGRAPH_ID, "aave", first=500, verbose=False)
    positions, missing = drop_unpriced(positions, static_prices, verbose=True)
    if not positions:
        print("No priceable multi-asset positions fetched.")
        sys.exit(1)

    for asset, venue in venues.items():
        static_prices[asset] = venue.collateral_price(asset)
    print(f"  {len(positions)} multi-asset accounts; {coverage}")

    assets = sorted({a for p in positions for a in p.collateral})
    known = [a for a in assets if a in DEFAULT_ETH_BETAS]
    unknown = [a for a in assets if a not in DEFAULT_ETH_BETAS]
    print(f"  collateral assets with a factor loading: {known}")
    if unknown:
        print(f"  NO factor loading (treated as beta 0, i.e. unaffected by the "
              f"market factor -- almost certainly wrong for these): {unknown}")

    economics = LiquidatorEconomics(gas_price_gwei=GAS_GWEI)
    decay = ImpactDecay(kappa=KAPPA)
    offchain = sum(v.depth_usd_per_1pct for v in router.linear)
    print(f"\n  {decay.describe()}; {GAS_GWEI:.0f} gwei; "
          f"off-chain depth ${offchain/1e6:,.0f}M per 1%\n")

    header = (f"{'scenario':46s}{'underwater':>11s}{'liquidated':>11s}"
              f"{'bad debt':>12s}{'extra move':>12s}")
    print(header)
    print("-" * len(header))

    for scenario in default_scenarios():
        row = run_scenario(positions, static_prices, venues, scenario,
                           economics, decay)
        s = row["summary"]
        print(f"{scenario.name[:45]:46s}"
              f"{row['underwater_before']:>11d}"
              f"{s.accounts_liquidated:>11d}"
              f"{'$' + format(s.bad_debt_usd / 1e6, ',.2f') + 'M':>12s}"
              f"{row['extra_move_pp']:>11.2f}pp")

    print("\n  'underwater' counts accounts already below HF 1 at the shocked "
          "prices, before any cascade. 'extra move' is how much further WETH "
          "fell than the shock itself implied -- the cascade's own "
          "contribution, which is the quantity this whole project exists to "
          "measure.")
    print("\n  Read the depeg rows carefully: where a broken stablecoin is "
          "DEBT it makes positions safer, because the debt is cheaper to "
          "repay. Whether that dominates depends on the book.")
    modelled = sorted(venues)
    unmodelled = sorted({a for p in positions for a in p.collateral}
                        - set(modelled))
    print(f"\n  Price impact is modelled for {modelled}. Everything else "
          f"({len(unmodelled)} assets) is priced statically and sells at its "
          f"mark with no slippage, which UNDERSTATES their contribution.")
    print(f"\n  EVERY NUMBER ABOVE assumes ${offchain/1e6:,.0f}M of off-chain "
          f"depth per 1% of price move, on top of the fetched Uniswap books. "
          f"That is the most consequential assumption here and it is an "
          f"order-of-magnitude judgement, not a measurement: on the same book, "
          f"the shock that breaks the system sits at -26% with on-chain "
          f"execution only and at -49% with this much off-chain depth. Run "
          f"critical_shock.py for that sweep, and read this table as "
          f"conditional on the figure above.")
    print(f"\n  Scoped to {coverage.coverage_pct:.1f}% of Aave borrows, one "
          f"snapshot, kappa={KAPPA}, and the betas in factor_model.py "
          f"(structural defaults unless estimate_betas has been run).")


if __name__ == "__main__":
    main()
