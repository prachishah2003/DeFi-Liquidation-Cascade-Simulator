"""
The liquidation cliff, and the bad debt on the other side of it.

Runs the same shock through the same book twice:

  1. FREE MODEL   -- every underwater position is liquidated instantly and
                    costlessly. This is what the rest of the project assumed,
                    and what most public liquidation-risk tools assume.
  2. PRICED MODEL -- a liquidation happens only if the liquidation bonus
                    covers slippage, gas and the flash-loan fee. Positions
                    nobody will touch stay open, and their shortfall accrues
                    to the protocol as bad debt.

The gap between the two is the point. The free model says a big shock
produces a lot of liquidations; the priced model says that past a certain
shock size liquidations *stop*, because unwinding the collateral costs more
than the bonus pays. Nothing gets liquidated, and the protocol is left
holding the loss instead.

    export GRAPH_API_KEY=...
    python3 bad_debt_sweep.py

Sensitivity note: the cliff location depends on `LiquidatorEconomics` --
gas price above all. Those are assumptions, not measurements, so the script
sweeps gas as well and prints how far the cliff moves.
"""

import copy
import os
import sys

from live_data import fetch_lending_positions_with_coverage, fetch_uniswap_pool, \
    fetch_venue_router, WETH_USDC_POOL, WETH_USDC_POOLS, AAVE_V3_SUBGRAPH_ID, \
    DEFAULT_OFFCHAIN_DEPTH_USD_PER_1PCT, require_data_access
from cascade_sim import run_cascade, summarize
from liquidator import LiquidatorEconomics
import chart_style as cs

SHOCK_SIZES_PCT = [-2, -4, -6, -8, -10, -12, -15, -18, -22, -26, -30, -35, -40]
GAS_SCENARIOS = [("calm", 10.0), ("busy", 60.0), ("crisis", 300.0)]

# Where the liquidated collateral is assumed to be sold. This turns out to
# matter more than anything else in the model: routing an entire protocol's
# liquidations through one 0.3% pool is not conservative, it is wrong, and it
# inflates every impact figure severalfold.
VENUE_SCENARIOS = [
    ("one 0.3% pool", {"0.30%": WETH_USDC_POOLS["0.30%"]}, 0.0),
    ("all fee tiers", WETH_USDC_POOLS, 0.0),
    ("tiers + off-chain", WETH_USDC_POOLS, DEFAULT_OFFCHAIN_DEPTH_USD_PER_1PCT),
]


def run_one(positions_template, venue_template, pct, economics):
    positions = copy.deepcopy(positions_template)
    venue = copy.deepcopy(venue_template)
    logs = run_cascade(positions, {"WETH": venue}, {"WETH": pct / 100},
                       max_rounds=100, verbose=False, economics=economics)
    summary = summarize(logs)
    return summary, venue.collateral_price("WETH")


def sweep(positions, venue, economics, label, quiet=False):
    if not quiet:
        print(f"\n--- {label} ---")
    rows = []
    start_price = venue.collateral_price("WETH")
    for pct in SHOCK_SIZES_PCT:
        s, final_price = run_one(positions, venue, pct, economics)
        rows.append({
            "shock_pct": pct,
            "accounts": s.accounts_liquidated,
            "skipped": s.accounts_skipped_unprofitable,
            "seized_usd": s.collateral_seized_usd,
            "bad_debt_usd": s.bad_debt_usd,
            "realized_pct": (final_price / start_price - 1) * 100,
            "data_exhausted": bool(s.pools_data_exhausted),
        })
        if not quiet:
            print(f"  shock {pct:+3d}%  liquidated {s.accounts_liquidated:4d}  "
                  f"skipped {s.accounts_skipped_unprofitable:4d}  "
                  f"seized ${s.collateral_seized_usd/1e6:7.1f}M  "
                  f"bad debt ${s.bad_debt_usd/1e6:7.2f}M  "
                  f"realized {rows[-1]['realized_pct']:+6.1f}%"
                  + ("   [tick data exhausted]" if s.pools_data_exhausted else ""))
    return rows


def find_turning_point(rows):
    """The shock at which liquidation VOLUME peaks.

    This is the economically interesting point, and it is not where
    liquidations stop outright -- a few always clear. Past this shock size a
    LARGER crash produces LESS liquidation, because the marginal position
    costs more to unwind than the bonus pays. Everything the market would
    otherwise have liquidated past this point turns into bad debt instead.

    Returns (shock_pct, peak_seized_usd), or None if volume never turns over
    within the swept range (in which case the sweep needs to go further)."""
    if not rows:
        return None
    peak = max(rows, key=lambda r: r["seized_usd"])
    deeper = [r for r in rows if r["shock_pct"] < peak["shock_pct"]]
    if not deeper:
        return None            # volume still rising at the edge of the sweep
    return peak["shock_pct"], peak["seized_usd"]


def plot(free_rows, priced_rows, gas_rows, out_path="bad_debt_sweep.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping chart; tables above have "
              "everything. `pip install matplotlib` for the PNG.)")
        return

    shocks = [r["shock_pct"] for r in free_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=cs.SURFACE)
    for ax in (ax1, ax2):
        cs.style_axes(ax)

    # --- panel 1: the cliff ------------------------------------------------
    ax1.plot(shocks, [r["seized_usd"] / 1e6 for r in free_rows], "--o",
             color=cs.SERIES_2, linewidth=2, markersize=7,
             markeredgecolor=cs.SURFACE, markeredgewidth=2,
             label="Free model (liquidations are costless)")
    ax1.plot(shocks, [r["seized_usd"] / 1e6 for r in priced_rows], "-o",
             color=cs.SERIES_1, linewidth=2, markersize=8,
             markeredgecolor=cs.SURFACE, markeredgewidth=2,
             label="Priced model (liquidator must profit)")

    turn = find_turning_point(priced_rows)
    if turn is not None:
        turn_pct, _ = turn
        ax1.axvspan(min(shocks), turn_pct, color=cs.CRITICAL, alpha=0.06,
                    zorder=0, linewidth=0)
        ax1.annotate(f"past {turn_pct:+d}%, a bigger crash\nliquidates LESS,"
                     f" not more",
                     xy=(0.02, 0.03), xycoords="axes fraction",
                     fontsize=8.5, color=cs.CRITICAL, va="bottom")
    ax1.set_xlabel("Exogenous shock (%)")
    ax1.set_ylabel("Collateral liquidated ($M)")
    cs.title(ax1, "Liquidations stop before liquidity does")
    cs.style_legend(ax1, loc="upper right")

    # --- panel 2: bad debt under three VENUE assumptions --------------------
    # Progressively lighter weights and dash patterns, so three near-identical
    # curves are visibly three curves rather than one thick one.
    styles = [(":", "^", cs.CRITICAL, 1.8), ("--", "s", cs.SERIES_2, 2.0),
              ("-", "o", cs.SERIES_1, 2.6)]
    spreads = []
    for (label, _addrs, _depth), (dash, marker, colour, lw) in zip(
            VENUE_SCENARIOS, styles):
        rows = gas_rows[label]
        series = [r["bad_debt_usd"] / 1e6 for r in rows]
        spreads.append(series)
        ax2.plot(shocks, series, linestyle=dash, marker=marker,
                 color=colour, linewidth=lw, markersize=6,
                 markeredgecolor=cs.SURFACE, markeredgewidth=1.2,
                 label=label)
    ax2.set_xlabel("Exogenous shock (%)")
    ax2.set_ylabel("Protocol bad debt ($M)")
    cs.title(ax2, "Bad debt depends on where you sell")
    cs.style_legend(ax2, loc="upper left")

    # quantify how close the three regimes actually are, and say it on the chart
    peak = max(max(s) for s in spreads) if spreads else 0.0
    if peak > 0:
        worst_gap = max(
            abs(a - b)
            for s1 in spreads for s2 in spreads
            for a, b in zip(s1, s2)
        )
        if worst_gap / peak < 0.15:
            ax2.annotate(
                "the three curves nearly coincide:\n"
                "the venue assumption barely moves bad debt here\n"
                f"(widest gap {worst_gap/peak:.0%} of peak)",
                xy=(0.97, 0.14), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=8, color=cs.MUTED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=cs.SURFACE)
    print(f"\nChart saved to {out_path}")


def main():
    require_data_access()

    print("Fetching live venues + Aave positions (once; every trial below runs "
          "on an independent copy of this same snapshot)...")
    routers = {}
    for label, addresses, depth in VENUE_SCENARIOS:
        routers[label] = fetch_venue_router(
            "WETH", pool_addresses=addresses,
            offchain_depth_usd_per_1pct=depth, verbose=True)

    realistic = routers[VENUE_SCENARIOS[-1][0]]
    all_positions, coverage = fetch_lending_positions_with_coverage(
        AAVE_V3_SUBGRAPH_ID, "aave", first=500)
    positions = [p for p in all_positions if p.collateral_asset == "WETH"]
    print(f"\n  {len(positions)} WETH-collateralized positions, "
          f"starting price ${realistic.collateral_price('WETH'):,.2f}")
    print(f"  {coverage}")

    free_rows = sweep(positions, realistic, None,
                      "FREE MODEL (liquidations costless, realistic venues)")
    priced_rows = sweep(positions, realistic,
                        LiquidatorEconomics(gas_price_gwei=60.0),
                        "PRICED MODEL (60 gwei, realistic venues)")

    gas_rows = {}
    for label, _addresses, _depth in VENUE_SCENARIOS:
        gas_rows[label] = sweep(positions, routers[label],
                                LiquidatorEconomics(gas_price_gwei=60.0),
                                f"PRICED MODEL (venues: {label})")

    print("\n--- Gas sensitivity (realistic venues) ---")
    for label, gwei in GAS_SCENARIOS:
        rows = sweep(positions, realistic,
                     LiquidatorEconomics(gas_price_gwei=gwei), label, quiet=True)
        worst = max(r["bad_debt_usd"] for r in rows)
        total_liq = max(r["seized_usd"] for r in rows)
        print(f"  {label:8s} ({gwei:5.0f} gwei): peak liquidated "
              f"${total_liq/1e6:7.1f}M, peak bad debt ${worst/1e6:7.2f}M")

    print("\n--- Venue sensitivity (60 gwei) ---")
    for label, _a, _d in VENUE_SCENARIOS:
        rows = gas_rows[label]
        deepest = min(rows, key=lambda r: r["shock_pct"])
        print(f"  {label:18s}: at {deepest['shock_pct']:+d}% the modelled move is "
              f"{deepest['realized_pct']:+7.2f}%, "
              f"${deepest['seized_usd']/1e6:7.1f}M liquidated, "
              f"${deepest['bad_debt_usd']/1e6:7.2f}M bad debt")
    one_pool = min(gas_rows[VENUE_SCENARIOS[0][0]], key=lambda r: r["shock_pct"])
    realistic_row = min(gas_rows[VENUE_SCENARIOS[-1][0]], key=lambda r: r["shock_pct"])
    if realistic_row["realized_pct"] < 0:
        ratio = one_pool["realized_pct"] / realistic_row["realized_pct"]
        print(f"  -> modelling only the 0.3% pool overstates the deepest move "
              f"by {ratio:.1f}x")

    print("\n=== Headline ===")
    turn = find_turning_point(priced_rows)
    worst_free = max(r["seized_usd"] for r in free_rows)
    worst_priced = max(r["seized_usd"] for r in priced_rows)
    worst_debt = max(r["bad_debt_usd"] for r in priced_rows)
    print(f"  Free model, worst case:   ${worst_free/1e6:,.1f}M liquidated, "
          f"$0 bad debt (it cannot produce any)")
    print(f"  Priced model, worst case: ${worst_priced/1e6:,.1f}M liquidated, "
          f"${worst_debt/1e6:,.2f}M bad debt")
    if turn is not None:
        turn_pct, peak = turn
        deepest = min(priced_rows, key=lambda r: r["shock_pct"])
        print(f"  Liquidation volume PEAKS at {turn_pct:+d}% "
              f"(${peak/1e6:,.1f}M). Past that, a larger crash liquidates "
              f"LESS: at {deepest['shock_pct']:+d}% only "
              f"${deepest['seized_usd']/1e6:,.1f}M clears, with "
              f"{deepest['skipped']} accounts left untouched and "
              f"${deepest['bad_debt_usd']/1e6:,.2f}M of bad debt.")
        print(f"  That turnover is the headline result: the binding constraint "
              f"on a cascade is liquidator profitability, not pool liquidity.")
    else:
        print("  Liquidation volume had not turned over by the deepest shock "
              "swept -- extend SHOCK_SIZES_PCT to find the peak.")
    print(f"\n  All figures are scoped to the sample above "
          f"({coverage.coverage_pct:.1f}% of Aave borrows) -- multiply with care.")
    print("  Gas prices, gas units and the flash-loan fee are assumptions, not "
          "measurements. The three-regime panel is the sensitivity check.")

    plot(free_rows, priced_rows, gas_rows)


if __name__ == "__main__":
    main()
