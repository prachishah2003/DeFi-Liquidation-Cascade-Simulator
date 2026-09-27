"""
Sweeps shock magnitude (-2% to -40%) against the SAME live position/pool
snapshot and records how much extra move the endogenous cascade adds at
each size. Saves a chart: shock size vs. (a) total realized move and
(b) amplification = realized - shock.

Motivation: the two backtest runs already hinted at nonlinear scaling
(-12% shock -> ~1.8-2pp amplification; -9.5% shock -> ~0.6pp). This
sweep checks whether that's a real pattern or a coincidence of those two
specific runs.
"""

import copy
import os
import sys

from live_data import fetch_aave_positions, fetch_uniswap_pool, \
    fetch_venue_router, WETH_USDC_POOL, require_data_access
from cascade_sim import run_cascade, summarize
from chart_style import (SURFACE, INK, MUTED, GRID, AXIS, SERIES_1, CRITICAL,
                         style_axes, style_legend, title)

SHOCK_SIZES_PCT = [-2, -4, -6, -8, -10, -12, -15, -18, -22, -26, -30, -35, -40]


def run_sweep(positions_template, pool_template, shock_sizes_pct=SHOCK_SIZES_PCT):
    """positions_template/pool_template are the FRESH, unmutated starting
    state -- each trial gets its own deep copy so trials don't leak state
    into each other (run_cascade mutates positions and the venue in place).

    `pool_template` may be a single Pool or a VenueRouter; the router is
    deliberately Pool-compatible, and the router is what you want."""
    results = []
    for pct in shock_sizes_pct:
        positions = copy.deepcopy(positions_template)
        pool = copy.deepcopy(pool_template)
        pools = {"WETH": pool}
        price_before = pool.collateral_price("WETH")

        logs = run_cascade(positions, pools, initial_shock={"WETH": pct / 100},
                            max_rounds=20, verbose=False)

        price_after = pool.collateral_price("WETH")
        realized_pct = (price_after / price_before - 1) * 100
        summary = summarize(logs)
        amplification = realized_pct - pct
        ran_dry = any(l.pool_ran_dry for l in logs)
        # a run cut short by missing tick data is not a result at all
        data_exhausted = bool(summary.pools_data_exhausted)

        results.append({
            "shock_pct": pct,
            "realized_pct": realized_pct,
            "amplification_pct": amplification,
            "accounts_liquidated": summary.accounts_liquidated,
            "liquidation_events": summary.liquidation_events,
            "collateral_seized_usd": summary.collateral_seized_usd,
            "rounds": len(logs),
            "ran_dry": ran_dry,
            "data_exhausted": data_exhausted,
        })
        if data_exhausted:
            dry_flag = ("  ** TICK DATA EXHAUSTED -- this point is a data limit, "
                        "NOT a finding; widen tick_window and re-run **")
        elif ran_dry:
            dry_flag = ("  ** POOL LIQUIDITY GENUINELY EXHAUSTED -- real economic "
                        "boundary, the book has nothing left to sell into **")
        else:
            dry_flag = ""
        print(f"  shock {pct:+3d}%  ->  realized {realized_pct:+6.2f}%  "
              f"(amplification {amplification:+.2f}pp, "
              f"{summary.accounts_liquidated} accounts / "
              f"{summary.liquidation_events} events, "
              f"${summary.collateral_seized_usd/1e6:,.1f}M seized, "
              f"{len(logs)} rounds){dry_flag}")
    return results


# Points where the model walked off the end of the fetched tick data are
# marked distinctly -- with a different MARKER and a shaded band, not with
# colour alone -- because the original version of this chart plotted them
# identically to real results, which is how a data-coverage artifact ended up
# in the README as "cascades have a real ceiling".
SERIES = SERIES_1


def plot_results(results, out_path="shock_sweep.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed, just save to file
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping chart, table above still "
              "has everything. `pip install matplotlib` if you want the PNG.)")
        return

    shocks = [r["shock_pct"] for r in results]
    realized = [r["realized_pct"] for r in results]
    amplification = [r["amplification_pct"] for r in results]
    bad = [bool(r.get("data_exhausted")) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)

    # shade the region where the fetched book ran out, on both panels
    bad_shocks = [s for s, b in zip(shocks, bad) if b]
    for ax in (ax1, ax2):
        style_axes(ax)
        if bad_shocks:
            ax.axvspan(min(shocks), max(bad_shocks), color=CRITICAL, alpha=0.06,
                       zorder=0, linewidth=0)

    ax1.plot(shocks, shocks, "--", color=AXIS, linewidth=2, zorder=2,
             label="No cascade (realized = shock)")
    ax1.plot(shocks, realized, "-", color=SERIES, linewidth=2, zorder=3,
             label="Model (with cascade)")
    ax2.plot(shocks, amplification, "-", color=SERIES, linewidth=2, zorder=3)

    for ax, series in ((ax1, realized), (ax2, amplification)):
        good_x = [s for s, b in zip(shocks, bad) if not b]
        good_y = [v for v, b in zip(series, bad) if not b]
        bad_x = [s for s, b in zip(shocks, bad) if b]
        bad_y = [v for v, b in zip(series, bad) if b]
        ax.plot(good_x, good_y, "o", color=SERIES, markersize=8, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2)
        if bad_x:
            # different marker AND different colour -- identity never rests on
            # colour alone (matters in print and for colour-vision deficiency)
            ax.plot(bad_x, bad_y, "X", color=CRITICAL, markersize=11, zorder=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    linestyle="none")

    if bad_shocks:
        ax1.plot([], [], "X", color=CRITICAL, markersize=11, linestyle="none",
                 label="Tick data exhausted -- not a result")

    ax1.set_xlabel("Exogenous shock (%)")
    ax1.set_ylabel("Realized total move (%)")
    title(ax1, "Shock vs. realized move")
    style_legend(ax1, loc="lower right")

    ax2.set_xlabel("Exogenous shock (%)")
    ax2.set_ylabel("Amplification (percentage points)")
    title(ax2, "Cascade amplification vs. shock size")
    if bad_shocks:
        ax2.annotate("shaded: fetched book ran out,\nso these understate the cascade",
                     xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8,
                     color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"\nChart saved to {out_path}")
    if bad_shocks:
        print(f"  NOTE: {len(bad_shocks)} point(s) marked X ran off the end of the "
              f"fetched tick data. Re-run with a wider tick_window before quoting "
              f"any ceiling in the cascade -- as drawn, those points are a "
              f"property of the fetch, not of the market.")


if __name__ == "__main__":
    require_data_access()

    print("Fetching live venues + Aave positions (once; each shock trial "
          "below runs on an independent copy of this same snapshot)...")
    # Route across every fee tier plus off-chain depth, not one 0.3% pool.
    # Selling a whole protocol's liquidations into a single pool is not a
    # conservative assumption, it is a wrong one: it inflated the amplification
    # figures this chart reports by more than an order of magnitude.
    #
    # tick_window is wide because extreme shocks (-35%, -40%) need liquidity
    # data covering a wider price range, or a swap runs off the end of the
    # FETCHED book and reports a data limit that looks like (but is not) a
    # real economic saturation effect.
    venue = fetch_venue_router("WETH", tick_window=15000)
    all_positions = fetch_aave_positions(first=500)
    positions = [p for p in all_positions if p.collateral_asset == "WETH"]
    print(f"  {len(positions)} WETH-collateralized positions, "
          f"starting price ${venue.collateral_price('WETH'):,.2f}\n")

    print("Running sweep...")
    results = run_sweep(positions, venue)
    plot_results(results)