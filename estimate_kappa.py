"""
Measure the impact-decay coefficient instead of assuming it.

`market_impact.py` introduced kappa -- the fraction of a cascade's price
dislocation that reverts before the next wave of liquidations -- and the
critical-shock grid showed it dominates every result the model produces. A
parameter that consequential should not be a guess.

It is directly observable. Large trades happen on this pool every day, each
one a small natural experiment: a trade pushes the price, and over the
following minutes the price either stays pushed (permanent impact) or comes
back (temporary impact). Measuring how far it comes back is measuring kappa.

    kappa_hat = (p_later - p_after) / (p_before - p_after)

    0  -> the dislocation never reverts; impact is fully permanent
    1  -> price is back where it started; impact is fully temporary

THE OBVIOUS OBJECTION, AND THE CONTROL FOR IT
---------------------------------------------
Price also moves for reasons that have nothing to do with the trade. Over any
window the market drifts, and drift back toward the starting price gets
scored as reversion even when the trade's impact is entirely permanent.

A first version of this script tried to control for that with a placebo: the
same kappa statistic computed from ordinary small swaps. That does not work,
and the reason is instructive -- an ordinary swap causes no measurable
dislocation, so the denominator of the kappa ratio is ~0 and every placebo
event gets filtered out. The control was structurally incapable of producing
a number.

What works is comparing magnitudes rather than ratios. For each horizon:

    typical dislocation = median |price move caused by a large trade|
    drift               = median |price move over the same horizon|,
                          measured from ordinary swaps

Their ratio is a signal-to-noise figure. When drift over a horizon is as big
as the dislocation being measured, reversion cannot be distinguished from
the market simply wandering, and any kappa read off that horizon is noise
dressed as a measurement. The table prints it, and short horizons -- where
the ratio is favourable -- are the ones to trust.

WHAT THIS DOES NOT ESTABLISH
----------------------------
Ordinary large trades are not forced liquidations. A liquidation is publicly
visible, mechanically predictable, and arrives alongside others, so informed
counterparties may step back exactly when it needs them. If anything this
biases kappa UPWARD relative to the crisis conditions the cascade model cares
about, which makes the resulting figure an optimistic bound -- a reason to
run the model across a range around it rather than at the point estimate.
"""

import os
import statistics
import sys
from typing import List, Optional

from live_data import fetch_recent_swaps, fetch_uniswap_pool, \
    WETH_USDC_POOLS, require_data_access

# Horizons measured in swaps, but reported in seconds too, because the
# translation is the whole point. A liquidation cascade propagates within a
# block or two -- 12 to 30 seconds -- so the only rows that bear on the
# cascade model are the ones covering that long. Reversion that takes ten
# minutes is real and irrelevant: by then the cascade has finished.
HORIZONS = [1, 3, 5, 10, 25, 50]

# Roughly how long a cascade round takes: one to a few Ethereum blocks.
CASCADE_HORIZON_SECONDS = 36.0

# A swap counts as "large" if its notional sits in the top this-fraction of
# the sample. Large enough to move price measurably, common enough to give a
# usable number of events.
LARGE_QUANTILE = 0.99
MIN_EVENTS = 15


def price_from_sqrt_x96(sqrt_price_x96: str, decimals0: int, decimals1: int) -> float:
    """Uniswap stores sqrt(price) as a Q64.96 fixed-point integer."""
    raw = int(sqrt_price_x96) / (2 ** 96)
    price_1_per_0 = raw * raw
    return price_1_per_0 * (10 ** (decimals0 - decimals1))


def drift_magnitude(prices: List[float], indices: List[int],
                    horizon: int) -> float:
    """Median absolute return over `horizon` swaps, from ordinary starts.

    This is what the price does anyway. Any reversion signal smaller than it
    is indistinguishable from the market wandering.
    """
    moves = []
    for i in indices:
        if i + horizon >= len(prices) or prices[i] <= 0:
            continue
        moves.append(abs(prices[i + horizon] / prices[i] - 1.0))
    return statistics.median(moves) if moves else float("nan")


def dislocation_magnitude(prices: List[float], indices: List[int]) -> float:
    """Median absolute price move caused by the large trades themselves."""
    moves = []
    for i in indices:
        if i == 0 or prices[i - 1] <= 0:
            continue
        moves.append(abs(prices[i] / prices[i - 1] - 1.0))
    return statistics.median(moves) if moves else float("nan")


def measure_reversion(prices: List[float], indices: List[int],
                      horizon: int) -> List[float]:
    """kappa for each event, at one horizon.

    Events whose dislocation is too small to measure are dropped: dividing by
    a near-zero denominator manufactures enormous kappas out of rounding.
    """
    out = []
    for i in indices:
        if i == 0 or i + horizon >= len(prices):
            continue
        before, after = prices[i - 1], prices[i]
        later = prices[i + horizon]
        dislocation = after - before
        if abs(dislocation) / before < 1e-4:        # < 1bp: nothing to revert
            continue
        out.append((later - after) / (before - after))
    return out


def summarise(values: List[float], label: str) -> Optional[dict]:
    if len(values) < MIN_EVENTS:
        return None
    clipped = [min(max(v, -1.0), 2.0) for v in values]   # tame the tails
    return {
        "label": label,
        "n": len(clipped),
        "median": statistics.median(clipped),
        "mean": statistics.fmean(clipped),
        "p25": statistics.quantiles(clipped, n=4)[0],
        "p75": statistics.quantiles(clipped, n=4)[2],
    }


def main():
    require_data_access()

    address = WETH_USDC_POOLS["0.05%"]      # the deepest tier, so the most trades
    print("Fetching pool metadata and recent swaps...")
    pool = fetch_uniswap_pool(address, tick_window=2000, verbose=False)
    # Uniswap orders tokens by address, so for this pool token0 is USDC and
    # the raw price is WETH-per-USDC -- a number around 0.00037. Everything
    # below works in the WETH price in dollars instead, because a reversion
    # ratio computed on an inverted price is not the same ratio, and because
    # a price column reading "$0.00" is a bug report.
    weth_is_token0 = pool.token0_symbol == "WETH"
    decimals0, decimals1 = (18, 6) if weth_is_token0 else (6, 18)

    swaps = fetch_recent_swaps(address, max_swaps=6000)
    if len(swaps) < 500:
        print(f"Only {len(swaps)} swaps returned -- too few to estimate kappa.")
        sys.exit(1)

    prices, notionals = [], []
    for swap in swaps:
        try:
            raw = price_from_sqrt_x96(swap["sqrtPriceX96"], decimals0, decimals1)
            if raw <= 0:
                continue
            prices.append(raw if weth_is_token0 else 1.0 / raw)
            notionals.append(abs(float(swap["amountUSD"] or 0)))
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            continue
    if len(prices) != len(swaps):
        print(f"  note: {len(swaps) - len(prices)} swaps unparseable, skipped")

    ranked = sorted(notionals)
    cutoff = ranked[int(LARGE_QUANTILE * (len(ranked) - 1))]
    large = [i for i, v in enumerate(notionals) if v >= cutoff]
    # placebo: ordinary swaps, sampled evenly so they span the same period
    ordinary = [i for i, v in enumerate(notionals)
                if 0 < v < ranked[int(0.5 * (len(ranked) - 1))]]
    step = max(1, len(ordinary) // max(1, len(large)))
    placebo = ordinary[::step][:len(large)]

    print(f"  {len(prices)} usable swaps; {len(large)} large events "
          f"(>= ${cutoff:,.0f} notional); {len(placebo)} placebo events")
    print(f"  price range over the window: ${min(prices):,.2f} .. ${max(prices):,.2f}")

    dislocation = dislocation_magnitude(prices, large)
    print(f"  a large trade moves price by {dislocation:.3%} (median)")

    timestamps = [int(sw["timestamp"]) for sw in swaps]
    span = max(timestamps) - min(timestamps)
    seconds_per_swap = span / max(1, len(timestamps) - 1)
    print(f"  {seconds_per_swap:.1f} seconds between swaps on average, so a "
          f"cascade round (~{CASCADE_HORIZON_SECONDS:.0f}s) spans roughly "
          f"{CASCADE_HORIZON_SECONDS / seconds_per_swap:.0f} swaps")

    print(f"\n{'horizon':>8s}{'~time':>9s}{'n':>5s}{'kappa':>9s}{'IQR':>18s}"
          f"{'drift':>9s}{'sig/noise':>11s}{'verdict':>12s}")
    print("-" * 82)

    rows = []
    for horizon in HORIZONS:
        treat = summarise(measure_reversion(prices, large, horizon), "large")
        if treat is None:
            continue
        drift = drift_magnitude(prices, placebo, horizon)
        snr = dislocation / drift if drift and drift == drift and drift > 0 else float("nan")
        trustworthy = snr == snr and snr >= 1.0
        seconds = horizon * seconds_per_swap
        rows.append((horizon, treat, drift, snr, trustworthy, seconds))
        iqr = "[{:.2f}, {:.2f}]".format(treat["p25"], treat["p75"])
        if not trustworthy:
            verdict = "noise"
        elif seconds <= CASCADE_HORIZON_SECONDS:
            verdict = "CASCADE"
        else:
            verdict = "too slow"
        timelabel = (f"{seconds:.0f}s" if seconds < 90
                     else f"{seconds/60:.1f}m")
        print(f"{horizon:>8d}{timelabel:>9s}{treat['n']:>5d}"
              f"{treat['median']:>9.3f}{iqr:>18s}{drift:>9.3%}"
              f"{snr:>11.2f}{verdict:>12s}")

    if not rows:
        print("\nToo few measurable events -- widen the swap window or lower "
              "LARGE_QUANTILE.")
        sys.exit(1)

    usable = [r for r in rows if r[4]]
    cascade_rows = [r for r in usable if r[5] <= CASCADE_HORIZON_SECONDS]
    slow_rows = [r for r in usable if r[5] > CASCADE_HORIZON_SECONDS]

    print("\n=== Reading this ===")
    if cascade_rows:
        est = [r[1]["median"] for r in cascade_rows]
        print(f"  AT CASCADE SPEED (within ~{CASCADE_HORIZON_SECONDS:.0f}s, "
              f"the rows marked CASCADE): kappa reads between {min(est):.2f} "
              f"and {max(est):.2f}.")
        if max(est) < 0.15:
            print(f"  That is effectively ZERO. Over the seconds in which one "
                  f"wave of liquidations triggers the next, a large trade's "
                  f"price impact does not measurably revert.")
            print(f"  So the permanent-impact assumption -- kappa=0, which "
                  f"every earlier version of this model made and which looked "
                  f"like its most aggressive choice -- is approximately CORRECT "
                  f"for the timescale that matters. The kappa=0 row of "
                  f"critical_shock.py is the relevant one, not a worst case.")
    if slow_rows:
        est = [r[1]["median"] for r in slow_rows]
        print(f"  Over longer windows kappa rises to {max(est):.2f} -- impact "
              f"does revert, but on a timescale of minutes. Real, and "
              f"irrelevant to a cascade that has already finished by then. "
              f"Quoting those figures would materially understate risk.")
    if not usable:
        print("  NO horizon has a usable signal-to-noise ratio: background "
              "drift is larger than the dislocation being measured at every "
              "horizon tested. This sample cannot identify kappa.")

    noisy = [r for r in rows if not r[4]]
    if noisy:
        print(f"  Horizons {', '.join(str(r[0]) for r in noisy)} are dominated "
              f"by drift -- price wanders further over those windows than a "
              f"large trade moves it, so any apparent reversion there is the "
              f"market moving, not impact decaying. Ignore those rows however "
              f"tidy the numbers look.")

    if cascade_rows:
        centre = statistics.median([r[1]["median"] for r in cascade_rows])
        centre = max(0.0, min(1.0, centre))
        print(f"\n  Suggested kappa for the cascade model: ~{centre:.2f} "
              f"(from the CASCADE rows only), swept over 0 to ~0.25 for "
              f"sensitivity.")
        print(f"  Spread across individual events is wide -- see the IQR "
              f"column -- so this is a median, not a law. One pool, "
              f"{len(swaps):,} swaps, "
              f"{(max(timestamps) - min(timestamps))/3600:.0f} hours of a "
              f"calm market (range ${min(prices):,.0f}-${max(prices):,.0f}). "
              f"Re-run it across a volatile window before leaning on it.")
    else:
        print(f"\n  No kappa suggested. Sweep the full range in "
              f"critical_shock.py and report the grid.")

    print(f"  Ordinary large trades are not forced liquidations: liquidations "
          f"are publicly predictable and arrive together, so informed "
          f"counterparties can step back exactly when they are needed. Real "
          f"crisis kappa is likely LOWER than anything measured here, making "
          f"this an optimistic bound.")


if __name__ == "__main__":
    main()
