"""
Structural VaR / Expected Shortfall.

Standard VaR asks "what's the P&L loss we won't exceed at 95% confidence?"
using a purely statistical/historical model of returns. This ties that
statistical question to our STRUCTURAL cascade model instead: we still
estimate the shock size distribution from real historical volatility, but
instead of stopping at "a shock of this size is possible," we feed that
shock through the actual endogenous cascade engine to get "and here's what
our model predicts actually liquidates in dollar terms" -- a model-based
risk figure that accounts for cascade amplification, not just the raw
shock. This general approach (statistical shock distribution + structural
model of consequences) has real precedent in market-risk literature under
names like "stressed VaR" / scenario-based VaR.

METHODOLOGY, stated plainly:
  1. Pull ~90 days of real WETH/USDC daily prices (same poolDayData query
     as historical_backtest.py) and compute daily log returns.
  2. Estimate daily volatility (sample std dev of those returns).
  3. Estimate VaR and ES three ways, because the answer depends on the
     assumption more than on the data: closed-form NORMAL (kept only as a
     baseline to measure the others against), HISTORICAL (the empirical
     quantile -- assumes nothing, but cannot see past the worst day in the
     sample), and CORNISH-FISHER (the normal quantile corrected for the
     sample's own skew and excess kurtosis -- extrapolates, without
     pretending the distribution is normal).
  4. Run the cascade on the Cornish-Fisher figures, and print all three so
     the size of the normality error is visible rather than asserted.
  5. Run each shock size through the validated cascade engine on live
     current positions, and report the dollar value of collateral actually
     liquidated (from cumulative_collateral_sold), not just position counts.
"""

import datetime
import os
import math
import statistics
import sys
from typing import List

from live_data import query_subgraph, fetch_aave_positions, fetch_uniswap_pool, \
    fetch_venue_router, UNISWAP_V3_SUBGRAPH_ID, WETH_USDC_POOL, require_data_access
from cascade_sim import run_cascade, summarize
from liquidator import LiquidatorEconomics

POOL_DAY_DATA_QUERY = """
query PoolDays($poolId: String!, $start: Int!, $first: Int!) {
  poolDayDatas(
    where: { pool: $poolId, date_gte: $start }
    orderBy: date
    orderDirection: asc
    first: $first
  ) {
    date
    token0Price
  }
}
"""


def fetch_daily_returns(days: int = 90) -> List[float]:
    start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    data = query_subgraph(UNISWAP_V3_SUBGRAPH_ID, POOL_DAY_DATA_QUERY, {
        "poolId": WETH_USDC_POOL.lower(),
        "start": int(start.timestamp()),
        "first": days + 5,
    })
    days_data = data["poolDayDatas"]
    if len(days_data) < 10:
        raise RuntimeError(f"Only got {len(days_data)} days of price history -- "
                            f"too few to estimate volatility reliably (want 30+).")

    # token0Price empirically tracks WETH's USD-ish price on this pool (see
    # historical_backtest.py's note -- open/close on this deployment were
    # unreliable, token0Price was the trustworthy series)
    prices = [float(d["token0Price"]) for d in days_data]
    returns = []
    for p0, p1 in zip(prices, prices[1:]):
        if p0 > 0 and p1 > 0:
            returns.append(math.log(p1 / p0))
    return returns


def compute_var_es(daily_returns: List[float], confidence: float = 0.95):
    """
    Returns (var_shock_pct, es_shock_pct) -- both negative, e.g. -0.08 means
    an 8% down move. Uses the closed-form normal-distribution VaR/ES
    formulas via the standard library's statistics.NormalDist (no scipy
    needed).

    KEPT FOR COMPARISON, NOT FOR USE. Crypto returns are not normal, and the
    error is not symmetric -- it understates exactly the tail this project is
    about. `compare_var_methods` runs this alongside two estimators that do
    not assume normality, so the size of the error is visible rather than
    asserted.
    """
    mu = statistics.mean(daily_returns)
    sigma = statistics.stdev(daily_returns)
    dist = statistics.NormalDist(mu=0, sigma=1)  # standard normal for z-scores

    alpha = 1 - confidence
    z_alpha = dist.inv_cdf(alpha)          # e.g. -1.645 for 95%
    var_shock = mu + sigma * z_alpha       # VaR as a return (negative = loss)

    # closed-form Expected Shortfall for a normal distribution:
    # ES = mu - sigma * phi(z_alpha) / alpha
    phi_z = dist.pdf(z_alpha)
    es_shock = mu - sigma * phi_z / alpha

    return var_shock, es_shock, sigma


def sample_moments(daily_returns: List[float]):
    """Mean, stdev, skewness and EXCESS kurtosis of the sample.

    Excess kurtosis is reported (normal = 0) because the whole point is how
    far the sample departs from normal. Both use the plain moment estimators
    rather than the bias-corrected forms; with ~90 observations the
    difference is immaterial next to the sampling error on a fourth moment.
    """
    n = len(daily_returns)
    mu = statistics.mean(daily_returns)
    sigma = statistics.stdev(daily_returns)
    if sigma <= 0 or n < 4:
        return mu, sigma, 0.0, 0.0
    m3 = sum((r - mu) ** 3 for r in daily_returns) / n
    m4 = sum((r - mu) ** 4 for r in daily_returns) / n
    skew = m3 / sigma ** 3
    excess_kurtosis = m4 / sigma ** 4 - 3.0
    return mu, sigma, skew, excess_kurtosis


def historical_var_es(daily_returns: List[float], confidence: float = 0.95):
    """VaR and ES straight from the sample -- no distribution assumed.

    VaR is the empirical alpha-quantile; ES is the mean of everything at or
    below it. This makes no parametric assumption at all, which is its
    strength, and it can say nothing about a loss larger than the worst day
    observed, which is its weakness.

    Returns (var, es, tail_count) so the caller can see how many observations
    the estimate actually rests on. At 99% over 90 days that is ONE day, and
    a quantile estimated from one observation is an anecdote with a decimal
    point. The caller warns when the tail is this thin.
    """
    if not daily_returns:
        raise ValueError("no returns")
    alpha = 1 - confidence
    ordered = sorted(daily_returns)
    n = len(ordered)

    # index of the alpha-quantile, floored so the estimate sits inside the
    # observed tail rather than interpolating past it
    idx = max(0, min(n - 1, int(math.floor(alpha * n))))
    var = ordered[idx]
    tail = ordered[:idx + 1]
    es = statistics.mean(tail)
    return var, es, len(tail)


def cornish_fisher_var_es(daily_returns: List[float], confidence: float = 0.95,
                          steps: int = 400):
    """VaR and ES with the normal quantile corrected for skew and kurtosis.

    The Cornish-Fisher expansion adjusts the standard-normal quantile z using
    the sample's own third and fourth moments:

        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36

    with S the skewness and K the excess kurtosis. It keeps the closed-form
    convenience of parametric VaR while dropping the assumption that does the
    damage, and unlike the historical estimator it can extrapolate past the
    worst day in the sample.

    ES has no clean closed form here, so it is integrated numerically:
    ES = (1/alpha) * integral of VaR(p) dp over p in (0, alpha], by the
    trapezoid rule over `steps` points.

    The expansion is a local correction, not a distribution. It misbehaves for
    extreme moments -- the adjusted quantile can stop being monotone in p --
    so the caller should treat a CF figure that sits inside the historical one
    as a signal that the sample is too wild for the expansion, not as good
    news.
    """
    mu, sigma, skew, kurt = sample_moments(daily_returns)
    dist = statistics.NormalDist(mu=0, sigma=1)

    def cf_quantile(p: float) -> float:
        z = dist.inv_cdf(p)
        z_cf = (z
                + (z ** 2 - 1) * skew / 6.0
                + (z ** 3 - 3 * z) * kurt / 24.0
                - (2 * z ** 3 - 5 * z) * skew ** 2 / 36.0)
        return mu + sigma * z_cf

    alpha = 1 - confidence
    var = cf_quantile(alpha)

    total = 0.0
    for i in range(steps + 1):
        p = alpha * (i + 0.5) / (steps + 1)     # midpoints, avoids p = 0
        total += cf_quantile(p)
    es = total / (steps + 1)
    return var, es, skew, kurt


def compare_var_methods(daily_returns: List[float], confidence: float = 0.95):
    """All three estimators side by side, with the tail count for honesty."""
    n_var, n_es, sigma = compute_var_es(daily_returns, confidence)
    h_var, h_es, tail_n = historical_var_es(daily_returns, confidence)
    c_var, c_es, skew, kurt = cornish_fisher_var_es(daily_returns, confidence)
    return {
        "normal": (n_var, n_es),
        "historical": (h_var, h_es),
        "cornish_fisher": (c_var, c_es),
        "sigma": sigma,
        "skew": skew,
        "excess_kurtosis": kurt,
        "tail_observations": tail_n,
        "sample": len(daily_returns),
    }


def run_scenario(shock_pct: float, label: str):
    # Route across every fee tier plus off-chain depth, and make the
    # liquidator decide whether each liquidation pays.
    #
    # This used to sell everything into the single 0.3% pool with no
    # liquidator economics at all -- the free-liquidation model the rest of
    # the project abandoned. Those two assumptions push in opposite
    # directions (one pool overstates impact, free liquidation understates
    # bad debt), so the old figure was not conservative in either direction,
    # just inconsistent with every other number in the README.
    router = fetch_venue_router("WETH", verbose=False)
    economics = LiquidatorEconomics(gas_price_gwei=60.0)
    positions = [p for p in fetch_aave_positions(first=500, verbose=False)
                 if p.collateral_asset == "WETH"]
    starting_price = router.collateral_price("WETH")

    logs = run_cascade(positions, {"WETH": router},
                       initial_shock={"WETH": shock_pct},
                       max_rounds=100, verbose=False, economics=economics)
    final_price = router.collateral_price("WETH")
    summary = summarize(logs)
    weth_sold = summary.collateral_sold.get("WETH", 0.0)
    dollar_liquidated = weth_sold * starting_price  # valued at pre-shock price,
    # i.e. what the seized collateral was worth before the crash -- the more
    # standard convention for "size of the liquidation event" in $ terms

    print(f"{label}: shock {shock_pct:+.2%} -> realized "
          f"{(final_price/starting_price-1)*100:+.2f}%, "
          f"{summary.accounts_liquidated} accounts "
          f"({summary.liquidation_events} events), "
          f"~${dollar_liquidated:,.0f} in collateral liquidated")
    return dollar_liquidated


def main():
    require_data_access()

    print("Fetching ~90 days of real WETH/USDC price history...")
    try:
        returns = fetch_daily_returns(days=90)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)
    print(f"  {len(returns)} daily returns fetched")

    moments = compare_var_methods(returns, 0.95)
    print(f"  sample: {moments['sample']} daily returns, "
          f"volatility {moments['sigma']:.2%}, skew {moments['skew']:+.2f}, "
          f"excess kurtosis {moments['excess_kurtosis']:+.2f} "
          f"(a normal distribution has both at 0)")

    for confidence in (0.95, 0.99):
        m = compare_var_methods(returns, confidence)
        print(f"\n=== {confidence:.0%} confidence ===")
        print(f"  {'method':18s}{'VaR':>10s}{'ES':>10s}")
        print("  " + "-" * 38)
        for key, name in (("normal", "normal"),
                          ("historical", "historical"),
                          ("cornish_fisher", "Cornish-Fisher")):
            var_s, es_s = m[key]
            print(f"  {name:18s}{var_s:>9.2%}{es_s:>10.2%}")

        # A quantile estimated from a handful of observations is an anecdote
        # with a decimal point. Say so rather than printing it plain.
        if m["tail_observations"] < 5:
            print(f"  NB: the historical figures rest on "
                  f"{m['tail_observations']} observation(s) in the tail -- too "
                  f"few to estimate a quantile from. Read the Cornish-Fisher "
                  f"row instead, which extrapolates from the whole sample.")

        worst = min(v for v, _ in (m["normal"], m["historical"],
                                   m["cornish_fisher"]))
        normal_var = m["normal"][0]
        if normal_var > worst:
            gap = (worst - normal_var) / abs(normal_var) * 100
            print(f"  The normal assumption is the mildest of the three, by "
                  f"{abs(gap):.0f}% on VaR.")

        # The cascade is run on the Cornish-Fisher figures: they neither
        # assume normality nor stop at the worst day in a 90-day sample.
        cf_var, cf_es = m["cornish_fisher"]
        run_scenario(cf_var, "  VaR scenario  ")
        run_scenario(cf_es, "  ES scenario   ")

    print("\n--- Headline (95% confidence, Cornish-Fisher) ---")
    m = compare_var_methods(returns, 0.95)
    cf_var, _ = m["cornish_fisher"]
    var_dollars = run_scenario(cf_var, "VaR")
    print(f"\nAt 95% confidence, the model predicts ~${var_dollars:,.0f} in WETH "
          f"collateral would be liquidated on Aave given a 1-day move of this "
          f"size -- a structural VaR figure that accounts for cascade "
          f"amplification, not just the raw shock.")
    print("\nWhat this still assumes: that tomorrow's return is drawn from the "
          "same distribution as the last 90 days. Volatility clusters, so the "
          "day after a crash is not drawn from the calm-period distribution, "
          "and none of these three estimators models that. A GARCH-style "
          "conditional volatility model is the honest next step.")


if __name__ == "__main__":
    main()