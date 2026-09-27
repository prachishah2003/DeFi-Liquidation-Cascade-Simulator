"""
Backtest against a real event: the January 31, 2026 crash. ETH dropped
~12% in 24h on Fed-Chair-nomination fears; Aave's own blog reports the
protocol processed over $140M in liquidations that single day
(https://aave.com/blog/historical-liquidations).

METHODOLOGY / SCOPE (read this before trusting the numbers)
-------------------------------------------------------------
This does NOT reconstruct the exact Jan 30 pre-crash position snapshot --
that needs block-pinned historical queries (`block: {number: N}`), which
I haven't verified work against this subgraph deployment, and given how
many rounds of schema debugging the *current-state* queries already took,
I didn't want to gamble another round on an unverified technique.

Instead, this is a two-part comparison:
  1. REAL ground truth, fetched live: the actual price move that day
     (Uniswap's poolDayData) and the actual liquidation count/volume
     (Aave's `liquidates` entity, filtered by timestamp -- no block
     numbers needed, just a date range).
  2. OUR MODEL's prediction: take TODAY's live positions and pool, apply
     a shock of the SAME PERCENTAGE MAGNITUDE as the real Jan 31 drop,
     and see what cascade our model predicts.

That's "does our mechanism produce a plausible magnitude of self-
amplification," not "did we replay history exactly." Documented as a
known limitation -- extending to a true block-pinned replay is a
reasonable next step if you want more rigor for the writeup.
"""

import datetime
import os
import sys

from live_data import query_subgraph, UNISWAP_V3_SUBGRAPH_ID, AAVE_V3_SUBGRAPH_ID, \
    fetch_uniswap_pool, fetch_lending_positions_with_coverage, paginate, \
    WETH_USDC_POOL, require_data_access
from cascade_sim import run_cascade, summarize


def _utc(ts) -> str:
    return datetime.datetime.fromtimestamp(
        int(ts), datetime.timezone.utc).strftime("%Y-%m-%d")

EVENT_START = datetime.datetime(2026, 1, 31, 0, 0, tzinfo=datetime.timezone.utc)
EVENT_END = datetime.datetime(2026, 2, 1, 0, 0, tzinfo=datetime.timezone.utc)
BUFFER_START = datetime.datetime(2026, 1, 30, 0, 0, tzinfo=datetime.timezone.utc)
BUFFER_END = datetime.datetime(2026, 2, 2, 0, 0, tzinfo=datetime.timezone.utc)

POOL_DAY_DATA_QUERY = """
query PoolDays($poolId: String!, $start: Int!, $end: Int!) {
  poolDayDatas(
    where: { pool: $poolId, date_gte: $start, date_lte: $end }
    orderBy: date
    orderDirection: asc
  ) {
    date
    open
    high
    low
    close
    token0Price
    token1Price
  }
}
"""

# Cursor-paginated on `id`, NOT `skip`. The gateway caps skip at 5000, so the
# previous version silently stopped counting there -- which is the most likely
# reason the fetched total came in under Aave's own reported figure for the day.
AAVE_LIQUIDATES_QUERY = """
query LiquidationsInWindow($start: Int!, $end: Int!, $first: Int!, $cursor: String!) {
  liquidates(
    where: { timestamp_gte: $start, timestamp_lte: $end, id_gt: $cursor }
    orderBy: id
    orderDirection: asc
    first: $first
  ) {
    id
    timestamp
    amount
    amountUSD
    asset { symbol }
  }
}
"""


def fetch_real_price_move():
    data = query_subgraph(UNISWAP_V3_SUBGRAPH_ID, POOL_DAY_DATA_QUERY, {
        "poolId": WETH_USDC_POOL.lower(),
        "start": int(BUFFER_START.timestamp()),
        "end": int(BUFFER_END.timestamp()),
    })
    days = data["poolDayDatas"]
    if not days:
        print("No poolDayData returned for this window -- either the query needs "
              "adjusting or this pool didn't exist/wasn't indexed that far back.")
        return None, []
    print("  Daily WETH/USDC prices around the event:")
    for d in days:
        date_str = _utc(d["date"])
        print(f"    {date_str}: open={d['open']} close={d['close']} "
              f"token0Price={d['token0Price']} token1Price={d['token1Price']}")

    # NOTE: open/close on this subgraph deployment show open==close on every
    # row in practice -- unreliable, don't use them. token0Price gives a
    # clean day-over-day trajectory that empirically matches known WETH
    # price levels, so that's the series used for the real move below.
    real_moves = []
    if len(days) >= 2:
        print("\n  Real day-over-day moves (from token0Price, NOT open/close -- see note above):")
        for prev, curr in zip(days, days[1:]):
            p0, p1 = float(prev["token0Price"]), float(curr["token0Price"])
            pct = (p1 - p0) / p0 * 100
            d0, d1 = _utc(prev["date"]), _utc(curr["date"])
            print(f"    {d0} -> {d1}: ${p0:,.2f} -> ${p1:,.2f}  ({pct:+.2f}%)")
            real_moves.append((d0, d1, pct))
    return days, real_moves


def fetch_real_liquidations(page_size: int = 1000):
    """Every Aave liquidation in the event window, via cursor pagination.

    There is no page cap here on purpose: an arbitrary cap is how the
    previous version under-counted a busy day and then compared the short
    total against Aave's full-day figure as though the gap were meaningful."""
    all_liquidations = paginate(
        AAVE_V3_SUBGRAPH_ID, AAVE_LIQUIDATES_QUERY,
        {"start": int(EVENT_START.timestamp()), "end": int(EVENT_END.timestamp())},
        "liquidates", page_size=page_size)

    total_usd = sum(float(l["amountUSD"] or 0) for l in all_liquidations)
    by_asset = {}
    for l in all_liquidations:
        symbol = (l.get("asset") or {}).get("symbol", "?")
        by_asset[symbol] = by_asset.get(symbol, 0.0) + float(l["amountUSD"] or 0)
    print(f"  {len(all_liquidations)} liquidation events fetched (cursor-paginated, "
          f"no page cap), totaling ${total_usd:,.0f}")
    top = sorted(by_asset.items(), key=lambda kv: -kv[1])[:6]
    print("  by collateral asset: " + ", ".join(f"{k} ${v/1e6:,.1f}M" for k, v in top))
    weth_usd = by_asset.get("WETH", 0.0)
    print(f"  WETH alone: ${weth_usd:,.0f} -- this is the like-for-like "
          f"denominator for a WETH-only model run, NOT the all-asset total")
    return all_liquidations, total_usd, weth_usd


def run_model_with_equivalent_shock(real_shock_pct: float):
    print(f"\nRunning our model on TODAY's live data with a {real_shock_pct:+.1%} "
          f"shock (matching the real Jan 31 magnitude)...")
    pool = fetch_uniswap_pool(WETH_USDC_POOL, tick_window=15000)
    all_positions, coverage = fetch_lending_positions_with_coverage(
        AAVE_V3_SUBGRAPH_ID, "aave", first=500)
    positions = [p for p in all_positions if p.collateral_asset == "WETH"]
    pools = {"WETH": pool}
    price_now = pool.collateral_price("WETH")
    logs = run_cascade(positions, pools, initial_shock={"WETH": real_shock_pct}, max_rounds=15)
    final_price = pool.collateral_price("WETH")
    summary = summarize(logs)

    print(f"\nModel prediction: {summary}")
    print(f"  price ${price_now:,.2f} -> ${final_price:,.2f} "
          f"({(final_price/price_now - 1)*100:.2f}% total move)")
    return summary, coverage, final_price, price_now


if __name__ == "__main__":
    require_data_access()

    print("=== Step 1: real observed price move (Jan 30 - Feb 2, 2026) ===")
    days, real_moves = fetch_real_price_move()

    print("\n=== Step 2: real Aave liquidations on Jan 31, 2026 ===")
    _, real_total_usd, real_weth_usd = fetch_real_liquidations()

    print("\n=== Step 3: our model's prediction, same-magnitude shock ===")
    # Use the actual measured Jan 30->31 move (from token0Price) rather than
    # the press-reported "~12%" headline figure, now that we have it.
    jan31_move_pct = next((pct for d0, d1, pct in real_moves if d1 == "2026-01-31"), -12.0)
    print(f"  Using the measured Jan 30->31 move: {jan31_move_pct:+.2f}%")
    summary, coverage, final_price, price_now = run_model_with_equivalent_shock(
        real_shock_pct=jan31_move_pct / 100)

    # ------------------------------------------------------------------
    # The actual like-for-like comparison. Two separate claims, kept apart
    # on purpose, because conflating them is what made the old headline
    # ("the model reconstructed $122.6M vs Aave's $140M") misleading: that
    # number came from Step 2 -- the DATA FETCH -- and said nothing about
    # whether the cascade model predicts anything.
    # ------------------------------------------------------------------
    print("\n=== Compare ===")
    print("\nCLAIM 1 -- is the data pipeline sound? (a fetch check, not a model check)")
    print(f"  Liquidations this pipeline independently reconstructed for the day: "
          f"${real_total_usd:,.0f}")
    print(f"  Aave's own published figure for the same day:                     ~$140,000,000")
    print(f"  Ratio: {real_total_usd / 140e6:.2f}x. This tests the subgraph query "
          f"and pagination ONLY.")

    print("\nCLAIM 2 -- does the cascade model predict the right magnitude?")
    model_usd = summary.collateral_seized_usd
    cov = coverage.coverage_pct
    print(f"  Model's predicted WETH collateral liquidated:  ${model_usd:,.0f}")
    print(f"  Real WETH collateral liquidated that day:      ${real_weth_usd:,.0f}")
    if real_weth_usd > 0:
        print(f"  Raw ratio (model / real): {model_usd / real_weth_usd:.2f}x")
    if cov == cov and cov > 0:
        scaled = model_usd * 100.0 / cov
        print(f"  The model saw only {cov:.1f}% of Aave's borrows, so scaling its "
              f"figure to full-protocol terms: ${scaled:,.0f}")
        if real_weth_usd > 0:
            print(f"  Coverage-scaled ratio (model / real): {scaled / real_weth_usd:.2f}x")
    else:
        print("  Coverage unknown for this run -- the raw ratio above is a "
              "fraction of the protocol compared against the whole of it, so "
              "it is a LOWER BOUND on the model's figure, not an estimate.")

    print("\nWhat this does and does not establish:")
    print("  - It compares the model run on TODAY's positions against a shock of "
          "the same magnitude as Jan 31's. It is NOT a block-pinned replay of "
          "that day's actual book, so a close ratio is encouraging, not proof.")
    print("  - A single event is an anecdote. Running this across several "
          "historical crashes would give an error distribution, which is the "
          "figure worth quoting.")