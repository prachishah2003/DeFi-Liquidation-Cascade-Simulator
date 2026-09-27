"""
Live data ingestion layer.

Wires real on-chain data into the cascade simulator:
  - Aave V3 open positions  -> cascade_sim.Position objects
  - Uniswap v3 pool liquidity (real tick distribution) -> cascade_sim.Pool objects

SETUP
-----
1. Get a free Graph API key: thegraph.com/studio  (100k queries/month free tier)
2. export GRAPH_API_KEY=your_key_here

HONESTY NOTE (read before running)
-----------------------------------
This sandbox's network egress is locked to a small allowlist (pypi, github,
npm, etc.) and does NOT include gateway.thegraph.com, so I could not actually
execute a live call against these endpoints while writing this file. What I
verified instead:
  - The Uniswap v3 tick-walking math below (`build_pool_from_ticks`) against
    a synthetic response shaped exactly like the real subgraph's schema --
    see test_live_data.py. That logic is solid.
  - The subgraph IDs / endpoint format below, via web search against current
    Uniswap and Aave documentation.
  - The Aave query shape follows the long-standing public Aave subgraph
    schema (User -> UserReserve -> Reserve), which is stable, but subgraph
    schemas do occasionally shift between versions -- if a field name below
    errors out, open the subgraph in Graph Explorer (link below), check the
    "Schema" tab, and send me the actual field names; it's a one-line fix.

Uniswap v3 mainnet subgraph: 5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV
Aave V3 mainnet subgraph:    JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk
  (verify both at https://thegraph.com/explorer before relying on them --
  IDs occasionally migrate when protocols redeploy a subgraph)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

import snapshot
from uniswap_v3_math import TickSegment
from cascade_sim import Position, Pool

GATEWAY = "https://gateway.thegraph.com/api/{key}/subgraphs/id/{subgraph_id}"
SLOW_QUERY_SECONDS = 10.0
UNISWAP_V3_SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
AAVE_V3_SUBGRAPH_ID = "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk"
WETH_USDC_POOL = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"  # 0.3% fee tier, mainnet
WBTC_USDC_POOL = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35"  # 0.3% fee tier, mainnet

# Every Uniswap v3 fee tier for the same pair is a separate pool with its own
# book, and an aggregator quotes all of them. Modelling only the 0.3% tier
# understates available depth badly -- for WETH/USDC the 0.05% tier normally
# carries the most volume by a wide margin.
WETH_USDC_POOLS = {
    "0.05%": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    "0.30%": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
    "1.00%": "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387",
}
WBTC_USDC_POOLS = {
    "0.05%": "0x9a772018fbd77fcd2d25657e5c547baff3fd7d16",
    "0.30%": "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35",
}

# Notional that moves WETH spot roughly 1% across everything NOT modelled
# tick-by-tick here: centralised venues, market-maker inventory, on-chain
# pools without a fetched book. This is the single most consequential
# assumption in the whole model -- it is an order-of-magnitude figure, not a
# measurement, and the sweeps vary it deliberately. Set it to 0 to model
# on-chain-only execution and see how much difference it makes.
DEFAULT_OFFCHAIN_DEPTH_USD_PER_1PCT = 100_000_000.0


def _live_query_subgraph(subgraph_id: str, query: str,
                         variables: Optional[dict] = None) -> dict:
    api_key = os.environ.get("GRAPH_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GRAPH_API_KEY (free key from thegraph.com/studio) as an env var."
        )
    url = GATEWAY.format(key=api_key, subgraph_id=subgraph_id)
    started = time.time()
    resp = requests.post(url, json={"query": query, "variables": variables or {}},
                         timeout=(10, 120))
    elapsed = time.time() - started
    if elapsed > SLOW_QUERY_SECONDS:
        # The gateway's free tier throttles, sometimes to tens of seconds per
        # query. Saying so beats a script that looks hung.
        print(f"  [slow] a subgraph query took {elapsed:.0f}s -- the gateway is "
              f"throttling. Scripts that fetch many pages will crawl; record a "
              f"snapshot once (GRAPH_SNAPSHOT_MODE=record) and replay it.")
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(f"Subgraph returned errors: {payload['errors']}")
    return payload["data"]


# Every query in this project goes through here. With GRAPH_SNAPSHOT set it is
# recorded to, or replayed from, a file -- so a result can be reproduced
# exactly, offline, by anyone, without an API key. See snapshot.py.
query_subgraph = snapshot.install(_live_query_subgraph)


def require_data_access() -> None:
    """Exit with a useful message unless we can actually get data -- either a
    live API key, or a snapshot to replay."""
    if snapshot.active() is not None and snapshot.active().mode == snapshot.MODE_REPLAY:
        return
    if not os.environ.get("GRAPH_API_KEY"):
        print("Missing GRAPH_API_KEY.\n"
              "  Either export a key (free at thegraph.com/studio), or replay a\n"
              "  recorded snapshot offline:\n"
              "      GRAPH_SNAPSHOT=snapshots/<file>.json.gz python3 " +
              os.path.basename(__import__("sys").argv[0]))
        raise SystemExit(1)


def paginate(subgraph_id: str, query: str, variables: dict, result_key: str,
             page_size: int = 1000, max_records: Optional[int] = None,
             cursor_field: str = "id") -> List[dict]:
    """Page through a collection using a CURSOR, not `skip`.

    The Graph's gateway caps `skip` at 5000, so skip-based pagination
    silently stops there -- which is exactly how this project under-counted
    real liquidation volume on a busy day. Cursor pagination (`id_gt` on an
    ascending id sort) has no such ceiling.

    `query` must accept $first and $cursor and sort ascending by
    `cursor_field`. Returns every record, or the first `max_records`.
    """
    out: List[dict] = []
    cursor = ""
    while True:
        page_vars = dict(variables)
        page_vars["first"] = min(page_size, (max_records - len(out)) if max_records else page_size)
        page_vars["cursor"] = cursor
        if page_vars["first"] <= 0:
            break
        page = query_subgraph(subgraph_id, query, page_vars)[result_key]
        out.extend(page)
        if len(page) < page_vars["first"]:
            break              # short page == end of collection
        if max_records and len(out) >= max_records:
            break
        cursor = page[-1][cursor_field]
    return out


@dataclass
class CoverageReport:
    """How much of a protocol's actual risk this sample captured.

    Every dollar figure downstream is a fraction of the protocol's true
    exposure, and without this the fraction is unknown -- which made the
    project's headline numbers uninterpretable. The old code fetched 500
    accounts ordered by `openPositionCount` descending, i.e. the accounts
    with the MOST POSITIONS, which correlates with sophistication rather
    than with size or risk. A 500-account sample chosen that way is not a
    sample of the risk-bearing book.
    """
    protocol: str
    accounts_scanned: int = 0
    accounts_kept: int = 0
    captured_borrow_usd: float = 0.0
    protocol_borrow_usd: float = 0.0      # from Market.totalBorrowBalanceUSD
    skipped: Dict[str, int] = field(default_factory=dict)
    sampling: str = ""

    @property
    def coverage_pct(self) -> float:
        if self.protocol_borrow_usd <= 0:
            return float("nan")
        return 100.0 * self.captured_borrow_usd / self.protocol_borrow_usd

    def __str__(self) -> str:
        pct = self.coverage_pct
        pct_str = "unknown" if pct != pct else f"{pct:.1f}%"
        skips = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items())) or "none"
        return (f"  [coverage:{self.protocol}] sampling={self.sampling}; "
                f"scanned {self.accounts_scanned} accounts, kept {self.accounts_kept}; "
                f"captured ${self.captured_borrow_usd:,.0f} of "
                f"${self.protocol_borrow_usd:,.0f} total protocol borrows "
                f"({pct_str}). Skipped: {skips}.")


MARKET_TOTALS_QUERY = """
query Markets($first: Int!, $cursor: String!) {
  markets(first: $first, where: { id_gt: $cursor }, orderBy: id, orderDirection: asc) {
    id
    name
    totalBorrowBalanceUSD
    totalDepositBalanceUSD
    liquidationThreshold
    inputToken { symbol decimals }
  }
}
"""


#: Aave's liquidation bonus varies by reserve -- 5% on blue-chip collateral,
#: 7.5-10% on thinner assets -- and the model applied a flat 5% to all of it.
#: Fetched as its OWN query rather than by extending the positions query,
#: because snapshots are keyed on query text: adding a field there would
#: invalidate every recorded snapshot and make the whole offline replay
#: workflow unusable. A snapshot taken before this query existed simply
#: misses it, and `fetch_market_risk_params` falls back to the default.
MARKET_RISK_QUERY = """
query MarketRisk($first: Int!, $cursor: String!) {
  markets(first: $first, where: { id_gt: $cursor }, orderBy: id, orderDirection: asc) {
    id
    liquidationPenalty
    inputToken { symbol }
  }
}
"""

#: Aave v3's close factor is protocol-level, not per-market: 50% normally and
#: 100% once health factor drops below CLOSE_FACTOR_HF_THRESHOLD. There is no
#: per-market close factor to read, so only the bonus is fetched here.
DEFAULT_LIQUIDATION_BONUS = 0.05


def fetch_market_risk_params(subgraph_id: str, verbose: bool = True
                             ) -> Dict[str, float]:
    """Liquidation bonus per collateral symbol, as the protocol reports it.

    Returns {symbol: bonus_as_fraction}, e.g. {"WETH": 0.05, "LINK": 0.075}.
    Returns {} when the data is unavailable -- notably when replaying a
    snapshot recorded before this query existed -- so callers fall back to
    `DEFAULT_LIQUIDATION_BONUS` rather than inventing numbers.
    """
    try:
        markets = paginate(subgraph_id, MARKET_RISK_QUERY, {}, "markets",
                           page_size=500)
    except RuntimeError as exc:
        if "not in the snapshot" not in str(exc):
            raise
        if verbose:
            print("  [risk] this snapshot predates the per-market risk query, "
                  f"so every position uses the {DEFAULT_LIQUIDATION_BONUS:.0%} "
                  "default bonus. Re-record to pick up real values.")
        return {}

    bonuses: Dict[str, float] = {}
    for m in markets:
        symbol = (m.get("inputToken") or {}).get("symbol")
        raw = m.get("liquidationPenalty")
        if not symbol or raw is None:
            continue
        # Messari reports this as a percentage (5.0 meaning 5%). A zero is a
        # market that reports nothing useful, not a market with no bonus.
        penalty = float(raw) / 100.0
        if penalty <= 0:
            continue
        bonuses[normalize_asset_symbol(symbol)] = penalty

    if verbose and bonuses:
        lo, hi = min(bonuses.values()), max(bonuses.values())
        print(f"  [risk] liquidation bonus read for {len(bonuses)} markets "
              f"({lo:.2%} to {hi:.2%}); anything not listed uses "
              f"{DEFAULT_LIQUIDATION_BONUS:.0%}")
    return bonuses


def fetch_market_totals(subgraph_id: str) -> Tuple[Dict[str, dict], float]:
    """Every market plus the protocol's total borrows in USD -- the
    denominator for CoverageReport. Returns ({symbol: market}, total_usd)."""
    markets = paginate(subgraph_id, MARKET_TOTALS_QUERY, {}, "markets", page_size=500)
    by_symbol: Dict[str, dict] = {}
    total = 0.0
    for m in markets:
        borrows = float(m.get("totalBorrowBalanceUSD") or 0)
        total += borrows
        symbol = (m.get("inputToken") or {}).get("symbol")
        if symbol:
            prev = by_symbol.get(symbol)
            if prev is None or borrows > float(prev.get("totalBorrowBalanceUSD") or 0):
                by_symbol[symbol] = m
    return by_symbol, total


TOP_BORROWERS_QUERY = """
query TopBorrowers($marketId: String!, $first: Int!) {
  positions(
    where: { market: $marketId, side: BORROWER, balance_gt: "0" }
    orderBy: balance
    orderDirection: desc
    first: $first
  ) {
    account { id }
  }
}
"""

ACCOUNTS_BY_ID_QUERY = """
query AccountsById($ids: [String!]!) {
  accounts(where: { id_in: $ids }, first: 1000) {
    id
    positions(where: { balance_gt: "0" }) {
      side
      isCollateral
      balance
      asset { symbol decimals }
      market { liquidationThreshold inputTokenPriceUSD }
    }
  }
}
"""


# One query per market, and Aave has ~67 of them. Borrows are heavily
# concentrated, so capping this to the largest markets cuts the fan-out
# sharply -- useful when the gateway is throttling, where 67 queries at 60s
# each is over an hour.
#
# The DEFAULT IS None (every market) on purpose. Capping it lowers the
# sample's coverage, and coverage is quoted alongside every result this
# project produces, so changing the default would silently change published
# figures. Pass max_markets explicitly when you want the speed and can
# accept the narrower sample; the coverage report will show what it cost.
DEFAULT_MAX_MARKETS = None


def fetch_top_borrower_accounts(subgraph_id: str, markets: Dict[str, dict],
                                per_market: int = 200, verbose: bool = True,
                                max_markets: int = DEFAULT_MAX_MARKETS
                                ) -> List[dict]:
    """Risk-weighted sample: the largest borrowers in each market.

    Balance is only comparable WITHIN a market (it's raw token units), which
    is why this samples per-market and unions the results rather than trying
    to rank all borrowers globally in one query. The union is biased toward
    large positions -- which is the correct bias for a liquidation-risk
    model, and is stated rather than hidden.
    """
    ranked = sorted(markets.items(),
                    key=lambda kv: -float(kv[1].get("totalBorrowBalanceUSD") or 0))
    borrows_all = sum(float(m.get("totalBorrowBalanceUSD") or 0)
                      for _s, m in ranked)
    selected = ranked[:max_markets] if max_markets else ranked
    borrows_kept = sum(float(m.get("totalBorrowBalanceUSD") or 0)
                       for _s, m in selected)
    if verbose and len(selected) < len(ranked):
        share = 100.0 * borrows_kept / borrows_all if borrows_all else 0.0
        print(f"  [top_borrowers] querying the {len(selected)} largest of "
              f"{len(ranked)} markets ({share:.1f}% of protocol borrows); "
              f"raise max_markets to include the tail")

    account_ids = set()
    for symbol, market in selected:
        if float(market.get("totalBorrowBalanceUSD") or 0) <= 0:
            continue
        try:
            rows = query_subgraph(subgraph_id, TOP_BORROWERS_QUERY, {
                "marketId": market["id"], "first": per_market,
            })["positions"]
        except RuntimeError as exc:
            if verbose:
                print(f"  [top_borrowers] market {symbol} query failed ({exc}) -- skipping")
            continue
        for row in rows:
            acct = (row.get("account") or {}).get("id")
            if acct:
                account_ids.add(acct)

    ids = sorted(account_ids)
    accounts: List[dict] = []
    for i in range(0, len(ids), 500):          # id_in has a practical size limit
        batch = ids[i:i + 500]
        accounts.extend(query_subgraph(subgraph_id, ACCOUNTS_BY_ID_QUERY,
                                       {"ids": batch})["accounts"])
    if verbose:
        print(f"  [top_borrowers] {len(markets)} markets -> {len(ids)} distinct "
              f"borrower accounts -> {len(accounts)} hydrated")
    return accounts


# ---------------------------------------------------------------------------
# Uniswap v3: real pool liquidity -> Pool object
# ---------------------------------------------------------------------------

POOL_META_QUERY = """
query PoolMeta($poolId: String!) {
  pool(id: $poolId) {
    tick
    sqrtPrice
    liquidity
    feeTier
    totalValueLockedToken0
    totalValueLockedToken1
    totalValueLockedUSD
    token0 { symbol decimals }
    token1 { symbol decimals }
  }
}
"""

# Ticks are paginated by CURSOR on tickIdx. The previous version asked for
# `first: 1000` with no pagination, so any pool/window needing more than
# 1,000 initialized ticks was silently truncated -- and a truncated book
# looks exactly like a pool that ran out of liquidity. That is precisely the
# ambiguity behind the project's "cascades have a real ceiling" claim, so the
# fetch now either reaches the edge of the requested window or reports that
# it stopped early. Never both silently.
TICKS_QUERY = """
query PoolTicks($poolId: String!, $tickFloor: BigInt!, $tickCeil: BigInt!,
                $first: Int!, $cursor: BigInt!) {
  ticks(
    where: {
      poolAddress: $poolId
      tickIdx_gte: $tickFloor
      tickIdx_lte: $tickCeil
      tickIdx_gt: $cursor
    }
    orderBy: tickIdx
    orderDirection: asc
    first: $first
  ) {
    tickIdx
    liquidityNet
  }
}
"""


MAX_TICK_PAGES = 40          # 40 x 1000 ticks is far more than any real window


def fetch_uniswap_pool(pool_address: str, tick_window: int = 6000,
                       verbose: bool = True) -> Pool:
    """
    tick_window: how far above/below the current tick to pull liquidity for
    (6000 ticks ~= a wide double-digit % price range for most fee tiers --
    widen it if a big simulated shock runs off the edge of the fetched data).

    Ticks are fetched with cursor pagination until the window is exhausted,
    so the resulting Pool covers the whole requested range rather than the
    first 1,000 initialized ticks. If the page cap is somehow hit anyway, the
    Pool is flagged `data_truncated` and every swap that walks off the end
    reports `stop_reason == "data_exhausted"` instead of masquerading as a
    real liquidity boundary.
    """
    meta = query_subgraph(UNISWAP_V3_SUBGRAPH_ID, POOL_META_QUERY,
                          {"poolId": pool_address.lower()})["pool"]
    current_tick = int(meta["tick"])
    tick_floor = current_tick - tick_window
    tick_ceil = current_tick + tick_window

    ticks: List[dict] = []
    cursor = tick_floor - 1
    truncated = False
    for page_num in range(MAX_TICK_PAGES):
        page = query_subgraph(UNISWAP_V3_SUBGRAPH_ID, TICKS_QUERY, {
            "poolId": pool_address.lower(),
            "tickFloor": str(tick_floor),
            "tickCeil": str(tick_ceil),
            "first": 1000,
            "cursor": str(cursor),
        })["ticks"]
        ticks.extend(page)
        if len(page) < 1000:
            break
        cursor = int(page[-1]["tickIdx"])
    else:
        truncated = True
        if verbose:
            print(f"  [fetch_uniswap_pool] WARNING: hit the {MAX_TICK_PAGES}-page "
                  f"cap with {len(ticks)} ticks -- the book is truncated and any "
                  f"'ran dry' result from this pool is a DATA limit, not a "
                  f"liquidity limit.")

    if verbose:
        fee = meta.get("feeTier")
        print(f"  [fetch_uniswap_pool] {len(ticks)} initialized ticks across "
              f"+/-{tick_window} ticks around tick {current_tick}"
              + (f" (fee tier {int(fee)/10_000:.2f}%)" if fee else ""))

    return build_pool_from_ticks(meta, ticks, data_truncated=truncated,
                                 tick_window=tick_window)


def build_pool_from_ticks(pool_data: dict, ticks_data: List[dict],
                          data_truncated: bool = False,
                          tick_window: Optional[int] = None) -> Pool:
    """
    Pure data-transformation logic (no network) -- this is the part covered
    by test_live_data.py's synthetic-response test.

    Walks outward from the pool's current tick using each initialized tick's
    liquidityNet to reconstruct the *actual* active-liquidity distribution,
    exactly like the Uniswap v3 SDK does client-side.
    """
    # subgraphs serialize BigInt fields (like decimals) as strings in JSON --
    # cast explicitly rather than assuming they're already ints
    decimals0 = int(pool_data["token0"]["decimals"])
    decimals1 = int(pool_data["token1"]["decimals"])
    decimal_adj = 10 ** (decimals0 - decimals1)

    current_tick = int(pool_data["tick"])

    # Liquidity from the subgraph is in RAW (undecimaled) units -- L relates
    # to the *raw* token reserves, not human-readable ones. Since price
    # above is decimal-adjusted to human terms, liquidity must be too, or
    # amount_in (in human WETH/USDC units) becomes negligible against a
    # liquidity value that's still ~10^12-10^18x too large -- which is
    # exactly what silently zeroed out all price impact in the first version
    # of this code. Derivation: L_human = L_raw / 10^((decimals0+decimals1)/2).
    liquidity_adj = 10 ** ((decimals0 + decimals1) / 2)
    current_liquidity = float(pool_data["liquidity"]) / liquidity_adj

    def tick_to_sqrt_p(tick: int) -> float:
        raw_price = 1.0001 ** tick
        return (raw_price * decimal_adj) ** 0.5

    ticks_sorted = sorted(ticks_data, key=lambda t: int(t["tickIdx"]))
    above = [t for t in ticks_sorted if int(t["tickIdx"]) > current_tick]
    below = [t for t in ticks_sorted if int(t["tickIdx"]) < current_tick][::-1]  # descending

    segments: List[TickSegment] = []

    # walk downward (price falling) -- this is the direction liquidation sells move
    liq = current_liquidity
    lower_bound_tick = current_tick
    for t in below:
        tick_idx = int(t["tickIdx"])
        segments.append(TickSegment(
            sqrt_p_lower=tick_to_sqrt_p(tick_idx),
            sqrt_p_upper=tick_to_sqrt_p(lower_bound_tick),
            liquidity=liq,
        ))
        liq -= float(t["liquidityNet"]) / liquidity_adj  # same adjustment applies here
        liq = max(liq, 0.0)
        lower_bound_tick = tick_idx

    # walk upward (price rising) -- included for completeness / two-sided use
    liq = current_liquidity
    upper_bound_tick = current_tick
    for t in above:
        tick_idx = int(t["tickIdx"])
        segments.append(TickSegment(
            sqrt_p_lower=tick_to_sqrt_p(upper_bound_tick),
            sqrt_p_upper=tick_to_sqrt_p(tick_idx),
            liquidity=liq,
        ))
        liq += float(t["liquidityNet"]) / liquidity_adj
        upper_bound_tick = tick_idx

    # NOTE: token0/token1 here are Uniswap's on-chain ordering (by ascending
    # contract address), which is NOT necessarily "collateral, numeraire" --
    # e.g. the USDC/WETH 0.3% pool has token0=USDC, token1=WETH. Pool's
    # collateral_price()/sell() methods handle either ordering correctly,
    # so nothing downstream needs to assume WETH is token0.
    # feeTier is in hundredths of a basis point: 500 -> 5bp, 3000 -> 30bp,
    # 10000 -> 100bp. Fall back to 30bp only if the field is absent.
    fee_raw = pool_data.get("feeTier")
    fee_bps = float(fee_raw) / 100.0 if fee_raw else 30.0

    def _maybe_float(key):
        raw = pool_data.get(key)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    return Pool(
        token0_symbol=pool_data["token0"]["symbol"],
        token1_symbol=pool_data["token1"]["symbol"],
        sqrt_p=tick_to_sqrt_p(current_tick),
        segments=segments,
        data_truncated=data_truncated,
        tick_window=tick_window,
        fee_bps=fee_bps,
        tvl_token0=_maybe_float("totalValueLockedToken0"),
        tvl_token1=_maybe_float("totalValueLockedToken1"),
        tvl_usd=_maybe_float("totalValueLockedUSD"),
    )


# Individual swaps, newest first. Used to CALIBRATE impact decay: how much of
# the price dislocation a large trade causes is given back afterwards.
# `sqrtPriceX96` is the pool price immediately AFTER each swap, which is what
# makes the measurement possible without reconstructing state.
SWAPS_QUERY = """
query PoolSwaps($poolId: String!, $before: BigInt!, $first: Int!) {
  swaps(
    where: { pool: $poolId, timestamp_lt: $before }
    orderBy: timestamp
    orderDirection: desc
    first: $first
  ) {
    id
    timestamp
    amount0
    amount1
    amountUSD
    sqrtPriceX96
    tick
  }
}
"""


def fetch_recent_swaps(pool_address: str, max_swaps: int = 6000,
                       page_size: int = 1000, verbose: bool = True) -> List[dict]:
    """The most recent `max_swaps` swaps on a pool, oldest-first.

    Paginated on a descending timestamp cursor rather than `skip`, for the
    same reason as everywhere else in this file: the gateway caps skip at
    5000 and silently truncates past it.
    """
    out: List[dict] = []
    before = 1 << 62                      # effectively "now"
    while len(out) < max_swaps:
        page = query_subgraph(UNISWAP_V3_SUBGRAPH_ID, SWAPS_QUERY, {
            "poolId": pool_address.lower(),
            "before": str(before),
            "first": min(page_size, max_swaps - len(out)),
        })["swaps"]
        if not page:
            break
        out.extend(page)
        oldest = min(int(sw["timestamp"]) for sw in page)
        if oldest >= before:
            break                          # no progress; avoid an infinite loop
        before = oldest
        if len(page) < page_size:
            break
    out.sort(key=lambda sw: (int(sw["timestamp"]), sw["id"]))
    if verbose and out:
        span_hours = (int(out[-1]["timestamp"]) - int(out[0]["timestamp"])) / 3600
        print(f"  [fetch_recent_swaps] {len(out)} swaps spanning "
              f"{span_hours:,.1f} hours")
    return out


# Liquid staking tokens trade against WETH, not against dollars, so selling
# seized wstETH is two hops. Rather than hardcode pool addresses -- which is
# how a project ends up quietly pointed at a dead pool after a redeploy --
# these are discovered by token symbol and ranked by TVL.
LST_INTERMEDIATE = "WETH"
# A reconstructed book should never imply paying out more of the other token
# than the pool actually holds. Above this multiple, say so loudly.
BOOK_CONSISTENCY_LIMIT = 1.5
DEFAULT_LST_SYMBOLS = ("wstETH", "weETH", "cbETH", "rETH", "osETH", "rsETH", "ETHx")

POOLS_BY_PAIR_QUERY = """
query PoolsForPair($symbols: [String!]!, $first: Int!) {
  pools(
    where: { token0_: { symbol_in: $symbols }, token1_: { symbol_in: $symbols } }
    orderBy: totalValueLockedUSD
    orderDirection: desc
    first: $first
  ) {
    id
    feeTier
    totalValueLockedUSD
    token0 { symbol }
    token1 { symbol }
  }
}
"""


def find_pools_for_pair(symbol_a: str, symbol_b: str, limit: int = 5,
                        verbose: bool = True) -> List[dict]:
    """Pools trading `symbol_a` against `symbol_b`, deepest first.

    Filters on both tokens at once with `symbol_in`, so either on-chain
    ordering matches in a single query -- Uniswap orders tokens by address,
    which is not something worth hardcoding per pair.
    """
    rows = query_subgraph(UNISWAP_V3_SUBGRAPH_ID, POOLS_BY_PAIR_QUERY, {
        "symbols": [symbol_a, symbol_b], "first": limit,
    })["pools"]
    wanted = {symbol_a, symbol_b}
    matched = [r for r in rows
               if {r["token0"]["symbol"], r["token1"]["symbol"]} == wanted]
    if verbose and matched:
        best = matched[0]
        print(f"  [pairs] {symbol_a}/{symbol_b}: {len(matched)} pool(s); deepest "
              f"is the {int(best['feeTier'])/10_000:.2f}% tier with "
              f"${float(best['totalValueLockedUSD']):,.0f} TVL")
    elif verbose:
        print(f"  [pairs] {symbol_a}/{symbol_b}: no pool found")
    return matched


def fetch_lst_venues(weth_router, symbols=DEFAULT_LST_SYMBOLS,
                     tick_window: int = 6000, min_tvl_usd: float = 1_000_000.0,
                     verbose: bool = True) -> Dict[str, object]:
    """Two-hop venues for every LST with a real market against WETH.

    `weth_router` is passed in, not built here, and the SAME object is shared
    by every venue on purpose: an LST liquidation ends with WETH being sold
    for dollars in the same books a WETH liquidation uses. Sharing the router
    is what carries that contagion.

    Assets with no pool, or a pool too thin to mean anything, are skipped and
    named -- they fall back to static pricing, which is the old behaviour and
    understates their selling pressure.
    """
    from routing import TwoHopVenue

    venues: Dict[str, object] = {}
    skipped = []
    for symbol in symbols:
        try:
            candidates = find_pools_for_pair(symbol, LST_INTERMEDIATE,
                                             verbose=False)
        except Exception as exc:
            skipped.append(f"{symbol} (query failed: {exc})")
            continue
        candidates = [c for c in candidates
                      if float(c.get("totalValueLockedUSD") or 0) >= min_tvl_usd]
        if not candidates:
            skipped.append(f"{symbol} (no pool above "
                           f"${min_tvl_usd/1e6:,.0f}M TVL)")
            continue

        best = candidates[0]
        pool = fetch_uniswap_pool(best["id"], tick_window=tick_window,
                                  verbose=False)
        venues[symbol] = TwoHopVenue(symbol, pool, weth_router,
                                     intermediate=LST_INTERMEDIATE)
        if verbose:
            ratio = pool.collateral_price(symbol)
            print(f"  [lst] {symbol}: {int(best['feeTier'])/10_000:.2f}% pool, "
                  f"${float(best['totalValueLockedUSD'])/1e6:,.0f}M TVL, "
                  f"{ratio:.4f} {LST_INTERMEDIATE} each, "
                  f"absorbs {pool.sell_capacity(symbol):,.0f} {symbol}")
        check = pool.book_consistency(symbol)
        if check is not None and verbose:
            implied, held, ratio_ = check
            if ratio_ > BOOK_CONSISTENCY_LIMIT:
                print(f"  [lst] WARNING: {symbol}'s reconstructed book implies "
                      f"it could pay out {implied:,.0f} {LST_INTERMEDIATE}, but "
                      f"the pool only holds {held:,.0f} ({ratio_:.1f}x). The "
                      f"tick reconstruction is overstating depth -- treat this "
                      f"venue's slippage as a LOWER bound, and do not quote "
                      f"its capacity.")

    if verbose and skipped:
        print(f"  [lst] no venue for {skipped} -- these keep static prices, "
              f"which understates their selling pressure")
    return venues


def fetch_venue_router(symbol: str = "WETH",
                       pool_addresses: Optional[Dict[str, str]] = None,
                       offchain_depth_usd_per_1pct: float =
                       DEFAULT_OFFCHAIN_DEPTH_USD_PER_1PCT,
                       tick_window: int = 15000, verbose: bool = True):
    """Build a router over every fee tier for `symbol`, plus off-chain depth.

    This is what should be passed to `run_cascade` instead of a single Pool.
    Selling an entire protocol's liquidations into one 0.3% pool is not a
    conservative assumption -- it is an incorrect one, and it inflates
    cascade amplification severalfold.
    """
    from routing import VenueRouter, LinearDepthVenue

    if pool_addresses is None:
        pool_addresses = WETH_USDC_POOLS if symbol == "WETH" else WBTC_USDC_POOLS

    pools = []
    for tier, address in pool_addresses.items():
        try:
            pool = fetch_uniswap_pool(address, tick_window=tick_window,
                                      verbose=False)
        except Exception as exc:
            if verbose:
                print(f"  [router] {symbol} {tier} tier unavailable ({exc}) -- skipped")
            continue
        capacity = pool.sell_capacity(symbol)
        price = pool.collateral_price(symbol)
        if verbose:
            print(f"  [router] {symbol} {tier} tier ({pool.fee_bps:.0f}bp fee): "
                  f"{len(pool.segments)} segments, absorbs up to "
                  f"{capacity:,.0f} {symbol} "
                  f"(~${capacity * price / 1e6:,.0f}M) before its book runs out")
        pools.append(pool)

    if not pools:
        raise RuntimeError(f"no {symbol} pools could be fetched -- cannot route")

    linear = []
    if offchain_depth_usd_per_1pct > 0:
        linear.append(LinearDepthVenue("off-chain + unmodelled",
                                       offchain_depth_usd_per_1pct))
    router = VenueRouter(symbol, pools=pools, linear_venues=linear)
    if verbose:
        print(f"  {router.describe()}")
    return router


# ---------------------------------------------------------------------------
# Generic Messari-standard lending protocol fetch. Aave V3 and Compound V3
# both publish subgraphs under the same `messari` namespace with the same
# schema shape (accounts -> positions, side=BORROWER/COLLATERAL), confirmed
# via schema introspection on Aave -- Compound reuses the same query rather
# than guessing a new one, though it hasn't been run against Compound's
# live API from here (same caveat as every other new query this session).
# ---------------------------------------------------------------------------

AAVE_V3_SUBGRAPH_ID = "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk"      # validated all session
COMPOUND_V3_SUBGRAPH_ID = "AwoxEZbiWLvv6e3QdvdMZw4WDURdGbvPfHmZRc8Dpfz9"  # found via search, UNVERIFIED live

# NOTE ON SAMPLING (this used to be the biggest silent problem in the file)
# -------------------------------------------------------------------------
# The original query was `accounts(first: 500, orderBy: openPositionCount,
# orderDirection: desc)` -- the 500 accounts with the MOST POSITIONS. That
# correlates with how sophisticated an account is, not with how much debt it
# carries or how close it is to liquidation, and there was no pagination, so
# every dollar figure the project produced was an unknown fraction of an
# unrepresentative slice. Two replacements, both honest about their bias:
#
#   sampling="top_borrowers" (default) -- the largest borrowers in each
#       market, unioned. Biased toward large positions, which is the right
#       bias for a liquidation-risk model and is reported, not hidden.
#   sampling="scan" -- cursor-paginated walk over accounts by id. Unbiased
#       with respect to size, but truncated by max_accounts, so on a
#       protocol with millions of accounts it is a small arbitrary slice.
#
# Either way `CoverageReport` states what fraction of the protocol's actual
# borrows the sample captured, so the reader can size the caveat themselves.
LENDING_ACCOUNTS_QUERY = """
query OpenAccounts($first: Int!, $cursor: String!) {
  accounts(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { openPositionCount_gt: 0, id_gt: $cursor }
  ) {
    id
    positions(where: { balance_gt: "0" }) {
      side
      isCollateral
      balance
      asset { symbol decimals }
      market { liquidationThreshold inputTokenPriceUSD }
    }
  }
}
"""

STABLECOINS = {"USDC", "USDT", "DAI"}
DEFAULT_SAMPLING = "top_borrowers"

# Aave reports collateral in aTokens -- interest-bearing receipts for the
# underlying, named per deployment (aEthWETH on Ethereum, aArbWETH on
# Arbitrum, ...). They are NOT separate assets: aEthWETH is WETH, and the
# subgraph prices it identically.
#
# Left unnormalised this quietly splits a book in two. A WETH-collateralised
# account can arrive as either `WETH` or `aEthWETH`, so:
#   - single-asset code filtering `collateral_asset == "WETH"` silently
#     dropped every aToken position (this is most of why the WETH sample was
#     295 accounts rather than the full book);
#   - the factor model assigned aEthwstETH a beta of zero -- treating staked
#     ether as uncorrelated with ether;
#   - an account holding both forms looked diversified when it holds one asset.
ATOKEN_PREFIXES = ("aEth", "aArb", "aOpt", "aPol", "aAva", "aBas", "aGno",
                   "aSep", "aScr", "aLin", "aZkS", "aMet", "aBnb")


def normalize_asset_symbol(symbol: str) -> str:
    """aEthWETH -> WETH. Leaves anything unrecognised alone."""
    for prefix in ATOKEN_PREFIXES:
        if symbol.startswith(prefix) and len(symbol) > len(prefix):
            remainder = symbol[len(prefix):]
            if remainder[:1].isalpha():
                return remainder
    return symbol

# HOW TO COLLAPSE A MULTI-COLLATERAL ACCOUNT INTO ONE SINGLE-ASSET POSITION
# ------------------------------------------------------------------------
# This is the setting that produced the single worst bug in the project.
#
# "legacy" kept an account's ENTIRE debt but only its LARGEST collateral
# asset. A real whale holding $10.7M WETH + $12.8M wstETH + $7.7M WBTC
# against $20M of USDC has a true health factor of 1.25 -- comfortably
# safe. Collapsed the legacy way it becomes $12.8M of collateral against
# $20M of debt: health factor 0.51, instantly and falsely insolvent, and
# worth $7.2M of bad debt that does not exist. Nothing caught it while the
# model liquidated everything for free, because a fake liquidation and a
# real one looked identical. It only became visible once liquidations could
# be refused and bad debt could accrue.
#
# Worse, it interacts with sampling: top-borrower sampling deliberately
# selects the biggest accounts, which are exactly the diversified ones this
# collapse mangles most. Better sampling made the bug bite harder.
#
#   "rescale" (default) -- keep the dominant collateral asset, and scale the
#       debt so the position's health factor equals the account's TRUE
#       blended health factor. Preserves both the account population and the
#       solvency condition (bad debt still occurs exactly when the real
#       account would be insolvent). Distorts WHICH asset gets sold: all
#       price impact is attributed to the dominant collateral, so cross-asset
#       selling pressure is missed. That is what multi-asset mode is for.
#   "strict"  -- keep only accounts that genuinely have one collateral asset
#       and one debt asset. Nothing is synthesised, but most of the book is
#       discarded; the dropped count is reported.
#   "legacy"  -- the original broken behaviour, kept ONLY so the bug can be
#       reproduced. Emits a warning.
# Below this, a position's collateral is dust left behind by a completed
# liquidation rather than a live position.
DUST_COLLATERAL_USD = 100.0

DEFAULT_COLLAPSE = "rescale"
VALID_COLLAPSE = ("rescale", "strict", "legacy")


def _get_accounts(subgraph_id: str, protocol_name: str, sampling: str,
                  max_accounts: int, per_market: int, verbose: bool,
                  max_markets: int = DEFAULT_MAX_MARKETS
                  ) -> Tuple[List[dict], CoverageReport, Dict[str, dict]]:
    """Acquire the account sample and open a coverage report for it."""
    coverage = CoverageReport(protocol=protocol_name, sampling=sampling)
    try:
        markets, protocol_borrows = fetch_market_totals(subgraph_id)
        coverage.protocol_borrow_usd = protocol_borrows
    except Exception as exc:
        if verbose:
            print(f"  [coverage:{protocol_name}] market totals unavailable ({exc}); "
                  f"coverage %% will be reported as unknown")
        markets = {}

    if sampling == "top_borrowers" and markets:
        accounts = fetch_top_borrower_accounts(subgraph_id, markets,
                                               per_market=per_market,
                                               verbose=verbose,
                                               max_markets=max_markets)
    else:
        if sampling == "top_borrowers" and verbose:
            print(f"  [sampling:{protocol_name}] no markets returned -- falling "
                  f"back to an id-ordered scan, which is NOT risk-weighted")
            coverage.sampling = "scan (fallback)"
        accounts = paginate(subgraph_id, LENDING_ACCOUNTS_QUERY, {}, "accounts",
                            page_size=min(1000, max_accounts),
                            max_records=max_accounts)
    coverage.accounts_scanned = len(accounts)
    return accounts, coverage, markets


def fetch_lending_positions(subgraph_id: str, protocol_name: str, first: int = 500,
                             verbose: bool = True, sampling: str = DEFAULT_SAMPLING,
                             per_market: int = 200,
                             collapse: str = DEFAULT_COLLAPSE) -> List[Position]:
    """Back-compat wrapper -- returns positions only. Prefer
    fetch_lending_positions_with_coverage() so the coverage figure travels
    with the data instead of only being printed."""
    positions, _ = fetch_lending_positions_with_coverage(
        subgraph_id, protocol_name, first, verbose, sampling, per_market, collapse)
    return positions


def fetch_lending_positions_with_coverage(
        subgraph_id: str, protocol_name: str, first: int = 500,
        verbose: bool = True, sampling: str = DEFAULT_SAMPLING,
        per_market: int = 200, collapse: str = DEFAULT_COLLAPSE
        ) -> Tuple[List[Position], CoverageReport]:
    """
    Generic single-asset fetch (collapses each account to its dominant
    collateral + dominant stablecoin debt) -- works against any Messari-
    standard lending subgraph. fetch_aave_positions/fetch_compound_positions
    below are thin wrappers over this for backward-compat call sites.

    Simplifications for v1 (extend once the pipeline is validated):
      - Collapses each account to its single largest collateral position and
        single largest borrow position, rather than modeling full multi-
        collateral/multi-debt accounts (see fetch_lending_positions_multi).
      - Only keeps accounts whose dominant debt is a stablecoin (USDC/USDT/
        DAI), since health_factor() assumes debt is priced 1:1 in the
        numeraire.

    ASSUMPTION FLAGGED: `market.liquidationThreshold` is treated as a
    percentage (e.g. 80.0 means 80%). If health factors come out ~100x off
    from sane values, this assumption is inverted.
    """
    if collapse not in VALID_COLLAPSE:
        raise ValueError(f"collapse must be one of {VALID_COLLAPSE}, got {collapse!r}")
    if collapse == "legacy" and verbose:
        print("  [WARNING] collapse='legacy' reproduces a known bug: it keeps an "
              "account's full debt against only its largest collateral asset, "
              "making diversified accounts look insolvent. For reproduction only.")

    accounts, coverage, _markets = _get_accounts(
        subgraph_id, protocol_name, sampling, first, per_market, verbose)
    positions: List[Position] = []
    risk_bonuses = fetch_market_risk_params(subgraph_id, verbose=verbose)
    no_pair = 0
    non_stable_debt = 0
    multi_collateral_dropped = 0
    scanned_borrow_usd = 0.0

    def scaled(raw: str, decimals) -> float:
        return float(raw) / (10 ** int(decimals))

    def usd_value(p: dict) -> float:
        qty = scaled(p["balance"], p["asset"]["decimals"])
        price = float(p["market"]["inputTokenPriceUSD"] or 0)
        return qty * price

    def symbol_of(p: dict) -> str:
        return normalize_asset_symbol(p["asset"]["symbol"])

    for account in accounts:
        borrow_positions = [p for p in account["positions"] if p["side"] == "BORROWER"]
        collateral_positions = [p for p in account["positions"] if p["side"] == "COLLATERAL"]
        if not borrow_positions or not collateral_positions:
            no_pair += 1
            continue

        scanned_borrow_usd += sum(usd_value(p) for p in borrow_positions)

        debt = max(borrow_positions, key=usd_value)
        if symbol_of(debt) not in STABLECOINS:
            non_stable_debt += 1
            continue

        coll = max(collateral_positions, key=usd_value)

        # --- the collapse, done honestly (see DEFAULT_COLLAPSE above) ------
        distinct_collateral = {c["asset"]["symbol"] for c in collateral_positions}
        distinct_debt = {b["asset"]["symbol"] for b in borrow_positions}
        if collapse == "strict" and (len(distinct_collateral) > 1
                                      or len(distinct_debt) > 1):
            multi_collateral_dropped += 1
            continue

        threshold = float(coll["market"]["liquidationThreshold"]) / 100.0
        if threshold <= 0:
            # Seen in practice on Compound V3: its "Comet" architecture has
            # one base market per deployment with collateral as side assets,
            # rather than Aave's one-market-per-asset pattern -- the generic
            # Messari `market.liquidationThreshold` field isn't reliably
            # populated for Compound's collateral entries as a result. A
            # threshold of exactly 0 zeroes out health_factor()'s numerator
            # regardless of position size, producing a fake "totally
            # wrecked" account -- treating it as missing data and skipping
            # is safer than quietly cascading on a wrong number. Proper fix
            # needs digging into where Compound V3's real collateral factor
            # lives in this schema -- flagged, not fixed, for now.
            no_pair += 1
            continue

        coll_qty = scaled(coll["balance"], coll["asset"]["decimals"])
        debt_qty = scaled(debt["balance"], debt["asset"]["decimals"])
        if coll_qty <= 0 or debt_qty <= 0:
            continue

        if collapse == "rescale":
            # Preserve the account's TRUE blended health factor rather than
            # inheriting a fake one from dropping most of its collateral.
            weighted_collateral = 0.0
            for c in collateral_positions:
                thr = float(c["market"]["liquidationThreshold"] or 0) / 100.0
                if thr <= 0:
                    continue
                weighted_collateral += usd_value(c) * thr
            total_debt_value = sum(usd_value(b) for b in borrow_positions)
            if weighted_collateral <= 0 or total_debt_value <= 0:
                no_pair += 1
                continue
            true_hf = weighted_collateral / total_debt_value

            coll_price = float(coll["market"]["inputTokenPriceUSD"] or 0)
            if coll_price <= 0:
                no_pair += 1
                continue
            # debt such that (coll_qty * price * threshold) / debt == true_hf
            debt_qty = coll_qty * coll_price * threshold / true_hf
            if debt_qty <= 0:
                continue

        coll_symbol = symbol_of(coll)
        positions.append(Position(
            position_id=f"{protocol_name[:2]}-{account['id'][:8]}",
            protocol=protocol_name,
            collateral_asset=coll_symbol,
            collateral_qty=coll_qty,
            debt_asset=symbol_of(debt),
            debt_qty=debt_qty,
            liquidation_threshold=threshold,
            liquidation_bonus=risk_bonuses.get(coll_symbol,
                                               DEFAULT_LIQUIDATION_BONUS),
        ))
        coverage.captured_borrow_usd += usd_value(debt)

    coverage.accounts_kept = len(positions)
    coverage.skipped = {"no_borrow_collateral_pair": no_pair,
                        "non_stablecoin_debt": non_stable_debt}
    if collapse == "strict":
        coverage.skipped["multi_asset_account"] = multi_collateral_dropped
    coverage.sampling = f"{coverage.sampling}, collapse={collapse}"

    if verbose:
        print(f"  [fetch_lending_positions:{protocol_name}] {len(accounts)} accounts returned, "
              f"{no_pair} had no borrow+collateral pair, "
              f"{non_stable_debt} skipped for non-stablecoin debt, "
              f"{len(positions)} kept")
        print(coverage)
        if scanned_borrow_usd > 0:
            kept_share = 100.0 * coverage.captured_borrow_usd / scanned_borrow_usd
            print(f"  [coverage:{protocol_name}] the single-asset filters kept "
                  f"{kept_share:.1f}% of the borrow value present in the sample "
                  f"(the rest is non-stablecoin debt this mode cannot price 1:1)")
        if no_pair == len(accounts) and accounts:
            print(f"  [debug] 100% failure rate on {protocol_name} -- dumping raw "
                  f"positions for the first 2 accounts to inspect actual field values:")
            for acct in accounts[:2]:
                print(f"    account {acct['id'][:10]}:")
                for p in acct["positions"][:5]:
                    print(f"      side={p['side']!r} isCollateral={p['isCollateral']!r} "
                          f"balance={p['balance']!r} asset={p['asset']['symbol']!r}")
                if not acct["positions"]:
                    print("      (positions list is empty)")

    return positions, coverage


def fetch_lending_positions_multi(subgraph_id: str, protocol_name: str, first: int = 500,
                                   verbose: bool = True, sampling: str = DEFAULT_SAMPLING,
                                   per_market: int = 200):
    """Back-compat wrapper -- returns (positions, static_prices)."""
    positions, static_prices, _ = fetch_lending_positions_multi_with_coverage(
        subgraph_id, protocol_name, first, verbose, sampling, per_market)
    return positions, static_prices


def fetch_lending_positions_multi_with_coverage(
        subgraph_id: str, protocol_name: str, first: int = 500,
        verbose: bool = True, sampling: str = DEFAULT_SAMPLING,
        per_market: int = 200):
    """
    Generic multi-asset fetch (keeps every collateral/debt position per
    account) -- works against any Messari-standard lending subgraph.
    Returns (positions, static_prices).
    """
    from multi_asset import MultiPosition  # local import -- avoids a hard
    # dependency from live_data.py on multi_asset.py for single-asset-only callers

    accounts, coverage, _markets = _get_accounts(
        subgraph_id, protocol_name, sampling, first, per_market, verbose)
    positions = []
    static_prices: Dict[str, float] = {}
    risk_bonuses = fetch_market_risk_params(subgraph_id, verbose=verbose)
    bonus_from_live = 0
    bonus_from_default = 0
    no_pair = 0
    dust_collateral = 0
    unpriced_assets_seen = set()
    normalized_symbols = set()

    def scaled(raw: str, decimals) -> float:
        return float(raw) / (10 ** int(decimals))

    for account in accounts:
        borrow_positions = [p for p in account["positions"] if p["side"] == "BORROWER"]
        collateral_positions = [p for p in account["positions"] if p["side"] == "COLLATERAL"]
        if not borrow_positions or not collateral_positions:
            no_pair += 1
            continue

        collateral: Dict[str, float] = {}
        collateral_thresholds: Dict[str, float] = {}
        for p in collateral_positions:
            raw_symbol = p["asset"]["symbol"]
            symbol = normalize_asset_symbol(raw_symbol)
            if symbol != raw_symbol:
                normalized_symbols.add(f"{raw_symbol}->{symbol}")
            qty = scaled(p["balance"], p["asset"]["decimals"])
            if qty <= 0:
                continue
            threshold = float(p["market"]["liquidationThreshold"]) / 100.0
            if threshold <= 0:
                # same issue as fetch_lending_positions -- a zero threshold
                # zeroes out this asset's contribution to the HF numerator
                # regardless of size. Here we can afford to just drop this
                # ONE collateral entry rather than the whole account, since
                # multi-asset accounts often have other, valid collateral.
                continue
            collateral[symbol] = collateral.get(symbol, 0.0) + qty
            collateral_thresholds[symbol] = threshold
            price = float(p["market"]["inputTokenPriceUSD"] or 0)
            if price > 0:
                static_prices[symbol] = price

        debt: Dict[str, float] = {}
        for p in borrow_positions:
            raw_symbol = p["asset"]["symbol"]
            symbol = normalize_asset_symbol(raw_symbol)
            if symbol != raw_symbol:
                normalized_symbols.add(f"{raw_symbol}->{symbol}")
            qty = scaled(p["balance"], p["asset"]["decimals"])
            if qty <= 0:
                continue
            debt[symbol] = debt.get(symbol, 0.0) + qty
            price = float(p["market"]["inputTokenPriceUSD"] or 0)
            if price > 0:
                static_prices[symbol] = price
                coverage.captured_borrow_usd += qty * price
            else:
                unpriced_assets_seen.add(symbol)

        if not collateral or not debt:
            continue

        # Positions whose collateral has already been seized down to dust but
        # whose debt record lingers are closed, not distressed. Counting them
        # as live accounts at health factor zero manufactures both phantom
        # insolvency and a meaningless "already underwater" headline.
        collateral_value = sum(qty * static_prices.get(a, 0.0)
                               for a, qty in collateral.items())
        if collateral_value < DUST_COLLATERAL_USD:
            dust_collateral += 1
            continue

        # The seize is pro-rata across every collateral asset, so the bonus
        # that applies is the value-weighted blend of those markets' bonuses
        # -- not any single asset's. Assets with no published bonus fall back
        # to the default and are counted so the fallback stays visible.
        blend_num = 0.0
        for a, qty in collateral.items():
            value = qty * static_prices.get(a, 0.0)
            if value <= 0:
                continue
            blend_num += value * risk_bonuses.get(a, DEFAULT_LIQUIDATION_BONUS)
            if a in risk_bonuses:
                bonus_from_live += 1
            else:
                bonus_from_default += 1
        blended = (blend_num / collateral_value) if collateral_value > 0 \
            else DEFAULT_LIQUIDATION_BONUS

        positions.append(MultiPosition(
            position_id=f"{protocol_name[:2]}-{account['id'][:8]}",
            protocol=protocol_name,
            collateral=collateral,
            collateral_thresholds=collateral_thresholds,
            debt=debt,
            liquidation_bonus=blended,
        ))

    coverage.accounts_kept = len(positions)
    coverage.skipped = {"no_borrow_collateral_pair": no_pair}

    if verbose:
        total_legs = bonus_from_live + bonus_from_default
        if total_legs:
            print(f"  [risk] liquidation bonus: {bonus_from_live} of "
                  f"{total_legs} collateral legs used the market's own value, "
                  f"{bonus_from_default} fell back to "
                  f"{DEFAULT_LIQUIDATION_BONUS:.0%}")

    if verbose:
        n_multi = sum(1 for p in positions if len(p.collateral) > 1 or len(p.debt) > 1)
        print(f"  [fetch_lending_positions_multi:{protocol_name}] {len(accounts)} accounts "
              f"returned, {no_pair} had no borrow+collateral pair, {len(positions)} kept "
              f"({n_multi} genuinely multi-asset)")
        print(coverage)
        if dust_collateral:
            print(f"  [fetch_lending_positions_multi:{protocol_name}] "
                  f"{dust_collateral} account(s) dropped as already-closed "
                  f"(collateral below ${DUST_COLLATERAL_USD:,.0f} against a "
                  f"lingering debt record)")
        if normalized_symbols:
            sample = sorted(normalized_symbols)[:6]
            print(f"  [fetch_lending_positions_multi:{protocol_name}] "
                  f"normalised {len(normalized_symbols)} aToken symbol(s) to "
                  f"their underlying, e.g. {sample}")
        if unpriced_assets_seen:
            print(f"  [coverage:{protocol_name}] assets with no price from the "
                  f"subgraph: {sorted(unpriced_assets_seen)} -- positions holding "
                  f"these must be filtered with multi_asset.drop_unpriced() "
                  f"before running a cascade; they are NOT silently valued now.")

    return positions, static_prices, coverage


# ---------------------------------------------------------------------------
# Protocol-specific thin wrappers -- kept so existing call sites
# (demo_cascade.py, historical_backtest.py, shock_sweep.py, dashboard.py)
# don't need to change.
# ---------------------------------------------------------------------------

def fetch_aave_positions(first: int = 500, verbose: bool = True,
                          sampling: str = DEFAULT_SAMPLING) -> List[Position]:
    return fetch_lending_positions(AAVE_V3_SUBGRAPH_ID, "aave", first, verbose, sampling)


def fetch_aave_positions_multi(first: int = 500, verbose: bool = True,
                                sampling: str = DEFAULT_SAMPLING):
    return fetch_lending_positions_multi(AAVE_V3_SUBGRAPH_ID, "aave", first, verbose, sampling)


def fetch_compound_positions(first: int = 500, verbose: bool = True,
                              sampling: str = DEFAULT_SAMPLING) -> List[Position]:
    return fetch_lending_positions(COMPOUND_V3_SUBGRAPH_ID, "compound", first, verbose, sampling)


def fetch_compound_positions_multi(first: int = 500, verbose: bool = True,
                                    sampling: str = DEFAULT_SAMPLING):
    return fetch_lending_positions_multi(COMPOUND_V3_SUBGRAPH_ID, "compound", first, verbose, sampling)