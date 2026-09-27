"""
One-shot check that every subgraph query this project uses still matches the
live schema -- run this before trusting any live result.

Several queries were added without being executable from the environment they
were written in (network egress there does not include gateway.thegraph.com),
so they are schema-plausible but unverified. This script exercises each one
and prints exactly what to fix if a field name has drifted.

    export GRAPH_API_KEY=...
    python3 verify_schema.py
"""

import os
import sys
import time
import traceback

import live_data as L
from live_data import require_data_access

# This script verifies QUERY SHAPES, not depth accuracy, so it fetches
# narrow tick windows. The analysis scripts use 15000, which for the 0.05%
# pool is ~2,500 initialised ticks over three pages per pool -- fine when you
# need the book, needlessly slow when you only need to know the field names
# still parse.
VERIFY_TICK_WINDOW = 1500

CHECKS = []
_POOL_CACHE = {}


def _pool(address, tick_window=VERIFY_TICK_WINDOW):
    """Fetch once per run. The fee-tier and router checks look at the same
    three pools, and fetching them twice doubled the runtime for nothing."""
    key = (address, tick_window)
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = L.fetch_uniswap_pool(address, tick_window=tick_window,
                                                verbose=False)
    return _POOL_CACHE[key]


def check(name, note=""):
    def wrap(fn):
        CHECKS.append((name, fn, note))
        return fn
    return wrap


@check("Uniswap: pool metadata", "POOL_META_QUERY")
def _pool_meta():
    d = L.query_subgraph(L.UNISWAP_V3_SUBGRAPH_ID, L.POOL_META_QUERY,
                         {"poolId": L.WETH_USDC_POOL.lower()})["pool"]
    return (f"tick={d['tick']} fee={d.get('feeTier')} "
            f"{d['token0']['symbol']}/{d['token1']['symbol']}")


@check("Uniswap: paginated ticks", "TICKS_QUERY -- cursor pagination")
def _ticks():
    pool = _pool(L.WETH_USDC_POOL)
    lo, hi = pool.collateral_price_bounds("WETH")
    now = pool.collateral_price("WETH")
    return (f"{len(pool.segments)} segments, WETH=${now:,.2f}, book covers "
            f"${lo:,.0f}..${hi:,.0f} ({(lo/now-1)*100:+.0f}%..{(hi/now-1)*100:+.0f}%), "
            f"truncated={pool.data_truncated}")


@check("Uniswap: all WETH fee tiers", "WETH_USDC_POOLS -- new addresses, unverified")
def _fee_tiers():
    out = []
    for tier, address in sorted(L.WETH_USDC_POOLS.items()):
        pool = _pool(address)
        price = pool.collateral_price("WETH")
        out.append(f"{tier}: ${price:,.0f}, {pool.fee_bps:.0f}bp fee, "
                   f"{len(pool.segments)} ticks")
    return " | ".join(out)


@check("Router: WETH across venues", "fetch_venue_router")
def _router():
    from routing import VenueRouter, LinearDepthVenue
    pools = [_pool(a) for _t, a in sorted(L.WETH_USDC_POOLS.items())]
    router = VenueRouter("WETH", pools=pools, linear_venues=[
        LinearDepthVenue("off-chain + unmodelled",
                         L.DEFAULT_OFFCHAIN_DEPTH_USD_PER_1PCT)])
    quote = router.quote_sell("WETH", 10_000.0)
    fills = ", ".join(f"{f.venue}={f.qty:,.0f}" for f in quote.fills)
    move = (quote.new_price / router.collateral_price("WETH") - 1) * 100
    # The books here are fetched at VERIFY_TICK_WINDOW, far shallower than
    # the analysis scripts use, so the split is not representative -- it is
    # only evidence that routing runs end to end on live data.
    return (f"10,000 WETH -> {move:+.2f}%; split on SHALLOW verify books "
            f"(not representative): {fills}")


@check("Uniswap: LST pool discovery", "POOLS_BY_PAIR_QUERY + fetch_lst_venues")
def _lst_pairs():
    found, missing = [], []
    for symbol in L.DEFAULT_LST_SYMBOLS:
        pools = L.find_pools_for_pair(symbol, "WETH", verbose=False)
        if pools:
            best = pools[0]
            found.append(f"{symbol}@{int(best['feeTier'])/10_000:.2f}%"
                         f"(${float(best['totalValueLockedUSD'])/1e6:,.0f}M)")
        else:
            missing.append(symbol)
    if not found:
        raise RuntimeError("no LST/WETH pools discovered -- the nested "
                           "token_ filter may not be supported by this "
                           "subgraph version")
    note = f"; no pool for {missing}" if missing else ""
    return ", ".join(found) + note


@check("Uniswap: recent swaps", "SWAPS_QUERY -- needed for kappa calibration")
def _swaps():
    swaps = L.fetch_recent_swaps(L.WETH_USDC_POOLS["0.05%"], max_swaps=1000,
                                 verbose=False)
    if not swaps:
        raise RuntimeError("no swaps returned")
    first, last = swaps[0], swaps[-1]
    hours = (int(last["timestamp"]) - int(first["timestamp"])) / 3600
    has_price = "sqrtPriceX96" in first and first["sqrtPriceX96"]
    return (f"{len(swaps)} swaps over {hours:,.1f}h; "
            f"sqrtPriceX96 present: {bool(has_price)}")


@check("Aave: aToken normalisation", "normalize_asset_symbol")
def _atokens():
    # sampling="scan" on purpose: this checks that aToken symbols are
    # normalised, which needs a couple of accounts, not a risk-weighted
    # sample. top_borrowers issues one query per market and would turn a
    # two-query check into dozens.
    positions, _prices, _cov = L.fetch_lending_positions_multi_with_coverage(
        L.AAVE_V3_SUBGRAPH_ID, "aave", first=200, verbose=False,
        sampling="scan")
    symbols = sorted({a for p in positions for a in p.collateral})
    leaked = [s for s in symbols if s.startswith("aEth")]
    if leaked:
        raise RuntimeError(f"aToken symbols survived normalisation: {leaked[:5]}")
    return f"{len(symbols)} distinct collateral symbols, none still aTokens"


@check("Aave: market totals", "MARKET_TOTALS_QUERY -- coverage denominator")
def _markets():
    markets, total = L.fetch_market_totals(L.AAVE_V3_SUBGRAPH_ID)
    top = sorted(markets.items(),
                 key=lambda kv: -float(kv[1].get("totalBorrowBalanceUSD") or 0))[:5]
    return (f"{len(markets)} markets, ${total:,.0f} total borrows; biggest: "
            + ", ".join(f"{k} ${float(v['totalBorrowBalanceUSD']):,.0f}" for k, v in top))


@check("Aave: top-borrower sample", "TOP_BORROWERS_QUERY + ACCOUNTS_BY_ID_QUERY")
def _top_borrowers():
    markets, _ = L.fetch_market_totals(L.AAVE_V3_SUBGRAPH_ID)
    # three markets is enough to prove the query shape; the analysis scripts
    # use twenty
    accounts = L.fetch_top_borrower_accounts(L.AAVE_V3_SUBGRAPH_ID, markets,
                                             per_market=25, verbose=False,
                                             max_markets=3)
    return (f"{len(accounts)} accounts hydrated from the top 3 markets "
            f"(analysis queries every market)")


@check("Aave: positions + coverage", "end-to-end fetch")
def _positions():
    positions, _coverage = L.fetch_lending_positions_with_coverage(
        L.AAVE_V3_SUBGRAPH_ID, "aave", first=200, verbose=False,
        sampling="scan")
    # Coverage is deliberately NOT reported here. This check walks accounts
    # by id to keep the query count down, so it lands on 200 arbitrary
    # accounts and the coverage figure is ~0% by construction. Printing it
    # would read like a collapse in sample quality when it is an artifact of
    # the check. The analysis scripts use risk-weighted sampling and report
    # real coverage; the check above exercises that path.
    return (f"{len(positions)} positions parsed from an arbitrary id-ordered "
            f"sample (coverage not meaningful here -- schema check only)")


@check("Compound: positions", "same query shape, never run live before")
def _compound():
    positions, _coverage = L.fetch_lending_positions_with_coverage(
        L.COMPOUND_V3_SUBGRAPH_ID, "compound", first=200, verbose=False,
        sampling="scan")
    return (f"{len(positions)} positions parsed from an arbitrary id-ordered "
            f"sample (coverage not meaningful here -- schema check only)")


def main():
    require_data_access()

    print("Verifying every live query against the current schema.")
    print("Schema shapes only -- narrow tick windows and small arbitrary "
          "samples, so the NUMBERS here are not results. Coverage, depth and "
          "routing splits come from the analysis scripts.\n")
    failures = []
    for name, fn, note in CHECKS:
        print(f"  {name:32s} ", end="", flush=True)
        started = time.time()
        try:
            result = fn()
            print(f"OK   [{time.time() - started:4.1f}s] {result}")
        except Exception as exc:
            print(f"FAIL {type(exc).__name__}: {str(exc)[:300]}")
            failures.append((name, note, traceback.format_exc()))

    print()
    if not failures:
        print("All queries verified against the live schema.")
        print("\nIf that was slow: the gateway's free tier rate-limits to "
              "roughly one query every 20 seconds under load. Nothing here can "
              "go faster than that -- record a snapshot once and replay it for "
              "everything except this check, which has to be live to mean "
              "anything.")
        return
    print(f"{len(failures)} check(s) failed. For each one, open the subgraph in "
          f"Graph Explorer, compare the 'Schema' tab against the query named "
          f"below, and the fix is usually a field rename:\n")
    for name, note, _ in failures:
        print(f"  - {name}: see {note} in live_data.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
