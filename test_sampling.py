"""
Tests for cursor pagination, the risk-weighted account sample, and the
coverage report -- all with a stubbed subgraph, so this runs with no API key
and no network.

Why these exist: the project used to fetch 500 accounts ordered by
`openPositionCount` descending with no pagination, then report dollar
figures derived from them. That sample is biased toward accounts with many
positions (sophistication) rather than large ones (risk), and nothing told
the reader what fraction of the protocol it represented. These tests pin
down the replacement: cursor pagination that can't hit the gateway's 5000
`skip` ceiling, a per-market top-borrower sample, and a coverage figure that
travels with the data.
"""

import live_data
from live_data import (paginate, CoverageReport, fetch_market_totals,
                       fetch_top_borrower_accounts,
                       fetch_lending_positions_with_coverage)


class StubSubgraph:
    """Records every query it serves so tests can assert on call shape."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, subgraph_id, query, variables=None):
        variables = variables or {}
        self.calls.append((query, dict(variables)))
        return self.handler(query, variables)


def install(handler):
    stub = StubSubgraph(handler)
    live_data.query_subgraph = stub
    return stub


# ---------------------------------------------------------------------------
# paginate
# ---------------------------------------------------------------------------

def test_paginate_walks_every_page_by_cursor():
    """2,500 records over pages of 1,000 -- and crucially past the 5,000-record
    point where `skip` pagination silently dies on the gateway."""
    total = 12_000
    records = [{"id": f"{i:06d}"} for i in range(total)]

    def handler(query, v):
        cursor, first = v["cursor"], v["first"]
        start = 0 if not cursor else next(
            i for i, r in enumerate(records) if r["id"] == cursor) + 1
        return {"things": records[start:start + first]}

    install(handler)
    got = paginate("sub", "q", {}, "things", page_size=1000)
    assert len(got) == total, f"expected {total}, got {len(got)}"
    assert got[0]["id"] == "000000" and got[-1]["id"] == f"{total-1:06d}"
    print(f"  paginated {len(got):,} records across pages of 1,000 "
          f"(skip-based pagination would have stopped at 5,000)")


def test_paginate_respects_max_records():
    records = [{"id": f"{i:06d}"} for i in range(5000)]

    def handler(query, v):
        cursor, first = v["cursor"], v["first"]
        start = 0 if not cursor else int(cursor) + 1
        return {"things": records[start:start + first]}

    install(handler)
    got = paginate("sub", "q", {}, "things", page_size=400, max_records=1000)
    assert len(got) == 1000, len(got)
    print(f"  max_records honoured exactly ({len(got)})")


def test_paginate_stops_on_short_page():
    def handler(query, v):
        return {"things": [{"id": "a"}, {"id": "b"}]}   # always short

    stub = install(handler)
    got = paginate("sub", "q", {}, "things", page_size=100)
    assert len(got) == 2 and len(stub.calls) == 1, (len(got), len(stub.calls))
    print("  short page ends the walk after one request")


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_pct_and_unknown_denominator():
    c = CoverageReport(protocol="aave", captured_borrow_usd=250e6,
                       protocol_borrow_usd=1e9, sampling="top_borrowers")
    assert abs(c.coverage_pct - 25.0) < 1e-9
    assert "25.0%" in str(c)
    blind = CoverageReport(protocol="aave", captured_borrow_usd=1.0)
    assert blind.coverage_pct != blind.coverage_pct      # NaN
    assert "unknown" in str(blind)
    print(f"  {c}")


def test_market_totals_sum_protocol_borrows():
    markets = [
        {"id": "m-weth", "name": "WETH", "totalBorrowBalanceUSD": "400000000",
         "totalDepositBalanceUSD": "9000000000", "liquidationThreshold": "80",
         "inputToken": {"symbol": "WETH", "decimals": "18"}},
        {"id": "m-usdc", "name": "USDC", "totalBorrowBalanceUSD": "600000000",
         "totalDepositBalanceUSD": "3000000000", "liquidationThreshold": "85",
         "inputToken": {"symbol": "USDC", "decimals": "6"}},
    ]
    served = {"done": False}

    def handler(query, v):
        if served["done"]:
            return {"markets": []}
        served["done"] = True
        return {"markets": markets}

    install(handler)
    by_symbol, total = fetch_market_totals("sub")
    assert total == 1_000_000_000.0, total
    assert set(by_symbol) == {"WETH", "USDC"}
    print(f"  protocol borrows ${total:,.0f} across {len(by_symbol)} markets")


# ---------------------------------------------------------------------------
# risk-weighted sampling
# ---------------------------------------------------------------------------

def _account(acct_id, coll_symbol, coll_qty, coll_dec, debt_symbol, debt_qty,
             debt_dec, threshold="80", coll_price="2800", debt_price="1"):
    return {
        "id": acct_id,
        "positions": [
            {"side": "COLLATERAL", "isCollateral": True,
             "balance": str(int(coll_qty * 10 ** coll_dec)),
             "asset": {"symbol": coll_symbol, "decimals": str(coll_dec)},
             "market": {"liquidationThreshold": threshold,
                        "inputTokenPriceUSD": coll_price}},
            {"side": "BORROWER", "isCollateral": False,
             "balance": str(int(debt_qty * 10 ** debt_dec)),
             "asset": {"symbol": debt_symbol, "decimals": str(debt_dec)},
             "market": {"liquidationThreshold": threshold,
                        "inputTokenPriceUSD": debt_price}},
        ],
    }


def _lending_stub(accounts, protocol_borrows=1_000_000_000.0):
    markets_served = {"done": False}

    def handler(query, v):
        if "markets(" in query:
            if markets_served["done"]:
                return {"markets": []}
            markets_served["done"] = True
            return {"markets": [{
                "id": "m-usdc", "name": "USDC",
                "totalBorrowBalanceUSD": str(protocol_borrows),
                "totalDepositBalanceUSD": "0", "liquidationThreshold": "85",
                "inputToken": {"symbol": "USDC", "decimals": "6"}}]}
        if "positions(" in query and "orderBy: balance" in query:
            return {"positions": [{"account": {"id": a["id"]}} for a in accounts]}
        if "id_in" in query:
            wanted = set(v["ids"])
            return {"accounts": [a for a in accounts if a["id"] in wanted]}
        return {"accounts": accounts}

    return handler


def test_top_borrower_sample_hydrates_accounts():
    accounts = [_account(f"0xacct{i:02d}", "WETH", 100 + i, 18, "USDC",
                         100_000 + i, 6) for i in range(5)]
    install(_lending_stub(accounts))
    markets, _ = fetch_market_totals("sub")
    got = fetch_top_borrower_accounts("sub", markets, per_market=50, verbose=False)
    assert len(got) == 5, len(got)
    assert {a["id"] for a in got} == {a["id"] for a in accounts}
    print(f"  {len(got)} borrower accounts hydrated from the per-market top list")


def test_fetch_reports_coverage_against_protocol_borrows():
    accounts = [_account(f"0xacct{i:02d}", "WETH", 100, 18, "USDC",
                         1_000_000, 6) for i in range(4)]     # $4M borrowed
    install(_lending_stub(accounts, protocol_borrows=40_000_000.0))

    positions, coverage = fetch_lending_positions_with_coverage(
        "sub", "aave", first=500, verbose=False)

    assert len(positions) == 4, len(positions)
    assert abs(coverage.captured_borrow_usd - 4_000_000) < 1, coverage.captured_borrow_usd
    assert abs(coverage.coverage_pct - 10.0) < 1e-6, coverage.coverage_pct
    assert coverage.accounts_kept == 4 and coverage.accounts_scanned == 4
    print(f"  {coverage}")


def test_non_stablecoin_debt_is_counted_as_skipped_not_silently_dropped():
    accounts = [_account("0xa1", "WETH", 100, 18, "USDC", 1_000_000, 6),
                _account("0xa2", "WETH", 100, 18, "WETH", 300, 18,
                         debt_price="2800")]
    install(_lending_stub(accounts, protocol_borrows=10_000_000.0))
    positions, coverage = fetch_lending_positions_with_coverage(
        "sub", "aave", first=500, verbose=False)
    assert len(positions) == 1
    assert coverage.skipped["non_stablecoin_debt"] == 1, coverage.skipped
    print(f"  non-stablecoin debt surfaced in the report: {coverage.skipped}")


# ---------------------------------------------------------------------------
# the single-asset collapse -- the worst bug the project had
# ---------------------------------------------------------------------------

def _diversified_whale():
    """$31.2M of collateral across three assets against $20M of USDC debt.
    True blended health factor: (10.75 + 12.80 + 7.68) * 0.80 / 20.00 = 1.249,
    i.e. comfortably safe."""
    def row(side, sym, qty, dec, price, thr="80"):
        return {"side": side, "isCollateral": side == "COLLATERAL",
                "balance": str(int(qty * 10 ** dec)),
                "asset": {"symbol": sym, "decimals": str(dec)},
                "market": {"liquidationThreshold": thr,
                           "inputTokenPriceUSD": price}}
    return {"id": "0xwhale01", "positions": [
        row("COLLATERAL", "WETH", 4_000, 18, "2687.61"),
        row("COLLATERAL", "wstETH", 4_000, 18, "3200.00"),
        row("COLLATERAL", "WBTC", 120, 8, "64000.00"),
        row("BORROWER", "USDC", 20_000_000, 6, "1.00"),
    ]}


WHALE_PRICES = {"WETH": 2687.61, "wstETH": 3200.0, "WBTC": 64000.0, "USDC": 1.0}
WHALE_TRUE_HF = 1.2492


def test_collapse_preserves_the_true_health_factor():
    """The regression test for the phantom-insolvency bug.

    The legacy collapse kept an account's ENTIRE debt against only its
    LARGEST collateral asset, so a diversified whale at HF 1.25 was recorded
    at HF 0.51 -- falsely insolvent, and worth millions in bad debt that did
    not exist. It survived unnoticed because a model that liquidates
    everything for free cannot tell a fake liquidation from a real one; it
    only surfaced once liquidations could be refused and bad debt could
    accrue.
    """
    from cascade_sim import health_factor

    install(_lending_stub([_diversified_whale()]))
    rescaled, _ = fetch_lending_positions_with_coverage(
        "sub", "aave", verbose=False, collapse="rescale")
    assert len(rescaled) == 1
    hf = health_factor(rescaled[0], WHALE_PRICES)
    print(f"  rescale -> HF {hf:.3f} (true {WHALE_TRUE_HF:.3f})")
    assert abs(hf - WHALE_TRUE_HF) < 0.01, (
        f"collapsed HF {hf:.3f} does not match the account's true blended "
        f"HF {WHALE_TRUE_HF:.3f}")

    collateral_value = rescaled[0].collateral_qty * WHALE_PRICES[rescaled[0].collateral_asset]
    assert collateral_value > rescaled[0].debt_qty, (
        "a solvent account must not be recorded as insolvent")


def test_legacy_collapse_still_reproduces_the_bug():
    """Kept deliberately: the bug must stay reproducible so the regression
    above is demonstrably testing something real."""
    from cascade_sim import health_factor

    install(_lending_stub([_diversified_whale()]))
    legacy, _ = fetch_lending_positions_with_coverage(
        "sub", "aave", verbose=False, collapse="legacy")
    hf = health_factor(legacy[0], WHALE_PRICES)
    collateral_value = legacy[0].collateral_qty * WHALE_PRICES[legacy[0].collateral_asset]
    phantom = max(0.0, legacy[0].debt_qty - collateral_value)
    print(f"  legacy  -> HF {hf:.3f}, phantom bad debt ${phantom:,.0f}")
    assert hf < 0.7 and phantom > 1e6, (
        "legacy mode no longer reproduces the bug -- either it was fixed by "
        "accident or the fixture stopped being diversified")


def test_strict_collapse_drops_multi_asset_accounts():
    install(_lending_stub([_diversified_whale()]))
    strict, coverage = fetch_lending_positions_with_coverage(
        "sub", "aave", verbose=False, collapse="strict")
    assert strict == [], "strict mode must not synthesise a single-asset whale"
    assert coverage.skipped.get("multi_asset_account") == 1
    print(f"  strict  -> dropped, and reported: {coverage.skipped}")


def test_single_collateral_account_is_untouched_by_rescale():
    """Rescaling must be a no-op for an account that really is single-asset."""
    from cascade_sim import health_factor

    accounts = [_account("0xplain", "WETH", 100, 18, "USDC", 200_000, 6,
                         coll_price="2800")]
    install(_lending_stub(accounts))
    got, _ = fetch_lending_positions_with_coverage(
        "sub", "aave", verbose=False, collapse="rescale")
    assert abs(got[0].debt_qty - 200_000) < 1.0, (
        f"debt was rewritten on a genuinely single-asset account: "
        f"{got[0].debt_qty:,.0f} vs 200,000")
    print(f"  single-collateral account passes through unchanged "
          f"(HF {health_factor(got[0], {'WETH': 2800.0}):.3f})")


def test_no_longer_orders_by_open_position_count():
    """The bias itself -- pinned so it can't come back."""
    accounts = [_account("0xa1", "WETH", 100, 18, "USDC", 1_000, 6)]
    stub = install(_lending_stub(accounts))
    fetch_lending_positions_with_coverage("sub", "aave", first=10, verbose=False)
    account_queries = [q for q, _ in stub.calls if "accounts(" in q]
    for q in account_queries:
        assert "openPositionCount\n    orderDirection: desc" not in q
        assert "orderBy: openPositionCount" not in q, (
            "account sample is ordered by openPositionCount again -- that is "
            "the sophistication bias this replaced")
    print("  account queries are no longer ordered by openPositionCount")


if __name__ == "__main__":
    _real = live_data.query_subgraph
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for fn in tests:
            print(f"{fn.__name__}:")
            fn()
    finally:
        live_data.query_subgraph = _real
    print(f"\nAll {len(tests)} checks passed.")
