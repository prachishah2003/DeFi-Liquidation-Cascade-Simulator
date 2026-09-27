"""Liquidation bonus comes from the market, not from a constant.

Aave's bonus varies by reserve — 5% on blue-chip collateral, 7.5–10% on
thinner assets — and the model applied a flat 5% everywhere. That understates
the incentive to liquidate exactly the assets whose books are thinnest.

Note what is NOT fetched here: Aave v3's close factor is protocol-level (50%,
rising to 100% below the health-factor threshold), so there is no per-market
close factor to read. The earlier limitation text claimed both were per-market.

The query is deliberately separate from the positions query. Snapshots are
keyed on query text, so adding a field to an existing query invalidates every
recorded snapshot; a separate one simply misses on old snapshots and falls
back, which keeps the offline replay workflow usable.
"""

import pytest

import live_data as L


@pytest.fixture
def fake_markets(monkeypatch):
    """Swap out the transport so these tests never touch the network."""
    def install(rows):
        monkeypatch.setattr(L, "paginate",
                            lambda *a, **k: rows)
    return install


def test_percentage_is_converted_to_a_fraction(fake_markets):
    fake_markets([
        {"id": "0x1", "liquidationPenalty": "5", "inputToken": {"symbol": "WETH"}},
        {"id": "0x2", "liquidationPenalty": "7.5", "inputToken": {"symbol": "LINK"}},
    ])
    bonuses = L.fetch_market_risk_params("sub", verbose=False)
    assert bonuses["WETH"] == pytest.approx(0.05)
    assert bonuses["LINK"] == pytest.approx(0.075)


def test_markets_reporting_nothing_are_skipped_not_zeroed(fake_markets):
    """A market reporting 0 is a market with no data, not a free liquidation.

    Recording it as a 0% bonus would make the liquidator's economics look
    strictly worse than the default and silently suppress liquidations there.
    """
    fake_markets([
        {"id": "0x1", "liquidationPenalty": "0", "inputToken": {"symbol": "WETH"}},
        {"id": "0x2", "liquidationPenalty": None, "inputToken": {"symbol": "DAI"}},
        {"id": "0x3", "liquidationPenalty": "6", "inputToken": {"symbol": "WBTC"}},
    ])
    bonuses = L.fetch_market_risk_params("sub", verbose=False)
    assert "WETH" not in bonuses
    assert "DAI" not in bonuses
    assert bonuses["WBTC"] == pytest.approx(0.06)


def test_atoken_symbols_are_normalised(fake_markets):
    """Same normalisation as everywhere else, or the lookup silently misses.

    aEthWETH *is* WETH; a bonus filed under the aToken symbol would never
    match a position whose collateral was normalised to the underlying.
    """
    fake_markets([
        {"id": "0x1", "liquidationPenalty": "5", "inputToken": {"symbol": "aEthWETH"}},
    ])
    bonuses = L.fetch_market_risk_params("sub", verbose=False)
    assert "WETH" in bonuses


def test_an_old_snapshot_falls_back_instead_of_exploding(monkeypatch):
    """Replaying a snapshot recorded before this query existed must degrade.

    The snapshot layer raises on a miss, by design — inventing data would be
    worse. Here that must become "use the default", not a crash, or adding
    this feature would break every existing snapshot.
    """
    def miss(*a, **k):
        raise RuntimeError("This query is not in the snapshot, so replaying "
                           "it would have to invent data")
    monkeypatch.setattr(L, "paginate", miss)
    assert L.fetch_market_risk_params("sub", verbose=False) == {}


def test_unrelated_runtime_errors_still_propagate(monkeypatch):
    """Only the snapshot miss is caught. Everything else is a real failure."""
    def boom(*a, **k):
        raise RuntimeError("gateway returned 502")
    monkeypatch.setattr(L, "paginate", boom)
    with pytest.raises(RuntimeError, match="502"):
        L.fetch_market_risk_params("sub", verbose=False)


def test_default_is_the_documented_aave_value():
    assert L.DEFAULT_LIQUIDATION_BONUS == pytest.approx(0.05)
