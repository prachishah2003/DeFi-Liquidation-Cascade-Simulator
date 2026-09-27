"""
Tests for liquidator economics and bad debt.

The claim being pinned down: a liquidation happens only when someone profits
from it, and when nobody does, the shortfall lands on the protocol as bad
debt. Both halves matter -- a model that liquidates everything can never
produce bad debt, and bad debt is the figure a risk team actually asks for.
"""

from uniswap_v3_math import TickSegment, sqrt_p_from_price
from cascade_sim import Pool, Position, run_cascade, summarize
from liquidator import (LiquidatorEconomics, quote_liquidation,
                        position_bad_debt)

MID = 2800.0


def book(mid, liquidity, steps=40, step=0.02):
    segs, price = [], mid
    for _ in range(steps):
        nxt = price * (1 - step)
        segs.append(TickSegment(sqrt_p_from_price(nxt), sqrt_p_from_price(price),
                                liquidity))
        price = nxt
    return segs


def pool(liquidity, mid=MID):
    return Pool("WETH", "USDC", sqrt_p_from_price(mid), book(mid, liquidity))


def position(pid, collateral, hf, threshold=0.80, mid=MID):
    """A position at a chosen health factor."""
    return Position(pid, "aave", "WETH", collateral, "USDC",
                    collateral * mid * threshold / hf, threshold)


# ---------------------------------------------------------------------------
# quoting
# ---------------------------------------------------------------------------

def test_quote_sell_does_not_move_the_pool():
    p = pool(20_000_000)
    before = p.collateral_price("WETH")
    p.quote_sell("WETH", 5_000.0)
    assert p.collateral_price("WETH") == before, "quoting must not execute"
    p.sell("WETH", 5_000.0)
    assert p.collateral_price("WETH") < before, "selling must execute"
    print("  quoting leaves the pool untouched; selling moves it")


def test_deep_book_liquidation_is_profitable():
    econ = LiquidatorEconomics()
    q = quote_liquidation(pool(80_000_000), "WETH", seize_qty=100.0,
                          mark_price=MID, repay_usd=100 * MID / 1.05,
                          econ=econ, eth_price_usd=MID)
    print(f"  deep book: slippage {q.slippage_pct:.3f}%, profit ${q.profit_usd:,.0f}")
    assert q.profitable and q.profit_usd > 0


def test_thin_book_slippage_eats_the_bonus():
    """A 5% bonus is worthless if unwinding the collateral costs 8%."""
    econ = LiquidatorEconomics()
    q = quote_liquidation(pool(400_000), "WETH", seize_qty=4_000.0,
                          mark_price=MID, repay_usd=4_000 * MID / 1.05,
                          econ=econ, eth_price_usd=MID)
    print(f"  thin book: slippage {q.slippage_pct:.2f}%, profit "
          f"${q.profit_usd:,.0f} -> {q.reason}")
    assert q.rejected and q.slippage_pct > 5.0


def test_gas_kills_dust_liquidations():
    """A $60 position isn't worth 450k gas, however healthy the book."""
    econ = LiquidatorEconomics(gas_price_gwei=80.0)
    q = quote_liquidation(pool(80_000_000), "WETH", seize_qty=0.02,
                          mark_price=MID, repay_usd=0.02 * MID / 1.05,
                          econ=econ, eth_price_usd=MID)
    print(f"  dust position: gas ${q.gas_usd:,.2f} vs profit ${q.profit_usd:,.2f} "
          f"-> {q.reason}")
    assert q.rejected


def test_disabled_economics_always_accepts():
    econ = LiquidatorEconomics(enabled=False)
    q = quote_liquidation(pool(400_000), "WETH", seize_qty=4_000.0,
                          mark_price=MID, repay_usd=4_000 * MID / 1.05,
                          econ=econ, eth_price_usd=MID)
    assert q.profitable, "disabled economics must reproduce the old model"
    print("  economics disabled -> old free-and-instant behaviour preserved")


# ---------------------------------------------------------------------------
# bad debt arithmetic
# ---------------------------------------------------------------------------

def test_position_bad_debt_is_the_shortfall():
    assert position_bad_debt(collateral_value_usd=90.0, debt_value_usd=100.0) == 10.0
    assert position_bad_debt(collateral_value_usd=150.0, debt_value_usd=100.0) == 0.0
    print("  shortfall = debt - collateral, floored at zero")


# ---------------------------------------------------------------------------
# end-to-end through the cascade
# ---------------------------------------------------------------------------

def test_cascade_without_economics_liquidates_everything():
    positions = [position(f"p{i}", 400, hf=1.02 + 0.01 * i) for i in range(8)]
    logs = run_cascade(positions, {"WETH": pool(400_000)},
                       {"WETH": -0.25}, verbose=False)
    s = summarize(logs)
    assert s.skipped_unprofitable == 0
    assert s.bad_debt_usd == 0.0, "the old model cannot produce bad debt"
    print(f"  no economics: {s.accounts_liquidated} accounts liquidated, "
          f"$0 bad debt (by construction)")


def test_cascade_with_economics_leaves_bad_debt_on_a_thin_book():
    positions = [position(f"p{i}", 400, hf=1.02 + 0.01 * i) for i in range(8)]
    logs = run_cascade(positions, {"WETH": pool(400_000)}, {"WETH": -0.25},
                       verbose=False, economics=LiquidatorEconomics())
    s = summarize(logs)
    print(f"  with economics: {s.accounts_liquidated} liquidated, "
          f"{s.accounts_skipped_unprofitable} skipped, "
          f"${s.bad_debt_usd:,.0f} bad debt")
    assert s.accounts_skipped_unprofitable > 0, (
        "on a book this thin some liquidations must be unprofitable")


def test_economics_reduce_liquidations_versus_free_model():
    """The headline comparison: the same shock, same book, with and without a
    liquidator who has to make money."""
    def run(econ):
        positions = [position(f"p{i}", 400, hf=1.02 + 0.01 * i) for i in range(8)]
        return summarize(run_cascade(positions, {"WETH": pool(600_000)},
                                     {"WETH": -0.25}, verbose=False,
                                     economics=econ))

    free = run(None)
    priced = run(LiquidatorEconomics())
    print(f"  free model:  {free.accounts_liquidated} liquidated, "
          f"${free.collateral_seized_usd:,.0f} seized, "
          f"${free.bad_debt_usd:,.0f} bad debt")
    print(f"  priced model:{priced.accounts_liquidated} liquidated, "
          f"${priced.collateral_seized_usd:,.0f} seized, "
          f"${priced.bad_debt_usd:,.0f} bad debt")
    assert priced.accounts_liquidated <= free.accounts_liquidated
    assert priced.collateral_seized_usd <= free.collateral_seized_usd


def test_high_gas_suppresses_liquidations():
    """Gas spikes during crashes. The monotone effect is on LIQUIDATIONS:
    dearer gas means strictly fewer of them clear the profitability bar.

    The effect on bad debt is NOT monotone, which is the interesting part and
    the reason this test does not assert a direction for it. Two forces pull
    opposite ways:

      - skipping a liquidation leaves a position open and underwater, which
        ADDS to bad debt;
      - skipping it also avoids dumping that collateral into the pool, so the
        price doesn't fall as far, which REDUCES every other position's
        shortfall.

    Either can dominate depending on book depth and how clustered health
    factors are. In the configuration below the second wins narrowly, so
    dearer gas actually lowers measured bad debt -- a genuinely
    counter-intuitive result worth reporting rather than asserting away, and
    a reminder that "liquidations failed" and "the protocol lost money" are
    not the same claim.
    """
    def run_at(gwei):
        positions = [position(f"p{i}", 3, hf=1.01 + 0.005 * i) for i in range(12)]
        return summarize(run_cascade(
            positions, {"WETH": pool(3_000_000)}, {"WETH": -0.22}, verbose=False,
            economics=LiquidatorEconomics(gas_price_gwei=gwei)))

    calm, spike = run_at(10.0), run_at(600.0)
    print(f"  10 gwei : {calm.accounts_liquidated} liquidated, "
          f"{calm.accounts_skipped_unprofitable} skipped, "
          f"${calm.bad_debt_usd:,.0f} bad debt")
    print(f"  600 gwei: {spike.accounts_liquidated} liquidated, "
          f"{spike.accounts_skipped_unprofitable} skipped, "
          f"${spike.bad_debt_usd:,.0f} bad debt")
    assert spike.accounts_liquidated <= calm.accounts_liquidated, (
        "dearer gas cannot increase the number of profitable liquidations")
    assert spike.accounts_skipped_unprofitable >= calm.accounts_skipped_unprofitable
    print("  (bad debt direction is deliberately not asserted -- see docstring)")


def test_fully_seized_positions_still_count_as_bad_debt():
    """The worst case must not be excluded from the total.

    A borrower whose collateral has been seized in full while debt remains
    IS the bad debt. Skipping such positions because they look "fully
    liquidated" made a deeper crash appear to cause less damage: on the live
    book a -30% shock reported $373M while -45% reported $32M, purely
    because the largest insolvent positions had been seized to zero and
    dropped out of the sum.
    """
    from cascade_sim import _bad_debt_state

    seized = Position("wiped", "aave", "WETH", 0.0, "USDC", 5_000_000.0, 0.80)
    healthy = Position("fine", "aave", "WETH", 1_000.0, "USDC", 500_000.0, 0.80)
    repaid = Position("closed", "aave", "WETH", 0.0, "USDC", 0.0, 0.80)

    total, count = _bad_debt_state([seized, healthy, repaid], {"WETH": MID})
    print(f"  collateral seized to zero with $5.0M debt outstanding -> "
          f"bad debt ${total:,.0f} across {count} position(s)")
    assert abs(total - 5_000_000.0) < 1.0, (
        "a position seized to zero collateral with debt left is bad debt")
    assert count == 1, "the healthy and the repaid position must not count"


def test_deeper_shocks_do_not_reduce_bad_debt_via_seizure():
    """Monotonicity where it should hold: on a book with no liquidations
    possible at all, a deeper crash cannot lower the shortfall."""
    from cascade_sim import _bad_debt_state
    p = Position("p", "aave", "WETH", 100.0, "USDC", 400_000.0, 0.80)
    worse = None
    for price in (3_000.0, 2_500.0, 2_000.0, 1_500.0):
        total, _ = _bad_debt_state([p], {"WETH": price})
        print(f"  WETH ${price:,.0f} -> bad debt ${total:,.0f}")
        if worse is not None:
            assert total >= worse, "a lower price reduced the shortfall"
        worse = total


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
    print(f"\nAll {len(tests)} checks passed.")
