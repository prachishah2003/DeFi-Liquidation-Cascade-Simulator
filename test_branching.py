"""
Tests for the R0 branching ratio.

The property that matters: R0 must cross 1 exactly where the cascade stops
dying out. If it does, it is a live indicator of the same phase transition
the shock scans find by brute force -- and a much cheaper one.
"""

from uniswap_v3_math import TickSegment, sqrt_p_from_price
from cascade_sim import Pool, Position, run_cascade, summarize
from liquidator import LiquidatorEconomics
from branching import (liquidatable_volume, branching_ratio, probe_sensitivity)

MID = 2800.0


def book(liquidity, steps=90, step=0.01):
    segs, price = [], MID
    for _ in range(steps):
        nxt = price * (1 - step)
        segs.append(TickSegment(sqrt_p_from_price(nxt), sqrt_p_from_price(price),
                                liquidity))
        price = nxt
    return segs


def pool(liquidity):
    return Pool("WETH", "USDC", sqrt_p_from_price(MID), book(liquidity))


def ladder(n, spacing, size=150.0, start_hf=0.995):
    """Positions stacked across the liquidation line, `spacing` apart in HF.

    `start_hf` is below 1 on purpose: R0 is a generation ratio, so it is
    defined only once a cascade has something to propagate FROM. With every
    position safe there is no generation zero, R0 is 0, and the question
    "would this cascade spread" has no meaning yet. Post-shock books always
    have a seed; synthetic ones have to be given one.
    """
    return [Position(f"p{i}", "aave", "WETH", size, "USDC",
                     size * MID * 0.80 / (start_hf + spacing * i), 0.80)
            for i in range(n)]


def test_no_underwater_positions_means_zero_r0():
    """With no generation zero there is nothing to propagate, so R0 is 0 --
    not because the book is safe under stress, but because no cascade has
    started. R0 answers "will this spread", not "could it ever"."""
    safe = [Position("p", "aave", "WETH", 100, "USDC", 100 * MID * 0.80 / 2.0, 0.80)]
    est = branching_ratio(safe, {"WETH": pool(20_000_000)}, {"WETH": MID})
    print(f"  a book with nothing near liquidation: R0 = {est.r0:.3f}")
    assert est.r0 == 0.0


def test_dense_ladder_on_a_thin_book_is_supercritical():
    """Many positions packed just above the line, thin depth: each sale should
    trigger more than it sells."""
    positions = ladder(n=400, spacing=0.0004, size=150.0)
    est = branching_ratio(positions, {"WETH": pool(250_000)}, {"WETH": MID})
    print(f"  {est}")
    assert est.supercritical, "a dense ladder on a thin book must be R0 > 1"


def test_sparse_ladder_on_a_deep_book_is_subcritical():
    positions = ladder(n=40, spacing=0.02, size=150.0)
    est = branching_ratio(positions, {"WETH": pool(80_000_000)}, {"WETH": MID})
    print(f"  {est}")
    assert not est.supercritical, "a sparse ladder on a deep book must be R0 < 1"


def test_r0_falls_as_the_book_deepens():
    positions = ladder(n=300, spacing=0.0006, size=150.0)
    last = None
    for liquidity in (200_000, 1_000_000, 5_000_000, 40_000_000):
        est = branching_ratio(positions, {"WETH": pool(liquidity)}, {"WETH": MID})
        print(f"  liquidity {liquidity:>12,}: R0 = {est.r0:8.3f}")
        if last is not None:
            assert est.r0 <= last + 1e-9, "deeper book must not raise R0"
        last = est.r0


def test_r0_predicts_whether_the_cascade_runs_away():
    """The claim R0 exists to support: its sign relative to 1 should agree
    with what a full cascade actually does."""
    for label, n, spacing, liquidity in (
            ("dense/thin", 400, 0.0004, 250_000),
            ("sparse/deep", 40, 0.02, 80_000_000)):
        positions = ladder(n, spacing, size=150.0)
        est = branching_ratio(positions, {"WETH": pool(liquidity)}, {"WETH": MID})

        sim_positions = ladder(n, spacing, size=150.0)
        sim_pool = pool(liquidity)
        logs = run_cascade(sim_positions, {"WETH": sim_pool}, {"WETH": -0.001},
                           max_rounds=30, verbose=False)
        summary = summarize(logs)
        realized = (sim_pool.collateral_price("WETH") / MID - 1) * 100
        amplification = realized + 0.1
        print(f"  {label:12s} R0={est.r0:7.3f} -> a -0.1% nudge realizes "
              f"{realized:+.2f}% ({amplification:+.2f}pp amplification, "
              f"{summary.accounts_liquidated} liquidated)")
        if est.supercritical:
            assert amplification < -0.1, (
                "R0 > 1 but a tiny nudge barely moved anything")
        else:
            assert amplification > -0.1, (
                "R0 < 1 but a tiny nudge set off a cascade")


def test_unprofitable_liquidations_do_not_count_toward_r0():
    """A position nobody will liquidate sells nothing, so it cannot propagate
    anything. Ignoring profitability overstates R0 exactly where it matters."""
    # A book thin enough that unwinding the seized collateral costs more
    # than the 5% bonus pays. Without this the gate never binds and the test
    # passes for the wrong reason -- an earlier version did exactly that,
    # comparing two identical numbers and asserting <=.
    positions = ladder(n=400, spacing=0.0004, size=150.0)
    venues = {"WETH": pool(40_000)}
    free = branching_ratio(positions, venues, {"WETH": MID})
    priced = branching_ratio(positions, venues, {"WETH": MID},
                             economics=LiquidatorEconomics(gas_price_gwei=200.0))
    print(f"  ignoring liquidator economics: R0 = {free.r0:.3f}")
    print(f"  requiring profitability:       R0 = {priced.r0:.3f}")
    assert priced.r0 < free.r0, (
        "the profitability gate did not bind -- this book is not thin enough "
        "for the test to be testing anything")
    assert free.r0 > 0


def test_probe_sensitivity_is_reported():
    positions = ladder(n=300, spacing=0.0006, size=150.0)
    estimates = probe_sensitivity(positions, {"WETH": pool(600_000)},
                                  {"WETH": MID})
    values = [e.r0 for e in estimates]
    for e in estimates:
        print(f"  probe ${e.probe_usd:>12,.0f}: R0 = {e.r0:7.3f}")
    assert len(values) == 5
    assert max(values) > 0, "every probe size read zero -- ladder too sparse"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
    print(f"\nAll {len(tests)} checks passed.")
