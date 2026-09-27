"""
Tests for temporary vs permanent price impact.

Two properties define correctness here:

  1. Reversion pulls toward the POST-SHOCK fundamental, never the pre-shock
     price. Getting this wrong would quietly undo the exogenous shock and
     make every cascade look harmless.
  2. kappa=1 removes reflexive feedback entirely -- the market is back at
     fundamental before the next round, so no liquidation can trigger
     another. A cascade under kappa=1 must be exactly the set of positions
     the shock alone put underwater.
"""

from uniswap_v3_math import TickSegment, sqrt_p_from_price
from cascade_sim import Pool, Position, run_cascade, summarize, health_factor
from market_impact import ImpactDecay

MID = 2800.0


def book(liquidity, steps=80, step=0.01):
    segs, price = [], MID
    for _ in range(steps):
        nxt = price * (1 - step)
        segs.append(TickSegment(sqrt_p_from_price(nxt), sqrt_p_from_price(price),
                                liquidity))
        price = nxt
    return segs


def pool(liquidity):
    return Pool("WETH", "USDC", sqrt_p_from_price(MID), book(liquidity))


def ladder(n=60, thin=True):
    """Positions spread across health factors just above 1."""
    out = []
    for i in range(n):
        hf = 1.005 + 0.012 * i
        size = 120.0
        out.append(Position(f"p{i}", "aave", "WETH", size, "USDC",
                            size * MID * 0.80 / hf, 0.80))
    return out


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

def test_revert_moves_the_right_fraction():
    decay = ImpactDecay(kappa=0.25)
    # dislocated to 2,000 against a fundamental of 2,400: gap 400
    got = decay.revert(current_price=2_000.0, reference_price=2_400.0)
    assert abs(got - 2_100.0) < 1e-9, got          # 25% of the gap recovered
    print(f"  25% of a $400 dislocation recovers -> ${got:,.2f}")


def test_kappa_bounds():
    assert ImpactDecay(kappa=0.0).revert(2_000.0, 2_400.0) == 2_000.0
    assert ImpactDecay(kappa=1.0).revert(2_000.0, 2_400.0) == 2_400.0
    for bad in (-0.1, 1.1):
        try:
            ImpactDecay(kappa=bad)
        except ValueError:
            continue
        raise AssertionError(f"kappa={bad} should have been rejected")
    print("  kappa=0 is a no-op, kappa=1 is full recovery, out-of-range rejected")


# ---------------------------------------------------------------------------
# through the cascade
# ---------------------------------------------------------------------------

def _run(kappa, liquidity=900_000, shock=-0.12):
    positions = ladder()
    p = pool(liquidity)
    logs = run_cascade(positions, {"WETH": p}, {"WETH": shock}, max_rounds=30,
                       verbose=False, decay=ImpactDecay(kappa=kappa))
    return summarize(logs), p.collateral_price("WETH")


def test_full_decay_removes_reflexive_feedback():
    """Under kappa=1 the cascade must liquidate exactly the positions the
    shock alone put underwater -- no more, because no price move survives to
    the next round to push anyone else under."""
    shock = -0.12
    shocked_prices = {"WETH": MID * (1 + shock)}
    directly_underwater = sum(
        1 for p in ladder() if health_factor(p, shocked_prices) < 1.0)

    summary, final = _run(kappa=1.0, shock=shock)
    print(f"  shock alone puts {directly_underwater} underwater; "
          f"kappa=1 liquidates {summary.accounts_liquidated}")
    assert summary.accounts_liquidated == directly_underwater
    assert abs(final - MID * (1 + shock)) / MID < 1e-9, (
        "kappa=1 must leave price exactly at the post-shock fundamental")


def test_decay_never_undoes_the_exogenous_shock():
    """The shock itself must survive full reversion."""
    _summary, final = _run(kappa=1.0, shock=-0.30)
    print(f"  after a -30% shock and kappa=1, price is ${final:,.2f} "
          f"(fundamental ${MID*0.70:,.2f}) -- not ${MID:,.2f}")
    assert abs(final - MID * 0.70) / MID < 1e-9


def test_more_decay_means_less_damage():
    """The monotone property the parameter exists to express."""
    rows = []
    for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
        summary, final = _run(kappa)
        rows.append((kappa, summary.accounts_liquidated, final))
        print(f"  kappa={kappa:.2f}: {summary.accounts_liquidated:3d} liquidated, "
              f"final ${final:,.2f} ({(final/MID-1)*100:+.2f}%)")
    counts = [r[1] for r in rows]
    prices = [r[2] for r in rows]
    assert counts == sorted(counts, reverse=True), (
        "more reversion must not liquidate more positions")
    assert prices == sorted(prices), "more reversion must not depress price further"


def test_permanent_impact_is_the_worst_case():
    """kappa=0 -- what every earlier version of this model assumed -- must be
    the most damaging point of the range, so results quoted without a decay
    assumption are an upper bound rather than an estimate."""
    worst, _ = _run(kappa=0.0)
    for kappa in (0.1, 0.5, 0.9, 1.0):
        other, _ = _run(kappa)
        assert other.accounts_liquidated <= worst.accounts_liquidated
    print(f"  kappa=0 liquidates {worst.accounts_liquidated}, the maximum "
          f"across the range -- permanent impact is an upper bound")


def test_round_log_records_both_prices():
    positions = ladder()
    p = pool(900_000)
    logs = run_cascade(positions, {"WETH": p}, {"WETH": -0.12}, max_rounds=30,
                       verbose=False, decay=ImpactDecay(kappa=0.5))
    first = logs[0]
    assert first.prices_before_decay, "pre-decay price not recorded"
    before = first.prices_before_decay["WETH"]
    after = first.prices_after["WETH"]
    print(f"  round 1: ${before:,.2f} at the point of sale -> ${after:,.2f} "
          f"after recovery")
    assert after >= before, "recovery should not lower the price"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
    print(f"\nAll {len(tests)} checks passed.")
