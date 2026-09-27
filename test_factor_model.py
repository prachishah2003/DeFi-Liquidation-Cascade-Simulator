"""
Tests for correlated shocks and depeg scenarios.

The claim being pinned: shocking correlated assets one at a time understates
risk, and a stablecoin depeg on the DEBT side moves health factors the
opposite way to the naive intuition.
"""

from factor_model import (FactorShock, DEFAULT_ETH_BETAS, LST_SYMBOLS,
                          market_crash, lst_depeg, stablecoin_depeg,
                          stable_depeg_in_a_crash, estimate_betas,
                          compare_isolated_vs_factor)
from multi_asset import MultiPosition, multi_health_factor

PRICES = {"WETH": 2700.0, "wstETH": 3200.0, "WBTC": 64000.0,
          "USDC": 1.0, "DAI": 1.0}


def test_factor_move_carries_every_correlated_asset():
    shocked = market_crash(-0.30).shocked_prices(PRICES)
    assert abs(shocked["WETH"] / PRICES["WETH"] - 0.70) < 1e-9
    assert abs(shocked["wstETH"] / PRICES["wstETH"] - 0.70) < 1e-9   # beta 1.0
    assert abs(shocked["WBTC"] / PRICES["WBTC"] - 0.76) < 1e-9       # beta 0.8
    assert shocked["USDC"] == 1.0                                    # beta 0
    print("  -30% factor: WETH -30%, wstETH -30%, WBTC -24%, USDC unchanged")


def test_isolated_shock_understates_a_correlated_book():
    """A wstETH-only shock leaves a diversified ETH book looking fine; the
    factor move that would actually produce it does not."""
    position = MultiPosition(
        "p1", "aave",
        collateral={"WETH": 1_000, "wstETH": 1_000, "WBTC": 30},
        collateral_thresholds={"WETH": 0.80, "wstETH": 0.78, "WBTC": 0.75},
        debt={"USDC": 6_000_000})

    isolated, factor = compare_isolated_vs_factor("wstETH", PRICES, shock=-0.20)
    hf_isolated = multi_health_factor(position, isolated)
    hf_factor = multi_health_factor(position, factor)
    print(f"  wstETH -20% in isolation -> HF {hf_isolated:.3f}")
    print(f"  the factor move implying it -> HF {hf_factor:.3f}")
    assert hf_factor < hf_isolated, (
        "a correlated move must hurt more than an isolated one")


def test_lst_depeg_is_worse_than_the_market_move_alone():
    position = MultiPosition(
        "p1", "aave", collateral={"wstETH": 1_000},
        collateral_thresholds={"wstETH": 0.78}, debt={"USDC": 1_900_000})
    plain = market_crash(-0.15).shocked_prices(PRICES)
    depegged = lst_depeg(factor_return=-0.15, depeg=-0.08).shocked_prices(PRICES)
    hf_plain = multi_health_factor(position, plain)
    hf_depeg = multi_health_factor(position, depegged)
    print(f"  -15% market: HF {hf_plain:.3f};  + 8% LST discount: HF {hf_depeg:.3f}")
    assert hf_depeg < hf_plain
    assert depegged["WETH"] == plain["WETH"], "the depeg must hit only LSTs"


def test_stablecoin_debt_depeg_improves_health_factors():
    """The counter-intuitive one: if your DEBT is the thing that breaks, your
    position gets safer, because the debt is cheaper to repay."""
    position = MultiPosition(
        "p1", "aave", collateral={"WETH": 1_000},
        collateral_thresholds={"WETH": 0.80}, debt={"USDC": 2_000_000})
    base = multi_health_factor(position, PRICES)
    depegged = multi_health_factor(
        position, stablecoin_depeg("USDC", -0.12).shocked_prices(PRICES))
    print(f"  USDC at $1.00 -> HF {base:.3f};  at $0.88 -> HF {depegged:.3f}")
    assert depegged > base, "cheaper debt must improve the health factor"


def test_stablecoin_collateral_depeg_hurts():
    """Same depeg, other side of the balance sheet."""
    position = MultiPosition(
        "p1", "aave", collateral={"USDC": 3_000_000},
        collateral_thresholds={"USDC": 0.85}, debt={"WETH": 800})
    base = multi_health_factor(position, PRICES)
    depegged = multi_health_factor(
        position, stablecoin_depeg("USDC", -0.12).shocked_prices(PRICES))
    print(f"  USDC collateral: HF {base:.3f} -> {depegged:.3f}")
    assert depegged < base


def test_combined_scenario_is_between_its_parts():
    position = MultiPosition(
        "p1", "aave", collateral={"WETH": 1_000},
        collateral_thresholds={"WETH": 0.80}, debt={"USDC": 1_800_000})
    crash_only = multi_health_factor(position, market_crash(-0.25).shocked_prices(PRICES))
    combined = multi_health_factor(
        position, stable_depeg_in_a_crash("USDC", -0.12, -0.25).shocked_prices(PRICES))
    print(f"  -25% crash: HF {crash_only:.3f};  with a USDC depeg too: "
          f"HF {combined:.3f}")
    # the depeg partly offsets the crash for a stablecoin BORROWER
    assert combined > crash_only


def test_estimate_betas_recovers_a_known_loading():
    factor = [0.01, -0.02, 0.03, -0.01, 0.005] * 8      # 40 observations
    returns = {
        "WETH": factor,
        "wstETH": [0.99 * r for r in factor],
        "WBTC": [0.60 * r for r in factor],
    }
    betas = estimate_betas(returns)
    print(f"  fitted betas: wstETH {betas['wstETH']:.2f}, WBTC {betas['WBTC']:.2f}")
    assert abs(betas["wstETH"] - 0.99) < 1e-6
    assert abs(betas["WBTC"] - 0.60) < 1e-6


def test_short_history_keeps_the_structural_default():
    betas = estimate_betas({"WETH": [0.01] * 40, "rETH": [0.5, -0.2]})
    assert betas["rETH"] == DEFAULT_ETH_BETAS["rETH"], (
        "a beta must not be fitted to two observations")
    print("  an asset with 2 observations keeps its structural beta, not a fit")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
    print(f"\nAll {len(tests)} checks passed.")
