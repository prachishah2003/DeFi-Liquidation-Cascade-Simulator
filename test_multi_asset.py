"""
Synthetic validation for multi_asset.py -- mixed collateral (WETH + WBTC)
against mixed debt (USDC + DAI), with a live-style pool only for WETH
(WBTC held at a static price) to exercise both code paths: assets whose
price impact IS modeled, and assets whose price is held constant.
"""

from uniswap_v3_math import TickSegment, sqrt_p_from_price
from cascade_sim import Pool
from multi_asset import MultiPosition, multi_health_factor, run_cascade_multi


def build_weth_pool(mid_price: float) -> Pool:
    sqrt_p = sqrt_p_from_price(mid_price)
    segments = []
    p = mid_price
    for i in range(15):
        p_next = p * 0.98
        segments.append(TickSegment(sqrt_p_lower=sqrt_p_from_price(p_next),
                                     sqrt_p_upper=sqrt_p_from_price(p),
                                     liquidity=8_000_000 * (0.85 ** i)))
        p = p_next
    return Pool(token0_symbol="WETH", token1_symbol="USDC", sqrt_p=sqrt_p, segments=segments)


def build_positions(weth_price: float, wbtc_price: float):
    return [
        # pure WETH collateral, single debt asset -- should match the
        # single-asset model's behavior as a sanity check
        MultiPosition("p1", "aave",
                      collateral={"WETH": 400}, collateral_thresholds={"WETH": 0.80},
                      debt={"USDC": 400 * weth_price * 0.80 / 1.10}),
        # blended WETH + WBTC collateral against blended USDC + DAI debt --
        # the actual new capability
        MultiPosition("p2", "aave",
                      collateral={"WETH": 200, "WBTC": 5},
                      collateral_thresholds={"WETH": 0.80, "WBTC": 0.75},
                      debt={"USDC": 150_000, "DAI": 200_000}),
        # WBTC-only collateral -- price impact NOT modeled (no pool for it),
        # so this position should only move if its static price is shocked
        MultiPosition("p3", "aave",
                      collateral={"WBTC": 10}, collateral_thresholds={"WBTC": 0.75},
                      debt={"USDC": 10 * wbtc_price * 0.75 / 1.08}),
    ]


def test_pure_weth_position_matches_expected_hf():
    weth_price = 2800.0
    positions = build_positions(weth_price, wbtc_price=60_000.0)
    prices = {"WETH": weth_price, "WBTC": 60_000.0, "USDC": 1.0, "DAI": 1.0}

    hf_p1 = multi_health_factor(positions[0], prices)
    print(f"p1 (pure WETH, target HF ~1.10): {hf_p1:.3f}")
    assert abs(hf_p1 - 1.10) < 0.01

    hf_p2 = multi_health_factor(positions[1], prices)
    coll_val = 200 * weth_price + 5 * 60_000.0
    debt_val = 150_000 + 200_000
    print(f"p2 (blended WETH+WBTC vs USDC+DAI): {hf_p2:.3f}")
    assert hf_p2 > 0

    hf_p3 = multi_health_factor(positions[2], prices)
    print(f"p3 (pure WBTC, target HF ~1.08): {hf_p3:.3f}")
    assert abs(hf_p3 - 1.08) < 0.01
    print("Health factor calculations check out.\n")


def test_cascade_with_mixed_pool_coverage():
    weth_price = 2800.0
    wbtc_price = 60_000.0
    positions = build_positions(weth_price, wbtc_price)
    pool = build_weth_pool(weth_price)
    pools = {"WETH": pool}
    static_prices = {"WBTC": wbtc_price, "USDC": 1.0, "DAI": 1.0}

    print("Positions before shock:")
    prices_before = dict(static_prices)
    prices_before["WETH"] = pool.collateral_price("WETH")
    for p in positions:
        print(f"  {p.position_id}: HF = {multi_health_factor(p, prices_before):.3f}")

    print("\nApplying -12% WETH shock (WBTC untouched) and running cascade...")
    logs = run_cascade_multi(positions, pools, static_prices,
                              initial_shock={"WETH": -0.12}, max_rounds=10)

    final_weth_price = pool.collateral_price("WETH")
    print(f"\nFinal WETH price: ${final_weth_price:,.2f} "
          f"(started ${weth_price:,.2f})")
    assert final_weth_price < weth_price

    # p3 is pure WBTC with no shock applied to WBTC -- should be untouched
    assert not any("p3" in l.liquidated for l in logs), \
        "p3 (pure WBTC, unshocked) should not have been liquidated"
    print("p3 (pure WBTC, unshocked) correctly untouched -- confirms static-price "
          "assets don't get swept up in a shock aimed at a different asset.")

    total_liquidated = sum(len(l.liquidated) for l in logs)
    print(f"\nTotal liquidation events across rounds: {total_liquidated}")
    print("All checks passed.")


if __name__ == "__main__":
    test_pure_weth_position_matches_expected_hf()
    test_cascade_with_mixed_pool_coverage()