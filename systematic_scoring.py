"""
Systemic asset scoring.

With multi-asset accounts now modeled properly, we can ask a genuinely
network-flavored question: which COLLATERAL ASSETS are most systemically
dangerous, not which accounts are? An asset used as collateral by many
marginal accounts is a bigger systemic risk than one used by a few very
safe ones, independent of that asset's own price volatility.

Method: for each asset, shock ONLY that asset's price (holding every other
asset's price fixed), and count how many accounts flip from healthy
(HF >= 1) to underwater (HF < 1). This isolates each asset's blast radius
without needing a live pool for every single asset -- it's a direct
repricing + health-factor recompute, using data already fetched.

This is a single-hop score (direct exposure only). A true network
centrality measure would also count second-order effects (an account that
flips because of asset A might itself be collateralized in asset B for
someone else) -- that's a natural deeper extension, flagged not built.
"""

from typing import Dict, List
from multi_asset import MultiPosition, multi_health_factor


def score_systemic_assets(positions: List[MultiPosition], prices: Dict[str, float],
                           shock_pct: float = -0.20) -> List[dict]:
    """
    shock_pct: the isolated shock applied to each asset in turn (default
    -20%, a meaningful but not extreme move -- change it to test sensitivity
    to shock size, same as the shock_sweep for the main cascade).
    """
    assets_present = set()
    for p in positions:
        assets_present.update(p.collateral.keys())

    baseline_hf = {p.position_id: multi_health_factor(p, prices) for p in positions}
    baseline_safe = {pid: hf >= 1.0 for pid, hf in baseline_hf.items()}

    results = []
    for asset in sorted(assets_present):
        if asset not in prices or prices[asset] <= 0:
            continue
        shocked_prices = dict(prices)
        shocked_prices[asset] *= (1 + shock_pct)

        exposed = 0
        flipped = 0
        value_at_risk = 0.0
        for p in positions:
            if asset not in p.collateral:
                continue
            exposed += 1
            if not baseline_safe[p.position_id]:
                continue
            new_hf = multi_health_factor(p, shocked_prices)
            if new_hf < 1.0:
                flipped += 1
                value_at_risk += sum(qty * prices.get(a, 0) for a, qty in p.debt.items())

        results.append({
            "asset": asset,
            "accounts_holding": exposed,
            "accounts_flipped": flipped,
            "flip_rate": flipped / exposed if exposed else 0.0,
            "debt_value_at_risk": value_at_risk,
        })

    results.sort(key=lambda r: r["accounts_flipped"], reverse=True)
    return results


def print_ranking(results: List[dict], shock_pct: float, top_n: int = 15):
    print(f"\nSystemic asset ranking (each shocked {shock_pct:+.0%} in isolation):")
    print(f"{'Asset':10s}{'Holders':>10s}{'Flipped':>10s}{'Flip rate':>12s}{'Debt at risk':>16s}")
    for r in results[:top_n]:
        print(f"{r['asset']:10s}{r['accounts_holding']:>10d}{r['accounts_flipped']:>10d}"
              f"{r['flip_rate']:>11.1%}{'$' + format(r['debt_value_at_risk'], ',.0f'):>16s}")