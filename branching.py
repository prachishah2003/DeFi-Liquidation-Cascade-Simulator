"""
R0: how much further liquidation each dollar of liquidation causes.

Everything else in this project answers "what happens if the market falls
X%". That requires picking an X, running a full cascade, and reading off the
damage. Useful for scenarios, useless as a live indicator -- you cannot watch
a scenario.

R0 is the same physics expressed as a local quantity. A cascade is a
branching process: each dollar of collateral sold moves the price, the price
move pushes some other position underwater, and that position's liquidation
sells more collateral. The branching ratio is

    R0 = (additional liquidation volume triggered) / (volume sold)

    R0 < 1   each wave is smaller than the last; the cascade dies out
    R0 = 1   the critical point
    R0 > 1   each wave is larger; the cascade runs until something breaks

That is a derivative, evaluated at the CURRENT state, and it needs no
scenario at all. It can be computed every block and watched like a gauge,
which is what makes it the right shape for a monitor rather than a report.
It also explains the phase transition the shock scans found empirically:
the threshold shock is simply where R0 crosses 1.

MEASUREMENT: GENERATIONS, NOT DERIVATIVES
-----------------------------------------
The obvious implementation is a derivative -- sell a small probe, see how
much new liquidation appears, divide. It does not work on a real book, and
the way it fails is worth recording.

Positions sit at discrete health factors. A probe small enough to be a
derivative moves price through a gap in that ladder and triggers nothing, so
R0 reads 0; a probe slightly larger crosses a cluster and R0 jumps to double
digits. Measured against the live Aave book this produced R0 below 1 at
every shock size while the full cascade demonstrably ran away -- the two
methods disagreeing because the probe was measuring the spacing of the
ladder rather than the physics.

The branching-process definition avoids it entirely. A cascade has
generations: the shock makes some set of positions liquidatable (generation
zero), selling that collateral moves the price, and the move makes a further
set liquidatable (generation one). The ratio between consecutive generations
IS the branching ratio, and it needs no arbitrary probe:

    R0 = (volume liquidatable after the first wave sells)
         / (volume the first wave sold)

R0 > 1 means each generation is larger than the last. That is the standard
epidemiological definition, applied to the quantity that actually
propagates.

R0 IS LOCAL; A CASCADE IS NOT
-----------------------------
The obvious use of R0 is as a live gauge: compute it each block, watch for
it to approach 1. That is weaker than it sounds, and the data says so.

Traced generation by generation through the live book just past the edge, at a
-26.1% shock, R0 goes 0.148, 1.448, 1.365, 0.808, 1.814, 1.509, then 0 --
subcritical on the first wave and supercritical from the second. Nothing
changed about the market; the cascade simply walked down into a denser part of
the position ladder, where each dollar of selling flips more accounts than it
did higher up. Held at -26.0%, a tenth of a point shallower, the same book
decays instead and bad debt stays at $0.1M rather than reaching $487M.

READ R0 ONLY WHERE THE DENOMINATOR IS REAL
------------------------------------------
Generation zero is whatever is liquidatable at the shocked price, and just
past the first liquidation that is a handful of accounts. At -20% it is $1.2M
against a $1,229M book, and it triggers $5.7M: R0 = 4.623. The arithmetic is
right and the reading is worthless -- 0.1% of the book flipping 0.5% of it
says nothing about systemic amplification. Deep shocks produce small
denominators too, for the opposite reason: past a point the price move makes
most liquidations unprofitable, so little is liquidatable at all.

So R0 carries information only where generation zero is a material share of
the book. `critical_shock.py` gates the verdict on that, reusing the same
threshold it uses for 'broken'. Ungated, the spurious -20% reading made R0
appear to contradict the brute-force scan by 6 points; gated, the two agree to
about a tenth of a point.

So initial R0 systematically UNDERSTATES danger. It answers "is the next
wave smaller than this one", which is not the same as "does this end well".
`path_branching_ratios` walks the generations and `max_path_r0` reports the
worst point along the way; that is the figure that agrees with running the
cascade to completion. Initial R0 remains useful as a cheap screen -- a book
already supercritical at generation zero is in trouble without question --
but a low reading is not an all-clear.

One detail matters either way: only liquidations a liquidator would actually
perform count. A position that is underwater but unprofitable to liquidate
sells nothing and therefore triggers nothing, and counting it overstates R0
exactly where the model is most interesting -- on a thin book, ignoring
profitability reads R0 = 290 where requiring it reads 0.
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

from cascade_sim import Position, Pool, health_factor
from liquidator import LiquidatorEconomics, quote_liquidation, largest_profitable_fraction


@dataclass
class BranchingEstimate:
    r0: float
    probe_usd: float              # what generation zero sold, in USD
    triggered_usd: float          # what generation one is worth, in USD
    baseline_liquidatable_usd: float
    price_before: float
    price_after: float

    @property
    def supercritical(self) -> bool:
        return self.r0 > 1.0

    def __str__(self) -> str:
        state = "SUPERCRITICAL" if self.supercritical else "subcritical"
        return (f"R0 = {self.r0:.3f} ({state}); a first wave of "
                f"${self.probe_usd:,.0f} moves price "
                f"{(self.price_after/self.price_before - 1)*100:+.3f}% "
                f"and makes ${self.triggered_usd:,.0f} more liquidatable")


def liquidatable_volume(positions: List[Position], prices: Dict[str, float],
                        economics: Optional[LiquidatorEconomics] = None,
                        venues: Optional[Dict[str, Pool]] = None,
                        gas_price_asset: str = "WETH") -> float:
    """Collateral value that liquidators would seize RIGHT NOW, in USD.

    Only counts liquidations that clear their own costs -- an underwater
    position nobody will touch sells nothing and therefore triggers nothing.
    """
    eth_price = prices.get(gas_price_asset, 0.0)
    total = 0.0
    for position in positions:
        if position.is_fully_liquidated():
            continue
        price = prices.get(position.collateral_asset)
        if not price or price <= 0:
            continue
        hf = health_factor(position, prices)
        if hf >= 1.0:
            continue

        repay_full = position.debt_qty * position.effective_close_factor(hf)

        def seize_for(fraction, _pos=position, _repay=repay_full, _price=price):
            value = _repay * fraction * (1 + _pos.liquidation_bonus)
            return min(_pos.collateral_qty, value / _price)

        seize_qty = seize_for(1.0)
        if seize_qty <= 0:
            continue

        if economics is not None and economics.enabled:
            pool = (venues or {}).get(position.collateral_asset)

            def proceeds_at(f, _pool=pool, _pos=position, _price=price):
                return quote_liquidation(
                    pool=_pool, collateral_asset=_pos.collateral_asset,
                    seize_qty=seize_for(f), mark_price=_price,
                    repay_usd=0.0,          # proceeds only; costs applied inside
                    econ=economics, eth_price_usd=eth_price).proceeds_usd

            # Must use the SAME liquidation rule as the cascade, or R0 is
            # cross-checking a different model than the one it validates.
            # This was all-or-nothing -- a position whose full close-factor
            # repayment was refused contributed ZERO here, while the cascade
            # went on to liquidate a profitable slice of it. R0 read low for
            # exactly the positions partial liquidation rescues.
            fraction = largest_profitable_fraction(
                proceeds_at, repay_full, economics, eth_price)
            if fraction <= 0.0:
                continue
            seize_qty = seize_for(fraction)

        total += seize_qty * price
    return total


def branching_ratio(positions: List[Position], venues: Dict[str, Pool],
                    prices: Dict[str, float], asset: str = "WETH",
                    economics: Optional[LiquidatorEconomics] = None
                    ) -> BranchingEstimate:
    """R0 at the current state: generation one divided by generation zero.

    Sells everything currently liquidatable, then measures how much MORE
    becomes liquidatable at the resulting price. No probe size to choose,
    and therefore nothing to tune.
    """
    venue = venues[asset]
    price_before = venue.collateral_price(asset)

    generation_zero = liquidatable_volume(positions, prices, economics, venues)
    if generation_zero <= 0 or price_before <= 0:
        # nothing is liquidatable, so no cascade can begin
        return BranchingEstimate(0.0, 0.0, 0.0, 0.0, price_before, price_before)

    sim_venues = {a: copy.deepcopy(v) for a, v in venues.items()}
    sold_qty = generation_zero / price_before
    sim_venues[asset].sell(asset, sold_qty)
    price_after = sim_venues[asset].collateral_price(asset)

    shocked_prices = dict(prices)
    shocked_prices[asset] = price_after

    # Generation one is what becomes liquidatable that was NOT already --
    # measured on positions with generation zero's collateral removed, so
    # the same dollars are not counted in both waves.
    remaining = [p for p in positions
                 if health_factor(p, prices) >= 1.0 and not p.is_fully_liquidated()]
    generation_one = liquidatable_volume(remaining, shocked_prices,
                                         economics, sim_venues)

    return BranchingEstimate(
        r0=generation_one / generation_zero,
        probe_usd=generation_zero,
        triggered_usd=generation_one,
        baseline_liquidatable_usd=generation_zero,
        price_before=price_before,
        price_after=price_after,
    )


def marginal_branching_ratio(positions: List[Position], venues: Dict[str, Pool],
                             prices: Dict[str, float], asset: str = "WETH",
                             probe_fraction: float = 0.002,
                             economics: Optional[LiquidatorEconomics] = None
                             ) -> BranchingEstimate:
    """The derivative form: how much a SMALL probe sale triggers.

    Kept because it is the intuitive definition and useful on a synthetic
    book with a dense position ladder. Do not use it on a real book without
    checking `probe_sensitivity` first -- on discrete positions the answer
    depends strongly on probe size, which is what made it disagree with the
    full cascade. `branching_ratio` is the one to trust.
    """
    venue = venues[asset]
    price_before = venue.collateral_price(asset)
    baseline = liquidatable_volume(positions, prices, economics, venues)

    capacity = venue.sell_capacity(asset) if hasattr(venue, "sell_capacity") else 0.0
    if not capacity or capacity <= 0:
        capacity = sum(p.collateral_qty for p in positions) or 1.0
    probe_qty = max(capacity * probe_fraction, 1e-9)

    sim_venues = {a: copy.deepcopy(v) for a, v in venues.items()}
    sim_venues[asset].sell(asset, probe_qty)
    price_after = sim_venues[asset].collateral_price(asset)

    shocked = dict(prices)
    shocked[asset] = price_after
    after = liquidatable_volume(positions, shocked, economics, sim_venues)
    triggered = max(0.0, after - baseline)
    probe_usd = probe_qty * price_before

    return BranchingEstimate(
        r0=triggered / probe_usd if probe_usd > 0 else 0.0,
        probe_usd=probe_usd, triggered_usd=triggered,
        baseline_liquidatable_usd=baseline,
        price_before=price_before, price_after=price_after)


def path_branching_ratios(positions: List[Position], venues: Dict[str, Pool],
                          prices: Dict[str, float], asset: str = "WETH",
                          economics: Optional[LiquidatorEconomics] = None,
                          max_generations: int = 12) -> List[BranchingEstimate]:
    """R0 at every generation of the cascade, not just the first.

    Executes each wave for real -- selling its collateral into the venues and
    removing it from the positions -- so the next measurement is taken at the
    state the cascade has actually reached.
    """
    sim_positions = copy.deepcopy(positions)
    sim_venues = {a: copy.deepcopy(v) for a, v in venues.items()}
    out: List[BranchingEstimate] = []

    for _ in range(max_generations):
        current = dict(prices)
        current[asset] = sim_venues[asset].collateral_price(asset)
        estimate = branching_ratio(sim_positions, sim_venues, current,
                                   asset, economics)
        out.append(estimate)
        if estimate.probe_usd <= 0:
            break

        # execute this generation: sell its collateral and retire its debt
        price_now = current[asset]
        sim_venues[asset].sell(asset, estimate.probe_usd / price_now)
        for position in sim_positions:
            if position.is_fully_liquidated():
                continue
            hf = health_factor(position, current)
            if hf >= 1.0:
                continue
            repay = position.debt_qty * position.effective_close_factor(hf)
            seize = min(position.collateral_qty,
                        repay * (1 + position.liquidation_bonus) / price_now)
            position.debt_qty -= repay
            position.collateral_qty -= seize
    return out


def max_path_r0(positions: List[Position], venues: Dict[str, Pool],
                prices: Dict[str, float], asset: str = "WETH",
                economics: Optional[LiquidatorEconomics] = None) -> float:
    """The worst R0 reached anywhere along the cascade.

    This is the figure that matches what running the cascade to completion
    does; initial R0 alone does not. See the note above.
    """
    path = path_branching_ratios(positions, venues, prices, asset, economics)
    return max((e.r0 for e in path), default=0.0)


def probe_sensitivity(positions, venues, prices, asset="WETH",
                      fractions=(0.0005, 0.001, 0.002, 0.005, 0.01),
                      economics=None) -> List[BranchingEstimate]:
    """The derivative form across probe sizes -- the diagnostic that shows
    why it cannot be trusted on a discrete book."""
    return [marginal_branching_ratio(positions, venues, prices, asset, f,
                                     economics)
            for f in fractions]
