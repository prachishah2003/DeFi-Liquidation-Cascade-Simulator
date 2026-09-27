"""
Multi-collateral, multi-debt position model.

The original Position/run_cascade (cascade_sim.py) collapses each account
to its single largest collateral asset and single largest debt asset --
a real simplification, since Aave accounts routinely supply several
assets as collateral against several borrowed assets simultaneously.

This module models accounts properly: health factor is the value-weighted
blend across ALL of an account's collateral (each asset has its own
liquidation threshold) against ALL of its debt. This is the actual Aave
health-factor formula, not an approximation of it.

SCOPE (documented honestly): price-impact modeling -- the actual novel
cascade mechanic -- still only covers assets we have a live Uniswap pool
for (WETH, via the one pool wired up so far). Other collateral/debt assets
are priced at their value as of the fetch (static through the cascade).
The architecture supports adding more pools (WBTC/USDC, etc.) by fetching
them the same way and adding to the `pools` dict passed to run_cascade_multi
-- that's the natural next extension, not a redesign.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from typing import Optional
from cascade_sim import (Pool, RoundLog, CLOSE_FACTOR_HF_THRESHOLD, DUST_DEBT,
                         summarize, _price_setting_order)
from liquidator import (LiquidatorEconomics, quote_liquidation,
                        position_bad_debt, largest_profitable_fraction)
from market_impact import ImpactDecay


class MissingPriceError(KeyError):
    """Raised when a position references an asset with no price.

    This used to be swallowed by `prices.get(asset, 0.0)` for collateral and
    `prices.get(asset, 1.0)` for debt. Both defaults are silently wrong and
    both bias the SAME way -- toward liquidation. An unpriced collateral asset
    was valued at zero, making a healthy account look insolvent; an unpriced
    debt asset was valued at $1, which understates any non-stablecoin debt
    (WETH debt is common on Aave and would have been booked at a dollar).
    Failing loudly and dropping the affected positions at the fetch layer is
    the only honest option -- see `drop_unpriced`."""


def _price(prices: Dict[str, float], asset: str) -> float:
    try:
        return prices[asset]
    except KeyError:
        raise MissingPriceError(
            f"no price for {asset!r}; refusing to guess. Either supply it in "
            f"static_prices or filter the position out with drop_unpriced()."
        ) from None


def unpriced_assets(positions, prices: Dict[str, float]) -> set:
    """Every asset referenced by these positions that has no usable price."""
    missing = set()
    for p in positions:
        for asset in list(p.collateral) + list(p.debt):
            if prices.get(asset) is None or prices.get(asset, 0.0) <= 0:
                missing.add(asset)
    return missing


def drop_unpriced(positions, prices: Dict[str, float], verbose: bool = True
                  ) -> Tuple[list, set]:
    """Filter out positions referencing an unpriced asset, loudly.

    Returns (kept_positions, missing_assets). Dropping a position understates
    total exposure, so the count is always reported rather than hidden."""
    missing = unpriced_assets(positions, prices)
    if not missing:
        return list(positions), missing
    kept = [p for p in positions
            if not (missing & (set(p.collateral) | set(p.debt)))]
    if verbose:
        dropped = len(positions) - len(kept)
        print(f"  [drop_unpriced] {dropped} of {len(positions)} positions dropped "
              f"for referencing unpriced assets: {sorted(missing)}")
        print(f"  [drop_unpriced] NOTE: dropped positions are real exposure that "
              f"this run does not capture -- totals below are a lower bound.")
    return kept, missing


@dataclass
class MultiPosition:
    position_id: str
    protocol: str
    collateral: Dict[str, float]             # asset symbol -> qty
    collateral_thresholds: Dict[str, float]  # asset symbol -> liquidation threshold
    debt: Dict[str, float]                   # asset symbol -> qty
    close_factor: float = 0.50
    liquidation_bonus: float = 0.05
    liquidated_rounds: List[int] = field(default_factory=list)

    def is_fully_liquidated(self) -> bool:
        return (sum(self.debt.values()) <= DUST_DEBT
                or sum(self.collateral.values()) <= 1e-9)

    def effective_close_factor(self, hf: float) -> float:
        """See cascade_sim.Position.effective_close_factor -- same Aave rule."""
        return 1.0 if hf < CLOSE_FACTOR_HF_THRESHOLD else self.close_factor


def multi_health_factor(position: MultiPosition, prices: Dict[str, float]) -> float:
    """
    HF = sum(collateral_i * price_i * threshold_i) / sum(debt_j * price_j)
    This is the real Aave formula for multi-asset accounts -- each
    collateral asset contributes its OWN threshold, blended by value,
    rather than applying one asset's threshold to the whole position.
    """
    collateral_weighted = sum(
        qty * _price(prices, asset) * position.collateral_thresholds.get(asset, 0.0)
        for asset, qty in position.collateral.items()
    )
    debt_value = sum(qty * _price(prices, asset) for asset, qty in position.debt.items())
    if debt_value <= 0:
        return float("inf")
    return collateral_weighted / debt_value


def _multi_bad_debt_state(positions: List[MultiPosition], prices: Dict[str, float]):
    """(protocol shortfall, count of insolvent open positions).

    Unlike the single-asset engine this values debt at its ACTUAL price, not
    1:1 -- which is the whole point of running depeg scenarios here. A
    stablecoin trading at $0.88 makes debt cheaper to repay, not dearer.

    A position is skipped only when its DEBT is gone. Skipping on
    `is_fully_liquidated()` -- which is also true when the COLLATERAL hits
    zero -- deleted exactly the worst case: a borrower whose collateral has
    been seized in full while debt remains is the textbook definition of bad
    debt, and it was being dropped from the total.

    It hid in plain sight because it only bites at deep shocks. On the live
    book a -30% crash left three whales unliquidated and underwater, worth
    $373M of bad debt; at -45% the same whales were seized down to zero
    collateral, vanished from the accounting, and the reported total FELL to
    $32M. A deeper crash appearing to cause less damage is the giveaway.
    """
    total = 0.0
    insolvent = 0
    for p in positions:
        if sum(p.debt.values()) <= DUST_DEBT:
            continue                     # the debt is gone; nothing is owed
        collateral_value = sum(qty * prices.get(a, 0.0)
                               for a, qty in p.collateral.items())
        debt_value = sum(qty * prices.get(a, 0.0) for a, qty in p.debt.items())
        shortfall = position_bad_debt(collateral_value, debt_value)
        if shortfall > 0:
            total += shortfall
            insolvent += 1
    return total, insolvent


def run_cascade_multi(positions: List[MultiPosition], pools: Dict[str, Pool],
                       static_prices: Dict[str, float], initial_shock: Dict[str, float],
                       max_rounds: int = 100, verbose: bool = True,
                       economics: Optional[LiquidatorEconomics] = None,
                       decay: Optional[ImpactDecay] = None,
                       gas_price_asset: str = "WETH") -> List[RoundLog]:
    """
    static_prices : fallback USD-ish prices for assets with no live pool
                     (held constant through the cascade -- see SCOPE above)
    pools         : live pools for assets whose price impact IS modeled
                     (their prices override static_prices and update as
                     liquidations sell into them)
    """
    prices = dict(static_prices)
    for asset, pool in pools.items():
        prices[asset] = pool.collateral_price(asset)

    missing = unpriced_assets(positions, prices)
    if missing:
        raise MissingPriceError(
            f"{len(missing)} asset(s) referenced by these positions have no "
            f"price: {sorted(missing)}. Call drop_unpriced(positions, prices) "
            f"first so the dropped exposure is reported rather than silently "
            f"mispriced."
        )

    # Venues first, in dependency order (see _price_setting_order): an LST
    # priced through WETH must be set after WETH, or it moves twice.
    for asset in _price_setting_order(pools):
        pct = initial_shock.get(asset)
        if pct is None:
            continue
        target = prices[asset] * (1 + pct)
        pools[asset].set_collateral_price(asset, target)
        prices[asset] = target
    for asset, pct in initial_shock.items():
        if asset not in pools and asset in prices:
            prices[asset] *= (1 + pct)  # no venue -- reprice statically

    logs: List[RoundLog] = []
    cumulative_sold: Dict[str, float] = {a: 0.0 for a in pools}
    reference_prices = dict(prices)

    for round_num in range(1, max_rounds + 1):
        underwater = [p for p in positions
                      if not p.is_fully_liquidated()
                      and multi_health_factor(p, prices) < 1.0]
        if not underwater:
            break

        liquidated_ids: List[str] = []
        ran_dry: List[str] = []
        data_out: List[str] = []
        skipped: List[str] = []
        partial: List[str] = []
        repaid_usd = 0.0
        seized_usd = 0.0
        eth_price = prices.get(gas_price_asset, 0.0)

        for pos in underwater:
            # liquidator targets the largest debt position by value
            hf = multi_health_factor(pos, prices)
            debt_asset = max(pos.debt, key=lambda a: pos.debt[a] * _price(prices, a))
            repay_qty_full = pos.debt[debt_asset] * pos.effective_close_factor(hf)
            repay_value_full = repay_qty_full * _price(prices, debt_asset)

            collateral_value_total = sum(qty * _price(prices, a)
                                          for a, qty in pos.collateral.items())
            if collateral_value_total <= 0:
                continue

            def proceeds_at(fraction: float, _pos=pos,
                            _total=collateral_value_total,
                            _repay=repay_value_full) -> float:
                """What the unwind fetches if the liquidator repays `fraction`.

                Pro-rata across every collateral asset by value share -- a
                documented simplification of "the liquidator picks whichever
                asset is most convenient", with pro-rata as the neutral
                default. Quotes only; nothing is mutated until the size is
                settled.
                """
                seize = min(_repay * fraction * (1 + _pos.liquidation_bonus),
                            _total)
                out = 0.0
                for asset, qty in _pos.collateral.items():
                    asset_value = qty * _price(prices, asset)
                    if asset_value <= 0:
                        continue
                    share = asset_value / _total
                    leg_qty = min((seize * share) / _price(prices, asset), qty)
                    pool = pools.get(asset)
                    if pool is not None and leg_qty > 0:
                        out += pool.quote_sell(asset, leg_qty).amount_out
                    else:
                        out += leg_qty * _price(prices, asset)
                return out

            # How much of this would a liquidator actually take? Not
            # all-or-nothing: when the full close-factor repayment does not
            # clear its costs, a smaller slice of the same position often
            # does, because slippage grows faster than linearly in size.
            fraction = 1.0
            if economics is not None and economics.enabled:
                fraction = largest_profitable_fraction(
                    proceeds_at, repay_value_full, economics, eth_price)
                if fraction <= 0.0:
                    skipped.append(pos.position_id)
                    continue

            repay_qty = repay_qty_full * fraction
            repay_value = repay_value_full * fraction
            seize_value = min(repay_value * (1 + pos.liquidation_bonus),
                              collateral_value_total)
            # NB: counted here rather than before the decision. It used to be
            # accumulated up front, so positions the liquidator refused still
            # showed up in the round's repaid total.
            repaid_usd += repay_value
            if fraction < 1.0:
                partial.append(pos.position_id)

            # seize pro-rata across every collateral asset by its share of value
            # -- a documented simplification of "liquidator picks the most
            # convenient asset"; pro-rata is the neutral default
            for asset, qty in list(pos.collateral.items()):
                asset_value = qty * _price(prices, asset)
                if asset_value <= 0:
                    continue
                share = asset_value / collateral_value_total
                seize_qty = min((seize_value * share) / _price(prices, asset), qty)
                pos.collateral[asset] -= seize_qty
                seized_usd += seize_qty * _price(prices, asset)

                pool = pools.get(asset)
                if pool is not None and seize_qty > 0:
                    result = pool.sell(asset, seize_qty)
                    prices[asset] = pool.collateral_price(asset)
                    cumulative_sold[asset] = cumulative_sold.get(asset, 0.0) + seize_qty
                    if result.ran_dry:
                        ran_dry.append(asset)
                        if result.exhausted_data:
                            data_out.append(asset)

            pos.debt[debt_asset] -= repay_qty
            pos.liquidated_rounds.append(round_num)
            liquidated_ids.append(pos.position_id)

        prices_before_decay = dict(prices)
        if decay is not None and decay.enabled:
            for asset in _price_setting_order(pools):
                recovered = decay.revert(prices[asset], reference_prices[asset])
                pools[asset].set_collateral_price(asset, recovered)
                prices[asset] = recovered

        bad_debt, insolvent = _multi_bad_debt_state(positions, prices)
        logs.append(RoundLog(round_num=round_num, liquidated=liquidated_ids,
                              prices_after=dict(prices),
                              prices_before_decay=prices_before_decay,
                              cumulative_collateral_sold=dict(cumulative_sold),
                              pool_ran_dry=ran_dry,
                              debt_repaid_usd=repaid_usd,
                              collateral_seized_usd=seized_usd,
                              pool_data_exhausted=data_out,
                              skipped_unprofitable=skipped,
                              partially_liquidated=partial,
                              bad_debt_usd=bad_debt,
                              positions_insolvent=insolvent))

        if skipped and not liquidated_ids:
            if verbose:
                print(f"round {round_num:2d}: {len(skipped)} eligible but none "
                      f"profitable -- cascade halts; ${bad_debt:,.0f} bad debt")
            break

        if verbose:
            price_str = ", ".join(f"{a}=${p:,.2f}" for a, p in prices.items()
                                   if a in pools)
            if data_out:
                dry_note = f"  ** TICK DATA EXHAUSTED: {data_out} (widen tick_window) **"
            elif ran_dry:
                dry_note = f"  ** POOL LIQUIDITY EXHAUSTED: {ran_dry} **"
            else:
                dry_note = ""
            print(f"round {round_num:2d}: liquidated {len(liquidated_ids)} "
                  f"position(s) -> prices now [{price_str}]{dry_note}")

    # A cascade stopped by the round cap is a cascade whose result is "as far
    # as we got", not an outcome. Partial liquidation made this bite: working
    # positions down in slices takes many more rounds than closing them whole,
    # and the old cap of 25 silently truncated runs that needed 28.
    if len(logs) >= max_rounds and logs and logs[-1].liquidated:
        print(f"  WARNING: hit the {max_rounds}-round cap with "
              f"{len(logs[-1].liquidated)} liquidation(s) still in flight. "
              f"This result is truncated, not converged -- raise max_rounds.")
    return logs