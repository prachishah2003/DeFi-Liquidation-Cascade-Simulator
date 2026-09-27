"""
Oracle staleness model.

Every cascade built so far assumes health factors are computed against the
TRUE, current market price -- but on-chain, Aave (and everyone else) reads
a Chainlink oracle, which does NOT update continuously. Chainlink feeds
typically update only when price moves beyond a deviation threshold (often
~0.5% for major pairs) OR a heartbeat timer elapses (often ~1 hour),
whichever comes first. Real reference: Chainlink's own docs describe this
deviation-threshold + heartbeat design.

Consequence: right after a real price move, positions can look artificially
SAFE on-chain for a stretch, because the oracle hasn't caught up yet --
until it does, at which point a batch of positions can become liquidation-
eligible simultaneously rather than smoothly over time. That's a real,
underexplored mechanism: oracle latency doesn't prevent liquidations, it
delays and BUNCHES them.

SIMPLIFICATION FLAGGED: this assumes liquidators always execute the moment
a position is oracle-eligible, regardless of whether it's actually
profitable given the true (possibly-lower) market price. Real liquidator
bots simulate profitability first and may skip marginal liquidations if the
oracle is stale enough to make them unprofitable. Modeling that is a
further extension, not attempted here.
"""

from dataclasses import dataclass
from typing import Dict, List
from cascade_sim import Position, Pool, RoundLog, health_factor, summarize


@dataclass
class OracleState:
    price: float
    rounds_since_update: int = 0


def run_cascade_oracle_lag(positions: List[Position], pools: Dict[str, Pool],
                            initial_shock: Dict[str, float],
                            deviation_threshold: float = 0.005,
                            heartbeat_rounds: int = 3600,
                            max_rounds: int = 25, verbose: bool = True) -> List[RoundLog]:
    """
    deviation_threshold : fraction move required to trigger an oracle update
                           (0.005 = 0.5%, a realistic major-pair value)
    heartbeat_rounds     : force an update after this many rounds regardless
                           of deviation (very large by default -- rounds
                           here represent cascade iterations, not wall-clock
                           time, so this mostly just disables the heartbeat
                           unless you deliberately set it low to test that
                           mechanism specifically)
    """
    true_prices = {asset: pool.collateral_price(asset) for asset, pool in pools.items()}
    oracle = {asset: OracleState(price=true_prices[asset]) for asset in pools}

    for asset, pct in initial_shock.items():
        if asset in pools:
            target = true_prices[asset] * (1 + pct)
            pools[asset].set_collateral_price(asset, target)
            true_prices[asset] = target
            # NOTE: the oracle does NOT update here -- that's the whole point.
            # It only catches up once the round loop below checks deviation.

    logs: List[RoundLog] = []
    cumulative_sold: Dict[str, float] = {a: 0.0 for a in pools}

    for round_num in range(1, max_rounds + 1):
        # oracle update check -- this is what creates the lag
        oracle_updated_this_round = []
        for asset, state in oracle.items():
            state.rounds_since_update += 1
            deviation = abs(true_prices[asset] - state.price) / state.price if state.price else 0
            if deviation >= deviation_threshold or state.rounds_since_update >= heartbeat_rounds:
                state.price = true_prices[asset]
                state.rounds_since_update = 0
                oracle_updated_this_round.append(asset)

        oracle_prices = {asset: state.price for asset, state in oracle.items()}
        underwater = [p for p in positions
                      if not p.is_fully_liquidated()
                      and health_factor(p, oracle_prices) < 1.0]

        if not underwater:
            if verbose and round_num == 1:
                print(f"round {round_num:2d}: no liquidations yet -- oracle still "
                      f"shows pre-shock prices (deviation hasn't crossed "
                      f"{deviation_threshold:.1%} threshold)")
            elif not any(l.liquidated for l in logs) and round_num > 1:
                pass  # already reported the "waiting" state
            if round_num > 3 and not underwater and not oracle_updated_this_round:
                break  # nothing pending and oracle stable -- done
            continue

        liquidated_ids: List[str] = []
        ran_dry: List[str] = []
        repaid_usd = 0.0
        seized_usd = 0.0
        for pos in underwater:
            # amounts computed against the ORACLE price -- that's genuinely
            # what the smart contract sees and uses on-chain, including the
            # close-factor decision (the contract reads the oracle HF, not the
            # true market HF, so a stale oracle can also mean a stale close
            # factor -- another way lag changes outcomes, not just timing)
            hf_oracle = health_factor(pos, oracle_prices)
            repay = pos.debt_qty * pos.effective_close_factor(hf_oracle)
            seize_value = repay * (1 + pos.liquidation_bonus)
            seize_qty = min(pos.collateral_qty,
                             seize_value / oracle_prices[pos.collateral_asset])
            repaid_usd += repay
            seized_usd += seize_qty * oracle_prices[pos.collateral_asset]
            pos.debt_qty -= repay
            pos.collateral_qty -= seize_qty
            pos.liquidated_rounds.append(round_num)
            liquidated_ids.append(pos.position_id)

            pool = pools.get(pos.collateral_asset)
            if pool is not None and seize_qty > 0:
                # but the SALE executes against the real pool at the TRUE
                # price -- the liquidator can't sell at a price that doesn't
                # actually exist on the market
                result = pool.sell(pos.collateral_asset, seize_qty)
                true_prices[pos.collateral_asset] = pool.collateral_price(pos.collateral_asset)
                cumulative_sold[pos.collateral_asset] += seize_qty
                if result.ran_dry:
                    ran_dry.append(pos.collateral_asset)

        logs.append(RoundLog(round_num=round_num, liquidated=liquidated_ids,
                              prices_after=dict(true_prices),
                              cumulative_collateral_sold=dict(cumulative_sold),
                              pool_ran_dry=ran_dry,
                              debt_repaid_usd=repaid_usd,
                              collateral_seized_usd=seized_usd))

        if verbose:
            oracle_str = ", ".join(f"{a}=${p:,.2f}" for a, p in oracle_prices.items())
            true_str = ", ".join(f"{a}=${p:,.2f}" for a, p in true_prices.items())
            updated_flag = f"  [oracle updated: {oracle_updated_this_round}]" if oracle_updated_this_round else ""
            print(f"round {round_num:2d}: liquidated {len(liquidated_ids)} position(s) "
                  f"-- oracle=[{oracle_str}] true=[{true_str}]{updated_flag}")

    return logs