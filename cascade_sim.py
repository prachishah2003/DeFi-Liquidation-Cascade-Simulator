"""
Endogenous DeFi liquidation cascade simulator.

Most liquidation-risk tools treat a price move as an *exogenous* shock and
just ask "who's underwater at this price?" That misses the actual mechanism:

    liquidation -> liquidator sells seized collateral into a DEX pool
                -> that sell moves the pool's price via AMM price impact
                -> the new (lower) price pushes MORE positions underwater
                -> repeat

This module closes that loop: every liquidation is routed through a real
Uniswap-v3-style pool (see uniswap_v3_math.py), and the resulting price
change feeds back into the health-factor check for the next round. The
cascade terminates when a round produces no new liquidations, or when a
pool "runs dry" (infinite slippage).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uniswap_v3_math import (TickSegment, swap_multi_tick, SwapResult,
                              input_to_reach_sqrt_p, max_input_available,
                              price_from_sqrt_p, sqrt_p_from_price)
from liquidator import (LiquidatorEconomics, LiquidationQuote,
                        quote_liquidation, position_bad_debt,
                        largest_profitable_fraction)
from market_impact import ImpactDecay


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Aave v3 raises the close factor to 100% once a position is deeply enough
# underwater; above that threshold only `close_factor` of the debt may be
# repaid in one go. Without this rule a position is repeatedly liquidated at
# 50% forever -- its debt decays geometrically but never reaches zero, so it
# never trips is_fully_liquidated() and gets counted as a fresh liquidation
# every round. Reference: Aave v3 CLOSE_FACTOR_HF_THRESHOLD.
CLOSE_FACTOR_HF_THRESHOLD = 0.95
DUST_DEBT = 1e-6          # below this a position is treated as closed


@dataclass
class Position:
    position_id: str
    protocol: str                 # e.g. "aave", "compound", "maker"
    collateral_asset: str         # e.g. "WETH"
    collateral_qty: float
    debt_asset: str               # e.g. "USDC"
    debt_qty: float
    liquidation_threshold: float  # e.g. 0.80 = 80% LTV triggers liquidation
    close_factor: float = 0.50    # fraction of debt a liquidator may repay at once
    liquidation_bonus: float = 0.05  # extra collateral liquidator receives, e.g. 5%
    liquidated_rounds: List[int] = field(default_factory=list)  # audit trail

    def is_fully_liquidated(self) -> bool:
        return self.debt_qty <= DUST_DEBT or self.collateral_qty <= 1e-9

    def effective_close_factor(self, hf: float) -> float:
        """Fraction of outstanding debt a liquidator may repay right now."""
        return 1.0 if hf < CLOSE_FACTOR_HF_THRESHOLD else self.close_factor


@dataclass
class Pool:
    """
    A DEX pool between token0 and token1, exactly as Uniswap v3 orders them
    on-chain (by ascending contract address -- NOT by which one you think of
    as "the collateral"). Do not assume token0 is your collateral asset --
    on real pools it frequently isn't (e.g. the USDC/WETH 0.3% pool has
    token0=USDC, token1=WETH because USDC's address sorts lower).

    sqrt_p tracks sqrt(price), where price = token1 per token0 (raw
    Uniswap convention). Everything else derives from that plus knowing
    which symbol is token0 vs token1.
    """
    token0_symbol: str
    token1_symbol: str
    sqrt_p: float
    segments: List[TickSegment]
    # The price range the fetched tick data actually covers. Outside it the
    # model has no information -- which is NOT the same as the pool having no
    # liquidity there. `data_truncated` records that the tick fetch stopped
    # early (page cap or window edge) rather than reaching the end of the book.
    data_truncated: bool = False
    tick_window: Optional[int] = None
    # The pool's actual fee, in basis points. Uniswap runs a separate pool per
    # fee tier and they charge very different amounts -- 5bp, 30bp, 100bp.
    # This defaulted to 30bp everywhere, so the 0.05% tier was being charged
    # six times its real fee. That tier is the deepest of the three and takes
    # most of the routed flow, so the error made routed execution look worse
    # than it is on exactly the venue that matters most.
    fee_bps: float = 30.0
    # What the protocol says this pool actually holds. Used to check the
    # reconstructed book against reality -- see `book_consistency`.
    tvl_token0: Optional[float] = None
    tvl_token1: Optional[float] = None
    tvl_usd: Optional[float] = None
    # cache of (sqrt_p, symbol) -> reserve-limited capacity, since the bound
    # only changes when the price does
    _capacity_cache: dict = field(default_factory=dict, repr=False)

    def data_price_bounds(self) -> Optional[tuple]:
        """(lowest, highest) price the fetched segments cover, in token1/token0."""
        if not self.segments:
            return None
        return (price_from_sqrt_p(min(s.sqrt_p_lower for s in self.segments)),
                price_from_sqrt_p(max(s.sqrt_p_upper for s in self.segments)))

    def collateral_price_bounds(self, collateral_symbol: str) -> Optional[tuple]:
        """(low, high) price for `collateral_symbol` that the fetched tick
        data covers. Outside this range the model is blind -- which is not
        the same as the pool being empty."""
        bounds = self.data_price_bounds()
        if bounds is None:
            return None
        low, high = bounds
        if collateral_symbol == self.token0_symbol:
            return (low, high)
        if collateral_symbol == self.token1_symbol:
            return (1.0 / high, 1.0 / low)
        raise ValueError(f"{collateral_symbol!r} is not in this pool")

    def _price_1_per_0(self) -> float:
        return price_from_sqrt_p(self.sqrt_p)

    def collateral_price(self, collateral_symbol: str) -> float:
        """Price of `collateral_symbol` in terms of whichever other token
        is in this pool (that other token is assumed to be your numeraire,
        e.g. a stablecoin)."""
        p1_per_0 = self._price_1_per_0()
        if collateral_symbol == self.token0_symbol:
            return p1_per_0
        elif collateral_symbol == self.token1_symbol:
            return 1.0 / p1_per_0
        raise ValueError(f"{collateral_symbol!r} is not in this pool "
                          f"({self.token0_symbol}/{self.token1_symbol})")

    def set_collateral_price(self, collateral_symbol: str, price: float) -> None:
        """Used to apply an exogenous shock directly (sets pool price without
        a swap -- represents 'the whole market repriced', not a single trade)."""
        if collateral_symbol == self.token0_symbol:
            p1_per_0 = price
        elif collateral_symbol == self.token1_symbol:
            p1_per_0 = 1.0 / price
        else:
            raise ValueError(f"{collateral_symbol!r} is not in this pool")
        self.sqrt_p = sqrt_p_from_price(p1_per_0)

    def input_to_reach_collateral_price(self, collateral_symbol: str,
                                        target_price: float) -> float:
        """Tokens of `collateral_symbol` needed to push its price here down to
        `target_price`. inf if the fetched book cannot reach it."""
        if collateral_symbol == self.token0_symbol:
            zero_for_one = True
            target_1_per_0 = target_price
        elif collateral_symbol == self.token1_symbol:
            zero_for_one = False
            target_1_per_0 = 1.0 / target_price
        else:
            raise ValueError(f"{collateral_symbol!r} is not in this pool")
        return input_to_reach_sqrt_p(self.sqrt_p, self.segments,
                                     sqrt_p_from_price(target_1_per_0),
                                     zero_for_one, self.fee_bps)

    def reserve_limited_capacity(self, collateral_symbol: str) -> float:
        """Largest sale whose payout the pool can actually cover.

        A swap hands out the OTHER token, so it cannot pay out more of it
        than the pool holds. The protocol reports those holdings, and they
        are ground truth in a way the reconstructed tick book is not.

        This matters because the tick reconstruction can be badly
        incomplete. On the live wstETH/WETH 0.01% pool the active liquidity
        is L = 9,645,623 while every `liquidityNet` crossing inside the
        fetched +/-6000 tick window totals just 543 -- the boundary ticks
        that should collapse L back down are simply not in the returned
        data. The pool's own reserves say that L is valid across a band
        about 0.04% wide; the model, seeing no crossings, applied it across
        the entire window and concluded the pool could pay out 919,345 WETH
        when it holds 1,980. A 464x overstatement, and every depth figure
        derived from it was fiction.

        Bisects on input size, since payout rises monotonically with it.
        Returns inf when the pool's holdings were not fetched -- absence of
        the constraint is not the same as the constraint being satisfied.
        """
        if collateral_symbol == self.token0_symbol:
            held = self.tvl_token1
        elif collateral_symbol == self.token1_symbol:
            held = self.tvl_token0
        else:
            raise ValueError(f"{collateral_symbol!r} is not in this pool")
        if held is None or held <= 0:
            return float("inf")

        key = (collateral_symbol, self.sqrt_p)
        cached = self._capacity_cache.get(key)
        if cached is not None:
            return cached

        book = self.book_capacity(collateral_symbol)
        if book <= 0:
            self._capacity_cache[key] = 0.0
            return 0.0
        # NB: _quote_uncapped, not quote_sell -- quote_sell enforces this very
        # cap, so calling it here would recurse forever.
        if self._quote_uncapped(collateral_symbol, book).amount_out <= held:
            self._capacity_cache[key] = book
            return book

        lo, hi = 0.0, book
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if self._quote_uncapped(collateral_symbol, mid).amount_out > held:
                hi = mid
            else:
                lo = mid
        self._capacity_cache[key] = lo
        return lo

    def book_capacity(self, collateral_symbol: str) -> float:
        """What the reconstructed tick book alone says it can absorb, with no
        reserve constraint applied. Kept separate so the two can be compared
        -- see `book_consistency`."""
        if collateral_symbol == self.token0_symbol:
            zero_for_one = True
        elif collateral_symbol == self.token1_symbol:
            zero_for_one = False
        else:
            raise ValueError(f"{collateral_symbol!r} is not in this pool")
        return max_input_available(self.sqrt_p, self.segments, zero_for_one,
                                   self.fee_bps)

    def sell_capacity(self, collateral_symbol: str) -> float:
        """The largest sale this pool's FETCHED book can absorb.

        Accumulated by walking the book, not by asking what it costs to reach
        the bottom: that question answers "infinity" whenever an empty tick
        range sits in between, and reading infinity as zero inverts the
        answer. A live rETH/WETH pool with real depth at the top and one
        empty range partway down reported zero capacity, which failed every
        rETH liquidation and produced bad debt that did not exist.

        Probing with a huge order does not work either: `amount_in_filled` is
        `amount_in - remaining`, and at 1e30 the finite capacity is below the
        float64 ulp of the probe, so the subtraction annihilates it.
        """
        return min(self.book_capacity(collateral_symbol),
                   self.reserve_limited_capacity(collateral_symbol))

    def book_consistency(self, collateral_symbol: str):
        """Compare the reconstructed book against the pool's reported holdings.

        A sale of one token can extract at most the OTHER token the pool
        actually holds -- that is an accounting fact, not a modelling choice.
        If walking the fetched ticks implies more output than the pool owns,
        the liquidity reconstruction is wrong somewhere and every depth
        figure derived from it is inflated.

        Returns (implied_out, reported_holdings, ratio) or None when the
        pool's holdings were not fetched. A ratio above ~1 means the book is
        claiming depth the pool does not have.
        """
        if collateral_symbol == self.token0_symbol:
            held = self.tvl_token1
        elif collateral_symbol == self.token1_symbol:
            held = self.tvl_token0
        else:
            raise ValueError(f"{collateral_symbol!r} is not in this pool")
        if held is None or held <= 0:
            return None
        capacity = self.book_capacity(collateral_symbol)
        if capacity <= 0:
            return (0.0, held, 0.0)
        # _quote_uncapped on purpose: this measures what the RAW reconstructed
        # book claims, which is the thing being checked. Quoting through the
        # capped path would compare the constraint against itself and always
        # report a tidy 1.0x.
        implied_out = self._quote_uncapped(collateral_symbol, capacity).amount_out
        return (implied_out, held, implied_out / held)

    def _quote_uncapped(self, collateral_symbol: str, amount: float) -> SwapResult:
        """The raw tick walk, with no reserve constraint applied."""
        if collateral_symbol == self.token0_symbol:
            zero_for_one = True
        elif collateral_symbol == self.token1_symbol:
            zero_for_one = False
        else:
            raise ValueError(f"{collateral_symbol!r} is not in this pool "
                              f"({self.token0_symbol}/{self.token1_symbol})")
        return swap_multi_tick(self.sqrt_p, self.segments, amount, zero_for_one,
                               self.fee_bps)

    def quote_sell(self, collateral_symbol: str, amount: float) -> SwapResult:
        """What `sell()` WOULD do, without doing it. A liquidator simulates
        the unwind before deciding whether to bid; so does this model.

        The reserve constraint is enforced HERE, at the single point every
        caller goes through, rather than only in the capacity accounting. An
        earlier version capped `sell_capacity` and the router's ladder but
        left this method quoting against the raw book, so anything holding a
        Pool directly -- `TwoHopVenue` selling an LST through its LST/WETH
        leg, for one -- still received quotes the pool could never honour.
        The cap has to live where the quote is produced or it is decorative.
        """
        capacity = self.sell_capacity(collateral_symbol)
        if amount <= capacity:
            return self._quote_uncapped(collateral_symbol, amount)

        result = self._quote_uncapped(collateral_symbol, capacity)
        return SwapResult(
            new_sqrt_p=result.new_sqrt_p,
            amount_out=result.amount_out,
            amount_in_filled=min(capacity, result.amount_in_filled),
            ticks_crossed=result.ticks_crossed,
            ran_dry=True,
            exhausted_data=result.exhausted_data,
            hit_zero_liquidity=result.hit_zero_liquidity,
        )

    def sell(self, collateral_symbol: str, amount: float):
        """Sell `amount` of `collateral_symbol` into the pool (what a
        liquidator does with seized collateral). Direction (zero_for_one)
        is derived from which token collateral_symbol actually is."""
        if collateral_symbol == self.token0_symbol:
            zero_for_one = True
        elif collateral_symbol == self.token1_symbol:
            zero_for_one = False
        else:
            raise ValueError(f"{collateral_symbol!r} is not in this pool "
                              f"({self.token0_symbol}/{self.token1_symbol})")
        result = self.quote_sell(collateral_symbol, amount)
        self.sqrt_p = result.new_sqrt_p

        # Settle the pool's holdings. A swap hands the pool the token being
        # sold and takes the other one out, so both balances move -- and the
        # payout token is exactly what `reserve_limited_capacity` rations.
        #
        # BUG HISTORY: this used to move only the price. The reserve cap was
        # therefore re-derived from the pool's ORIGINAL balance every call, so
        # a venue could be sold into without limit: the live wstETH leg
        # returned a capacity of 1,591 wstETH, then 1,592, then 1,594, for as
        # many rounds as the cascade cared to run, each one paying out WETH the
        # pool had already spent.
        #
        # Uniswap's L does NOT change as price moves through a range, so
        # leaving the tick book itself alone is correct; what was wrong was
        # treating the payout token as inexhaustible. Covered by
        # test_reserve_depletion.py.
        if zero_for_one:
            if self.tvl_token0 is not None:
                self.tvl_token0 += result.amount_in_filled
            if self.tvl_token1 is not None:
                self.tvl_token1 = max(0.0, self.tvl_token1 - result.amount_out)
        else:
            if self.tvl_token1 is not None:
                self.tvl_token1 += result.amount_in_filled
            if self.tvl_token0 is not None:
                self.tvl_token0 = max(0.0, self.tvl_token0 - result.amount_out)
        # holdings changed, so every cached capacity is stale
        self._capacity_cache.clear()
        return result


@dataclass
class RoundLog:
    round_num: int
    liquidated: List[str]          # liquidation EVENTS this round, not accounts
    prices_after: Dict[str, float]
    cumulative_collateral_sold: Dict[str, float]
    pool_ran_dry: List[str]
    prices_before_decay: Dict[str, float] = field(default_factory=dict)
    debt_repaid_usd: float = 0.0        # valued at the price used by the liquidator
    collateral_seized_usd: float = 0.0
    # Assets whose sale could not be filled because the FETCHED TICK DATA ran
    # out, as opposed to the pool genuinely having no liquidity left. Only the
    # latter is an economic result; the former is a fetch-window problem.
    pool_data_exhausted: List[str] = field(default_factory=list)
    # Positions that WERE eligible for liquidation but which no rational
    # liquidator would touch, because the bonus did not cover slippage + gas.
    skipped_unprofitable: List[str] = field(default_factory=list)
    # Positions a liquidator took only PART of, because the full close-factor
    # repayment did not clear its costs but a smaller slice did.
    partially_liquidated: List[str] = field(default_factory=list)
    bad_debt_usd: float = 0.0        # protocol shortfall as of end of round
    positions_insolvent: int = 0     # open positions with collateral < debt


@dataclass
class CascadeSummary:
    """What actually happened, in the units people quote.

    `liquidation_events` and `accounts_liquidated` are DIFFERENT numbers and
    the distinction matters: one account can be liquidated in several
    consecutive rounds as the price keeps falling. Summing len(log.liquidated)
    across rounds -- which every script in this project used to do and report
    as "positions liquidated" -- counts events, and overstated account counts
    by ~70% on a six-position test book."""
    rounds: int
    liquidation_events: int
    accounts_liquidated: int
    collateral_sold: Dict[str, float]
    debt_repaid_usd: float
    collateral_seized_usd: float
    pools_ran_dry: List[str]
    pools_data_exhausted: List[str] = field(default_factory=list)
    skipped_unprofitable: int = 0
    accounts_skipped_unprofitable: int = 0
    accounts_partially_liquidated: int = 0
    bad_debt_usd: float = 0.0
    positions_left_insolvent: int = 0

    @property
    def result_is_trustworthy(self) -> bool:
        """False when the run walked off the end of the fetched tick data --
        the cascade was cut short by missing data, not by the market."""
        return not self.pools_data_exhausted

    def __str__(self) -> str:
        parts = [f"{self.accounts_liquidated} accounts liquidated across "
                 f"{self.liquidation_events} events in {self.rounds} round(s); "
                 f"${self.debt_repaid_usd:,.0f} debt repaid, "
                 f"${self.collateral_seized_usd:,.0f} collateral seized"]
        if self.skipped_unprofitable:
            parts.append(f"  {self.accounts_skipped_unprofitable} account(s) went "
                         f"unliquidated ({self.skipped_unprofitable} skipped bids) "
                         f"because no liquidator could profit")
        if self.bad_debt_usd > 0:
            parts.append(f"  ** BAD DEBT: ${self.bad_debt_usd:,.0f} across "
                         f"{self.positions_left_insolvent} insolvent position(s) -- "
                         f"the protocol eats this **")
        if self.pools_data_exhausted:
            parts.append(f"  ** TICK DATA EXHAUSTED for "
                         f"{sorted(set(self.pools_data_exhausted))} -- widen "
                         f"tick_window; this is a data limit, NOT a finding **")
        real_dry = sorted(set(self.pools_ran_dry) - set(self.pools_data_exhausted))
        if real_dry:
            parts.append(f"  pools with genuinely exhausted liquidity: {real_dry}")
        return "\n".join(parts)


def summarize(logs: List[RoundLog]) -> CascadeSummary:
    """Collapse a cascade's round logs into headline figures. Always use this
    rather than summing len(log.liquidated) by hand."""
    sold: Dict[str, float] = dict(logs[-1].cumulative_collateral_sold) if logs else {}
    return CascadeSummary(
        rounds=len(logs),
        liquidation_events=sum(len(l.liquidated) for l in logs),
        accounts_liquidated=len({pid for l in logs for pid in l.liquidated}),
        collateral_sold=sold,
        debt_repaid_usd=sum(l.debt_repaid_usd for l in logs),
        collateral_seized_usd=sum(l.collateral_seized_usd for l in logs),
        pools_ran_dry=[a for l in logs for a in l.pool_ran_dry],
        pools_data_exhausted=[a for l in logs for a in l.pool_data_exhausted],
        skipped_unprofitable=sum(len(l.skipped_unprofitable) for l in logs),
        accounts_skipped_unprofitable=len(
            {pid for l in logs for pid in l.skipped_unprofitable}),
        accounts_partially_liquidated=len(
            {pid for l in logs for pid in l.partially_liquidated}),
        bad_debt_usd=logs[-1].bad_debt_usd if logs else 0.0,
        positions_left_insolvent=logs[-1].positions_insolvent if logs else 0,
    )


# ---------------------------------------------------------------------------
# Health factor
# ---------------------------------------------------------------------------

def health_factor(position: Position, prices: Dict[str, float],
                   numeraire_per_debt: float = 1.0) -> float:
    """
    HF = (collateral_value * liquidation_threshold) / debt_value
    Assumes debt_asset is priced 1:1 in the numeraire (fine for stablecoin
    debt like USDC; pass numeraire_per_debt for non-stable debt assets).
    """
    collateral_value = position.collateral_qty * prices[position.collateral_asset]
    debt_value = position.debt_qty * numeraire_per_debt
    if debt_value <= 0:
        return float("inf")
    return (collateral_value * position.liquidation_threshold) / debt_value


# ---------------------------------------------------------------------------
# Cascade simulation
# ---------------------------------------------------------------------------

def _price_setting_order(pools: Dict[str, object]) -> List[str]:
    """Assets in the order their venue prices must be set.

    A venue that quotes its asset through an intermediate (an LST priced as
    ratio x WETH -- see routing.TwoHopVenue) reads the intermediate's CURRENT
    price to translate a target. So the intermediate has to be repriced
    first, or the dependent asset is moved twice: once by its own shock and
    again when the intermediate moves under it.

    Detected by duck-typing on `intermediate` rather than importing routing,
    which imports this module.
    """
    direct = [a for a, v in pools.items() if not hasattr(v, "intermediate")]
    dependent = [a for a, v in pools.items() if hasattr(v, "intermediate")]
    return direct + dependent


def _bad_debt_state(positions: List[Position], prices: Dict[str, float]):
    """(total protocol shortfall, count of insolvent open positions).

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
        if p.debt_qty <= DUST_DEBT:
            continue                     # the debt is gone; nothing is owed
        collateral_value = p.collateral_qty * prices.get(p.collateral_asset, 0.0)
        debt_value = p.debt_qty            # stablecoin debt, priced 1:1
        shortfall = position_bad_debt(collateral_value, debt_value)
        if shortfall > 0:
            total += shortfall
            insolvent += 1
    return total, insolvent


def run_cascade(positions: List[Position], pools: Dict[str, Pool],
                 initial_shock: Dict[str, float], max_rounds: int = 100,
                 verbose: bool = True,
                 economics: Optional[LiquidatorEconomics] = None,
                 gas_price_asset: str = "WETH",
                 decay: Optional[ImpactDecay] = None) -> List[RoundLog]:
    """
    positions      : all open positions across protocols
    pools          : {collateral_asset: Pool} — one DEX pool per collateral
                      asset, used both to read live price and route sales.
                      Each Pool's token0/token1 can be in either order --
                      collateral_price()/sell() handle that automatically.
    initial_shock  : {asset: pct_change}, e.g. {"WETH": -0.15} for a 15% drop
    economics      : liquidator cost structure. Pass one and each liquidation
                      is checked for profitability before it happens; an
                      unprofitable position is LEFT OPEN and its shortfall
                      accrues as protocol bad debt. Pass None (the default)
                      to keep the old free-and-instant behaviour, which
                      overstates liquidations and can never produce bad debt.
    gas_price_asset: which asset's price to value gas in (gas is paid in ETH)
    decay          : how much cascade-induced price dislocation reverts
                      between rounds (arbitrage, market makers, passive
                      liquidity refilling). None or kappa=0 keeps impact
                      fully permanent, which is what every earlier version of
                      this model assumed and is the most aggressive end of
                      the range. See market_impact.py.
    """
    prices = {asset: pool.collateral_price(asset) for asset, pool in pools.items()}
    cumulative_sold: Dict[str, float] = {a: 0.0 for a in pools}
    logs: List[RoundLog] = []

    # --- apply the exogenous shock first (this is what starts the cascade) ---
    for asset in _price_setting_order(pools):
        pct = initial_shock.get(asset)
        if pct is None:
            continue
        target_price = prices[asset] * (1 + pct)
        pools[asset].set_collateral_price(asset, target_price)
        prices[asset] = target_price

    # The post-shock fundamental each asset reverts TOWARD. Captured after the
    # exogenous shock, so reversion undoes only the dislocation the cascade's
    # own selling caused -- never the shock itself.
    reference_prices = dict(prices)

    for round_num in range(1, max_rounds + 1):
        # find every position currently underwater and not yet fully liquidated
        underwater = [p for p in positions
                      if not p.is_fully_liquidated()
                      and health_factor(p, prices) < 1.0]

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
            # close factor depends on how far underwater the position is --
            # deeply underwater positions can be closed in full, which is what
            # lets them terminate instead of being half-liquidated forever
            hf = health_factor(pos, prices)
            repay_full = pos.debt_qty * pos.effective_close_factor(hf)
            mark = prices[pos.collateral_asset]

            def seize_for(fraction: float, _pos=pos, _repay=repay_full,
                          _mark=mark) -> float:
                value = _repay * fraction * (1 + _pos.liquidation_bonus)
                return min(_pos.collateral_qty, value / _mark) if _mark > 0 else 0.0

            # How much would anyone actually take? Price it before committing.
            fraction = 1.0
            if economics is not None and economics.enabled:
                pool_for_quote = pools.get(pos.collateral_asset)

                def proceeds_at(f, _pool=pool_for_quote, _pos=pos, _mark=mark):
                    return quote_liquidation(
                        pool=_pool,
                        collateral_asset=_pos.collateral_asset,
                        seize_qty=seize_for(f),
                        mark_price=_mark,
                        repay_usd=0.0,      # proceeds only; costs applied below
                        econ=economics,
                        eth_price_usd=eth_price,
                    ).proceeds_usd

                fraction = largest_profitable_fraction(
                    proceeds_at, repay_full, economics, eth_price)
                if fraction <= 0.0:
                    # Nobody bids, at any size. The position stays open and
                    # underwater -- the mechanism that produces bad debt, and
                    # the reason cascades stop short of exhausting liquidity.
                    skipped.append(pos.position_id)
                    continue
                if fraction < 1.0:
                    partial.append(pos.position_id)

            repay = repay_full * fraction
            seize_qty = seize_for(fraction)
            repaid_usd += repay
            seized_usd += seize_qty * prices[pos.collateral_asset]

            pos.debt_qty -= repay
            pos.collateral_qty -= seize_qty
            pos.liquidated_rounds.append(round_num)
            liquidated_ids.append(pos.position_id)

            pool = pools.get(pos.collateral_asset)
            if pool is not None and seize_qty > 0:
                result = pool.sell(pos.collateral_asset, seize_qty)
                prices[pos.collateral_asset] = pool.collateral_price(pos.collateral_asset)
                cumulative_sold[pos.collateral_asset] += seize_qty
                if result.ran_dry:
                    ran_dry.append(pos.collateral_asset)
                    if result.exhausted_data:
                        data_out.append(pos.collateral_asset)

        # Liquidity comes back between waves: the liquidator already ate the
        # slippage at the depressed price, but the market recovers part of it
        # before the NEXT round's health factors are computed. That recovery
        # is the whole reason a cascade can fail to propagate.
        prices_before_decay = dict(prices)
        if decay is not None and decay.enabled:
            for asset in _price_setting_order(pools):
                recovered = decay.revert(prices[asset], reference_prices[asset])
                pools[asset].set_collateral_price(asset, recovered)
                prices[asset] = recovered

        bad_debt, insolvent = _bad_debt_state(positions, prices)
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

        # If every eligible position was rejected as unprofitable, nothing
        # moved and nothing will next round either -- the cascade is over,
        # halted by economics rather than by liquidity.
        if skipped and not liquidated_ids:
            if verbose:
                print(f"round {round_num:2d}: {len(skipped)} position(s) eligible "
                      f"but NONE profitable to liquidate -- cascade halts here; "
                      f"${bad_debt:,.0f} of bad debt left on the protocol")
            break

        if verbose:
            price_str = ", ".join(f"{a}=${p:,.2f}" for a, p in prices.items())
            if data_out:
                dry_note = f"  ** TICK DATA EXHAUSTED: {data_out} (widen tick_window) **"
            elif ran_dry:
                dry_note = f"  ** POOL LIQUIDITY EXHAUSTED: {ran_dry} **"
            else:
                dry_note = ""
            skip_note = (f"  [{len(skipped)} skipped as unprofitable]"
                          if skipped else "")
            debt_note = (f"  [bad debt ${bad_debt:,.0f}]" if bad_debt > 0 else "")
            print(f"round {round_num:2d}: liquidated {len(liquidated_ids)} "
                  f"position(s) -> prices now [{price_str}]"
                  f"{skip_note}{debt_note}{dry_note}")

    # A cascade stopped by the round cap is a cascade whose result is "as far
    # as we got", not an outcome. Partial liquidation made this bite: working
    # positions down in slices takes many more rounds than closing them whole,
    # and the old cap of 25 silently truncated runs that needed 28.
    if len(logs) >= max_rounds and logs and logs[-1].liquidated:
        print(f"  WARNING: hit the {max_rounds}-round cap with "
              f"{len(logs[-1].liquidated)} liquidation(s) still in flight. "
              f"This result is truncated, not converged -- raise max_rounds.")
    return logs