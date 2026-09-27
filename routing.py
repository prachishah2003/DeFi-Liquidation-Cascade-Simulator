"""
Route a liquidation across every venue that would actually take it.

The model used to sell every liquidated position into one Uniswap v3 pool.
On live Aave data that meant pushing hundreds of millions of dollars of WETH
through a single 0.3% pool holding perhaps a hundred million of depth, which
drove the modelled price to the floor of the fetched book at essentially
every shock size. That is not a finding about the market. It is an artifact
of pretending one pool is the whole market.

Real liquidators do not do this. They route through an aggregator that splits
an order across every Uniswap fee tier, Curve, Balancer and whatever else
quotes, and the largest venues for WETH are not on-chain at all. Ignoring
that overstates price impact -- and therefore cascade amplification -- by a
wide margin.

HOW THE SPLIT IS DECIDED
------------------------
Not by a heuristic: by the condition that maximises what the seller actually
receives, which is what an aggregator is paid to find.

The first version equalised the post-trade PRICE across venues, on the
reasoning that arbitrage holds venues together. That is right when fees are
equal and wrong once they are not. A pool charging 100bp needs more input to
reach any given price than one charging 5bp, because the fee is taken out of
the input rather than moving the price -- so equalising price sends MORE
flow to the dearer pool. Exactly backwards, and it only became visible once
each pool carried its real fee instead of an assumed 30bp.

What a seller maximises is output, and total output is maximised when the
MARGINAL proceeds per unit sold are equal everywhere. For an AMM the
marginal proceeds are `(1 - fee) * price`, so the common quantity is the
fee-adjusted price:

    find m such that  sum over venues of (input each absorbs reaching
                                          price m / (1 - fee_i)) = size

One unknown, solved by bisection; each venue answers in closed form; and a
cheap venue is now correctly given more of the order than an expensive one
of identical depth.

OFF-CHAIN DEPTH
---------------
`LinearDepthVenue` stands in for everything not modelled tick-by-tick:
centralised exchanges, market-maker inventory, on-chain venues without a
fetched book. It is a linear order book parameterised by the notional that
moves price 1%, which is the form depth is usually quoted in and the form a
trader can sanity-check. It is a deliberate simplification -- real books are
neither linear nor infinitely replenishing -- and it is the single most
important assumption in this file, so it is a constructor argument and the
sweeps vary it.
"""

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from cascade_sim import Pool
from uniswap_v3_math import SwapResult, sqrt_p_from_price, price_from_sqrt_p

BISECTION_STEPS = 60


class _PoolLadder:
    """Precomputed cost curve for one pool at its current price.

    Routing bisects on a common clearing price, asking every venue "how much
    do you absorb getting to P" at each step. Answering that by walking the
    tick list each time is O(segments) per question, and a real 0.05% pool
    has thousands of segments -- the first version re-sorted ~3,100 of them
    on all 60 bisection steps of every liquidation of every round, which made
    a single sweep take minutes.

    Building the cumulative cost ladder once per routing decision turns each
    question into a binary search: O(log segments) instead of O(segments).
    """

    __slots__ = ("edges", "cum", "liquidity", "near", "zero_for_one",
                 "fee_factor", "total", "collateral_is_token0", "fee_bps",
                 "reserve_cap")

    def __init__(self, pool: Pool, symbol: str, fee_bps: Optional[float] = None):
        fee_bps = pool.fee_bps if fee_bps is None else fee_bps
        # The pool's reserves cap what this ladder may claim, however much
        # depth the tick reconstruction appears to show. See
        # Pool.reserve_limited_capacity.
        self.reserve_cap = pool.sell_capacity(symbol)
        if symbol == pool.token0_symbol:
            self.zero_for_one = True
            self.collateral_is_token0 = True
        elif symbol == pool.token1_symbol:
            self.zero_for_one = False
            self.collateral_is_token0 = False
        else:
            raise ValueError(f"{symbol!r} is not in this pool")

        self.fee_bps = fee_bps
        self.fee_factor = 1.0 - fee_bps / 10_000
        current = pool.sqrt_p
        zfo = self.zero_for_one

        if zfo:
            usable = [seg for seg in pool.segments if seg.sqrt_p_lower < current]
        else:
            usable = [seg for seg in pool.segments if seg.sqrt_p_upper > current]
        usable.sort(key=lambda s: s.sqrt_p_lower, reverse=zfo)

        self.near: List[float] = []
        self.edges: List[float] = []
        self.liquidity: List[float] = []
        self.cum: List[float] = []

        total = 0.0
        for seg in usable:
            if zfo and current > seg.sqrt_p_upper:
                current = seg.sqrt_p_upper           # free fall through a gap
            elif not zfo and current < seg.sqrt_p_lower:
                current = seg.sqrt_p_lower
            if seg.liquidity <= 0:
                break
            far = seg.sqrt_p_lower if zfo else seg.sqrt_p_upper
            if zfo:
                amount = seg.liquidity * (1.0 / far - 1.0 / current)
            else:
                amount = seg.liquidity * (far - current)
            amount /= self.fee_factor
            self.near.append(current)
            self.edges.append(far)
            self.liquidity.append(seg.liquidity)
            self.cum.append(total)
            total += amount
            current = far
        self.total = min(total, self.reserve_cap)

    def input_to_reach_sqrt_p(self, target: float) -> float:
        """Input needed to move this pool's price to `target`; capacity if it
        cannot get that far."""
        if not self.edges:
            return 0.0
        zfo = self.zero_for_one
        if zfo:
            if target >= self.near[0]:
                return 0.0
            if target <= self.edges[-1]:
                return self.total
            # edges descend; find the first edge strictly below target
            lo, hi = 0, len(self.edges) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self.edges[mid] <= target:
                    hi = mid
                else:
                    lo = mid + 1
            i = lo
            partial = self.liquidity[i] * (1.0 / target - 1.0 / self.near[i])
        else:
            if target <= self.near[0]:
                return 0.0
            if target >= self.edges[-1]:
                return self.total
            lo, hi = 0, len(self.edges) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self.edges[mid] >= target:
                    hi = mid
                else:
                    lo = mid + 1
            i = lo
            partial = self.liquidity[i] * (target - self.near[i])
        # never promise more than the reserve cap, at any target price
        return min(self.cum[i] + partial / self.fee_factor, self.total)

    def input_to_reach_price(self, collateral_price: float) -> float:
        target_1_per_0 = (collateral_price if self.collateral_is_token0
                          else 1.0 / collateral_price)
        return self.input_to_reach_sqrt_p(sqrt_p_from_price(target_1_per_0))

    def capacity(self) -> float:
        return self.total


@dataclass
class LinearDepthVenue:
    """A linear order book: `depth_usd_per_1pct` of notional moves price 1%.

    Selling Q tokens takes price from P0 to P0 * (1 - Q / Q_full), where
    Q_full is the size that would drive price to zero. Proceeds are the
    integral under that line, i.e. the average of the start and end price.
    """
    name: str
    depth_usd_per_1pct: float
    price: float = 0.0

    @property
    def q_full(self) -> float:
        """Tokens that would drive price to zero on this book."""
        if self.price <= 0:
            return 0.0
        return 100.0 * self.depth_usd_per_1pct / self.price

    def input_to_reach_price(self, target_price: float) -> float:
        if target_price >= self.price or self.price <= 0:
            return 0.0
        return self.q_full * (1.0 - target_price / self.price)

    def capacity(self) -> float:
        return self.q_full

    def proceeds(self, qty: float) -> float:
        """Integral under the linear book -- average of start and end price."""
        if qty <= 0 or self.q_full <= 0:
            return 0.0
        qty = min(qty, self.q_full)
        end_price = self.price * (1.0 - qty / self.q_full)
        return qty * (self.price + end_price) / 2.0

    def apply_sale(self, qty: float) -> None:
        if self.q_full <= 0:
            return
        self.price = max(0.0, self.price * (1.0 - min(qty, self.q_full) / self.q_full))


@dataclass
class VenueFill:
    venue: str
    qty: float
    proceeds: float


@dataclass
class RouteQuote:
    """What routing `amount` across the venue set would do."""
    new_price: float
    amount_out: float               # total proceeds, in numeraire
    amount_in_filled: float
    fills: List[VenueFill] = field(default_factory=list)
    ran_dry: bool = False
    exhausted_data: bool = False
    hit_zero_liquidity: bool = False

    @property
    def stop_reason(self) -> str:
        if not self.ran_dry:
            return "filled"
        return "data_exhausted" if self.exhausted_data else "liquidity_exhausted"


class TwoHopVenue:
    """Sells an asset that has no direct market against the numeraire.

    Liquid staking tokens do not trade against USDC in any depth; they trade
    against WETH. Selling seized wstETH therefore means two hops -- wstETH
    into the wstETH/WETH pool, then that WETH into the WETH venues -- and
    both legs move.

    Modelling this closes a gap that flattered every LST result in the
    project. With no pool for an LST its collateral sold at MARK with zero
    slippage, and since 83% of underwater collateral in the depeg scenarios
    is LST, essentially all of the selling pressure was invisible.

    The second hop is the SHARED WETH venue set, deliberately. An LST
    liquidation ends in WETH being sold for dollars, in the same books a
    WETH liquidation sells into. That is a real contagion channel -- staked
    ether unwinding pushes the ether price, which pushes more positions
    underwater -- and passing the same router object here is what makes the
    model carry it.

    Prices decompose the way the factor model already assumes:

        LST price in USD  =  (LST per WETH, from hop one)  x  (WETH in USD)

    so a market-wide shock moves the WETH leg while a depeg moves the ratio,
    and the two are independent inputs rather than one conflated number.
    """

    def __init__(self, symbol: str, leg: Pool, terminal: "VenueRouter",
                 intermediate: str = "WETH"):
        self.collateral_symbol = symbol
        self.leg = leg                       # symbol/intermediate pool
        self.terminal = terminal             # venues selling intermediate
        self.intermediate = intermediate

    # -- Pool-compatible surface ------------------------------------------
    def _ratio(self) -> float:
        """How much intermediate one unit of the collateral fetches."""
        return self.leg.collateral_price(self.collateral_symbol)

    def collateral_price(self, collateral_symbol: str) -> float:
        self._check(collateral_symbol)
        return self._ratio() * self.terminal.collateral_price(self.intermediate)

    def set_collateral_price(self, collateral_symbol: str, price: float) -> None:
        """Reprice by moving the RATIO, leaving the intermediate alone.

        The caller must shock the intermediate first -- see
        `shock_order` -- because this reads the intermediate's current price
        to work out what ratio the target implies. Shocking WETH afterwards
        would move this asset a second time.
        """
        self._check(collateral_symbol)
        base = self.terminal.collateral_price(self.intermediate)
        if base <= 0:
            raise ValueError(f"{self.intermediate} price is not positive")
        self.leg.set_collateral_price(collateral_symbol, price / base)

    def quote_sell(self, collateral_symbol: str, amount: float) -> RouteQuote:
        self._check(collateral_symbol)
        first = self.leg.quote_sell(collateral_symbol, amount)
        second = self.terminal.quote_sell(self.intermediate, first.amount_out)

        # price after both legs: the new ratio times the new intermediate price
        new_ratio = price_from_sqrt_p(first.new_sqrt_p)
        if collateral_symbol == self.leg.token1_symbol:
            new_ratio = 1.0 / new_ratio if new_ratio else 0.0
        return RouteQuote(
            new_price=new_ratio * second.new_price,
            amount_out=second.amount_out,
            amount_in_filled=min(amount, first.amount_in_filled),
            fills=([VenueFill(f"{collateral_symbol}->{self.intermediate}",
                              first.amount_in_filled, first.amount_out)]
                   + list(second.fills)),
            ran_dry=first.ran_dry or second.ran_dry,
            exhausted_data=first.exhausted_data or second.exhausted_data,
            hit_zero_liquidity=first.hit_zero_liquidity or second.hit_zero_liquidity,
        )

    def sell(self, collateral_symbol: str, amount: float) -> RouteQuote:
        self._check(collateral_symbol)
        quote = self.quote_sell(collateral_symbol, amount)
        first = self.leg.sell(collateral_symbol, amount)
        self.terminal.sell(self.intermediate, first.amount_out)
        return quote

    def sell_capacity(self, collateral_symbol: str) -> float:
        self._check(collateral_symbol)
        return self.leg.sell_capacity(collateral_symbol)

    @property
    def segments(self):
        return list(self.leg.segments)

    @property
    def data_truncated(self) -> bool:
        return self.leg.data_truncated or self.terminal.data_truncated

    def _check(self, symbol: str) -> None:
        if symbol != self.collateral_symbol:
            raise ValueError(f"this venue handles {self.collateral_symbol!r}, "
                             f"not {symbol!r}")

    def describe(self) -> str:
        return (f"two-hop[{self.collateral_symbol} -> {self.intermediate} -> "
                f"numeraire]: {self.leg.fee_bps:.0f}bp leg, then "
                f"{self.terminal.describe()}")


def shock_order(pools: Dict[str, object]) -> List[str]:
    """Assets in the order their prices must be set.

    A two-hop venue prices its asset as (ratio x intermediate), so the
    intermediate has to be shocked BEFORE anything quoted through it --
    otherwise the ratio is computed against a stale base and the asset moves
    twice. Direct venues first, two-hop venues after.
    """
    direct = [a for a, v in pools.items() if not isinstance(v, TwoHopVenue)]
    hopped = [a for a, v in pools.items() if isinstance(v, TwoHopVenue)]
    return direct + hopped


class VenueRouter:
    """A set of venues that behaves like a single `Pool`.

    Deliberately quacks like `Pool` -- `collateral_price`, `set_collateral_price`,
    `quote_sell`, `sell` -- so it drops into `run_cascade`'s `pools` dict with
    no change to the cascade engine.
    """

    def __init__(self, collateral_symbol: str,
                 pools: Optional[List[Pool]] = None,
                 linear_venues: Optional[List[LinearDepthVenue]] = None):
        self.collateral_symbol = collateral_symbol
        self.pools: List[Pool] = list(pools or [])
        self.linear: List[LinearDepthVenue] = list(linear_venues or [])
        if not self.pools and not self.linear:
            raise ValueError("a router needs at least one venue")

        # arbitrage holds venues together, so they all start at one price --
        # taken from the first AMM, which is the one with a real fetched book
        start = (self.pools[0].collateral_price(collateral_symbol)
                 if self.pools else self.linear[0].price)
        for venue in self.linear:
            if venue.price <= 0:
                venue.price = start

    # -- Pool-compatible surface ------------------------------------------
    def collateral_price(self, collateral_symbol: str) -> float:
        self._check(collateral_symbol)
        if self.pools:
            return self.pools[0].collateral_price(collateral_symbol)
        return self.linear[0].price

    def set_collateral_price(self, collateral_symbol: str, price: float) -> None:
        """An exogenous shock reprices the whole market, not one venue."""
        self._check(collateral_symbol)
        for pool in self.pools:
            pool.set_collateral_price(collateral_symbol, price)
        for venue in self.linear:
            venue.price = price

    def quote_sell(self, collateral_symbol: str, amount: float) -> RouteQuote:
        self._check(collateral_symbol)
        return self._route(amount)

    def sell(self, collateral_symbol: str, amount: float) -> RouteQuote:
        self._check(collateral_symbol)
        quote = self._route(amount)
        by_name = {f.venue: f.qty for f in quote.fills}
        for i, pool in enumerate(self.pools):
            qty = by_name.get(self._pool_name(i, pool), 0.0)
            if qty > 0:
                pool.sell(collateral_symbol, qty)
        for venue in self.linear:
            qty = by_name.get(venue.name, 0.0)
            if qty > 0:
                venue.apply_sale(qty)
        return quote

    @property
    def segments(self):
        """Present so anything inspecting a Pool's book still works."""
        return [s for pool in self.pools for s in pool.segments]

    @property
    def data_truncated(self) -> bool:
        return any(p.data_truncated for p in self.pools)

    # -- routing ----------------------------------------------------------
    def _pool_name(self, index: int, pool: Pool) -> str:
        return f"amm{index}:{pool.token0_symbol}/{pool.token1_symbol}"

    def _ladders(self) -> List[_PoolLadder]:
        return [_PoolLadder(pool, self.collateral_symbol) for pool in self.pools]

    @staticmethod
    def _venue_target(marginal_price: float, fee_bps: float) -> float:
        """The pool price at which this venue's marginal proceeds equal
        `marginal_price`."""
        factor = 1.0 - fee_bps / 10_000.0
        return marginal_price / factor if factor > 0 else float("inf")

    def _input_at_price(self, ladders: List[_PoolLadder],
                        marginal_price: float) -> float:
        total = 0.0
        for ladder in ladders:
            total += ladder.input_to_reach_price(
                self._venue_target(marginal_price, ladder.fee_bps))
        for venue in self.linear:
            # the off-chain book is quoted net of costs already
            total += venue.input_to_reach_price(marginal_price)
        return total

    def _route(self, amount: float) -> RouteQuote:
        start_price = self.collateral_price(self.collateral_symbol)
        if amount <= 0:
            return RouteQuote(new_price=start_price, amount_out=0.0,
                              amount_in_filled=0.0)

        ladders = self._ladders()
        capacity = sum(l.capacity() for l in ladders) + \
            sum(v.capacity() for v in self.linear)
        ran_dry = amount > capacity
        target_amount = min(amount, capacity)

        # bisect for the common MARGINAL price (post-fee proceeds per unit)
        lo, hi = 1e-12 * start_price, start_price
        for _ in range(BISECTION_STEPS):
            mid = (lo + hi) / 2.0
            if self._input_at_price(ladders, mid) >= target_amount:
                lo = mid          # mid is deep enough; try shallower
            else:
                hi = mid
        clearing_price = (lo + hi) / 2.0

        fills: List[VenueFill] = []
        proceeds = 0.0
        filled = 0.0
        exhausted_data = False
        hit_zero = False

        for i, (pool, ladder) in enumerate(zip(self.pools, ladders)):
            need = ladder.input_to_reach_price(
                self._venue_target(clearing_price, ladder.fee_bps))
            if need >= ladder.capacity() * (1 - 1e-12) and ran_dry:
                exhausted_data = True
            if need <= 0:
                continue
            result = pool.quote_sell(self.collateral_symbol, need)
            fills.append(VenueFill(self._pool_name(i, pool), need, result.amount_out))
            proceeds += result.amount_out
            filled += result.amount_in_filled
            exhausted_data = exhausted_data or (result.ran_dry and result.exhausted_data)
            hit_zero = hit_zero or result.hit_zero_liquidity

        for venue in self.linear:
            qty = min(venue.input_to_reach_price(clearing_price), venue.capacity())
            if qty <= 0:
                continue
            fills.append(VenueFill(venue.name, qty, venue.proceeds(qty)))
            proceeds += venue.proceeds(qty)
            filled += qty

        return RouteQuote(new_price=clearing_price, amount_out=proceeds,
                          amount_in_filled=filled, fills=fills, ran_dry=ran_dry,
                          exhausted_data=ran_dry and exhausted_data,
                          hit_zero_liquidity=hit_zero)

    def _check(self, symbol: str) -> None:
        if symbol != self.collateral_symbol:
            raise ValueError(f"this router handles {self.collateral_symbol!r}, "
                             f"not {symbol!r}")

    def describe(self) -> str:
        bits = [f"{len(self.pools)} AMM pool(s)"]
        for venue in self.linear:
            bits.append(f"{venue.name} (${venue.depth_usd_per_1pct/1e6:,.0f}M per 1%)")
        return f"router[{self.collateral_symbol}]: " + ", ".join(bits)
