"""
Uniswap v3 concentrated-liquidity swap math.

Core idea: within a single tick range, Uniswap v3 behaves like a constant-product
AMM on *virtual* reserves defined by the liquidity L and sqrt(price):

    virtual_x = L / sqrt(P)      (token0 reserve)
    virtual_y = L * sqrt(P)      (token1 reserve)
    virtual_x * virtual_y = L^2  (constant, within the tick range)

Selling token0 for token1 (zero_for_one=True) increases virtual_x, so price
(P = token1/token0) falls. Selling token1 for token0 does the opposite.

If a swap is big enough to exhaust the liquidity available in the current tick
range, execution has to "cross" into the next tick, where liquidity changes
(possibly to zero -> infinite slippage, i.e. the pool runs dry on that side).
`swap_multi_tick` walks across as many ticks as needed, exactly like the real
protocol does, using a supplied liquidity-per-tick-range distribution.
"""

from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Single tick-range swap (exact Uniswap v3 formula)
# ---------------------------------------------------------------------------

def swap_within_range(sqrt_p: float, liquidity: float, amount_in: float,
                       zero_for_one: bool, fee_bps: float = 30.0
                       ) -> Tuple[float, float]:
    """
    Execute (part of) a swap assuming liquidity stays constant (single tick
    range). Returns (new_sqrt_p, amount_out).

    sqrt_p       : current sqrt(price), price = token1 per token0
    liquidity    : active liquidity L in this range
    amount_in    : amount of the input token being sold (pre-fee)
    zero_for_one : True = selling token0 for token1 (price falls)
    fee_bps      : pool fee in basis points (30 = 0.30%, Uniswap's common tier)
    """
    if liquidity <= 0 or amount_in <= 0:
        return sqrt_p, 0.0

    amount_in_after_fee = amount_in * (1 - fee_bps / 10_000)

    if zero_for_one:
        # x_new = x_old + dx  =>  sqrtP_new = L / (L/sqrtP_old + dx)
        new_sqrt_p = liquidity / (liquidity / sqrt_p + amount_in_after_fee)
        amount_out = liquidity * (sqrt_p - new_sqrt_p)          # token1 out
    else:
        # y_new = y_old + dy  =>  sqrtP_new = sqrtP_old + dy/L
        new_sqrt_p = sqrt_p + amount_in_after_fee / liquidity
        amount_out = liquidity * (1 / sqrt_p - 1 / new_sqrt_p)  # token0 out

    return new_sqrt_p, amount_out


# ---------------------------------------------------------------------------
# Multi-tick swap: walk across liquidity segments until amount_in is spent
# ---------------------------------------------------------------------------

@dataclass
class TickSegment:
    """One contiguous liquidity range, bounded by two sqrt(price) values."""
    sqrt_p_lower: float
    sqrt_p_upper: float
    liquidity: float


@dataclass
class SwapResult:
    new_sqrt_p: float
    amount_out: float
    amount_in_filled: float      # may be < amount_in if liquidity ran out
    ticks_crossed: int
    ran_dry: bool                # True if the swap could not be filled in full

    # WHY it could not be filled. These are very different claims and
    # conflating them is how a data-coverage artifact gets published as an
    # economic finding:
    #
    #   exhausted_data      -- the walk ran off the end of the tick data we
    #                          FETCHED. Says nothing about the real pool;
    #                          the fix is a wider tick_window or more pages.
    #   hit_zero_liquidity  -- the walk reached a price range that genuinely
    #                          has no liquidity in it. This IS a real fact
    #                          about the pool.
    exhausted_data: bool = False
    hit_zero_liquidity: bool = False

    @property
    def stop_reason(self) -> str:
        if not self.ran_dry:
            return "filled"
        if self.hit_zero_liquidity:
            return "liquidity_exhausted"
        return "data_exhausted"


def swap_multi_tick(sqrt_p: float, segments: List[TickSegment], amount_in: float,
                     zero_for_one: bool, fee_bps: float = 30.0) -> SwapResult:
    """
    Walk across tick segments (in the direction the price is moving),
    applying `swap_within_range` on each segment's liquidity until amount_in
    is exhausted or liquidity runs out.

    `segments` must be sorted ascending by sqrt_p, and cover the tick range
    starting at the segment containing `sqrt_p`.
    """
    remaining = amount_in
    current_sqrt_p = sqrt_p
    total_out = 0.0
    ticks_crossed = 0
    ran_dry = False
    hit_zero_liquidity = False

    # Residual input below this counts as fully filled.
    #
    # BUG HISTORY: this used to be a literal `remaining <= 0`, and the
    # difference mattered enormously. Summing `in_to_boundary_gross` across
    # segments never lands on the total exactly, so a trade sized at exactly
    # the book's capacity finishes with float dust (~1e-12) still "remaining".
    # Non-zero dust kept the walk alive for one more iteration, where the
    # gap-crossing snap below teleported the price to the near edge of the
    # next liquidity range -- a free fall across an empty span, bought with
    # 1e-12 tokens of flow.
    #
    # On the live weETH/WETH book that put a cliff at exactly `sell_capacity`:
    # selling 95% of capacity moved the price -0.25%, selling 100% moved it
    # -24.47%. In the cascade, weETH flow saturates at capacity, so whether a
    # scenario touched the cliff or stopped ten tokens short of it (4,327 vs
    # 4,337) decided whether every weETH-collateralised account got marked
    # down a quarter. That is what made bad debt NON-MONOTONIC in shock size:
    # $373.28M at -30% against $32.34M at -45%, from a boundary artifact
    # rather than any economic mechanism.
    #
    # Scaling with `amount_in` is the point -- an absolute floor cannot serve
    # both a 1e-3 WBTC trade and a 1e6 USDC one. Covered by
    # test_book_truncation.py::test_selling_exactly_capacity_does_not_free_fall.
    fill_eps = max(1e-12, abs(amount_in) * 1e-12)

    # Order segments in the direction of travel and cut the walk at the end of
    # real book -- see `usable_segments`. `data_tail` records that ranges were
    # dropped, so consuming everything here means we reached the edge of our
    # DATA and the resulting price is not a mark anyone should trust.
    ordered, data_tail = usable_segments(sqrt_p, segments, zero_for_one)

    for seg in ordered:
        if remaining <= fill_eps:
            remaining = 0.0
            break

        # Keep every segment that lies -- even partly -- in the direction of
        # travel. Crucially this INCLUDES the active segment, the one
        # containing current_sqrt_p, whose liquidity the trade consumes first.
        #
        # BUG HISTORY: this test used to compare the segment's NEAR boundary
        # against the current price (`seg.sqrt_p_upper > current_sqrt_p`),
        # which skipped the active range whenever the price sat strictly
        # inside it. At construction the price sits exactly ON a boundary so
        # the old test passed by a hair, but `Pool.set_collateral_price()`
        # teleports the price mid-segment -- i.e. on every single cascade run
        # -- after which the active range was dropped and the walk integrated
        # the NEXT segment's liquidity across the skipped span. On a
        # concentrated book that overstated price impact by ~12x. Covered by
        # test_uniswap_v3_math.py::test_active_segment_is_consumed_first.
        if zero_for_one and seg.sqrt_p_lower >= current_sqrt_p:
            continue  # entirely at/above current price -- irrelevant selling token0
        if not zero_for_one and seg.sqrt_p_upper <= current_sqrt_p:
            continue  # entirely at/below current price -- irrelevant selling token1

        # If the price sits in a GAP between segments (a range with no
        # liquidity at all), it crosses that gap for free: zero liquidity
        # means zero input is needed to traverse it. Snap to the segment's
        # near edge before quoting, rather than integrating this segment's
        # liquidity across a span it doesn't actually cover.
        if zero_for_one and current_sqrt_p > seg.sqrt_p_upper:
            current_sqrt_p = seg.sqrt_p_upper
        elif not zero_for_one and current_sqrt_p < seg.sqrt_p_lower:
            current_sqrt_p = seg.sqrt_p_lower

        boundary = seg.sqrt_p_lower if zero_for_one else seg.sqrt_p_upper

        # how much input would it take to push price to this segment's boundary?
        if zero_for_one:
            in_to_boundary = seg.liquidity * (1 / boundary - 1 / current_sqrt_p)
        else:
            in_to_boundary = seg.liquidity * (boundary - current_sqrt_p)
        in_to_boundary_gross = in_to_boundary / (1 - fee_bps / 10_000) if in_to_boundary > 0 else 0.0

        if remaining <= in_to_boundary_gross or seg.liquidity == 0:
            # this segment absorbs the rest of the trade (or has no liquidity)
            if seg.liquidity == 0:
                ran_dry = True
                hit_zero_liquidity = True
                break
            new_sqrt_p, out = swap_within_range(current_sqrt_p, seg.liquidity,
                                                 remaining, zero_for_one, fee_bps)
            total_out += out
            current_sqrt_p = new_sqrt_p
            remaining = 0.0
        else:
            # consume this whole segment, cross into the next tick
            total_out += swap_within_range(current_sqrt_p, seg.liquidity,
                                            in_to_boundary_gross, zero_for_one,
                                            fee_bps)[1]
            current_sqrt_p = boundary
            remaining -= in_to_boundary_gross
            ticks_crossed += 1

    exhausted_data = False
    if remaining > fill_eps:
        ran_dry = True
        # we ran out of SEGMENTS, not necessarily out of pool liquidity
        exhausted_data = not hit_zero_liquidity
    elif data_tail and ordered and current_sqrt_p == (
            ordered[-1].sqrt_p_lower if zero_for_one else ordered[-1].sqrt_p_upper):
        # Filled, but only by consuming the book to its last real range while
        # a dropped tail lies beyond. The fill is honest; the resting price is
        # the edge of our data, so flag it rather than letting a caller mark a
        # portfolio to it.
        ran_dry = True
        exhausted_data = True

    return SwapResult(new_sqrt_p=current_sqrt_p, amount_out=total_out,
                       amount_in_filled=amount_in - remaining,
                       ticks_crossed=ticks_crossed, ran_dry=ran_dry,
                       exhausted_data=exhausted_data,
                       hit_zero_liquidity=hit_zero_liquidity)


def input_to_reach_sqrt_p(sqrt_p: float, segments: List[TickSegment],
                          target_sqrt_p: float, zero_for_one: bool,
                          fee_bps: float = 30.0) -> float:
    """How much input it takes to push this pool's price exactly to
    `target_sqrt_p`. Returns inf if the fetched book runs out first.

    This is the inverse of `swap_multi_tick` -- price in, size out -- and it
    is what makes optimal routing across venues tractable: arbitrage holds
    every venue at the same price, so routing a large sale means finding the
    one common price all venues end up at, then asking each venue how much it
    absorbed getting there.
    """
    if (zero_for_one and target_sqrt_p >= sqrt_p) or \
       (not zero_for_one and target_sqrt_p <= sqrt_p):
        return 0.0

    total = 0.0
    current = sqrt_p
    ordered = sorted(segments, key=lambda s: s.sqrt_p_lower, reverse=zero_for_one)

    for seg in ordered:
        if zero_for_one and seg.sqrt_p_lower >= current:
            continue
        if not zero_for_one and seg.sqrt_p_upper <= current:
            continue
        if zero_for_one and current > seg.sqrt_p_upper:
            current = seg.sqrt_p_upper        # free fall through an empty gap
        elif not zero_for_one and current < seg.sqrt_p_lower:
            current = seg.sqrt_p_lower
        if zero_for_one and current <= target_sqrt_p:
            break
        if not zero_for_one and current >= target_sqrt_p:
            break
        if seg.liquidity <= 0:
            return float("inf")               # no depth here; cannot get there

        boundary = seg.sqrt_p_lower if zero_for_one else seg.sqrt_p_upper
        stop = max(boundary, target_sqrt_p) if zero_for_one \
            else min(boundary, target_sqrt_p)
        if zero_for_one:
            amount = seg.liquidity * (1 / stop - 1 / current)
        else:
            amount = seg.liquidity * (stop - current)
        total += amount / (1 - fee_bps / 10_000)
        current = stop
        if (zero_for_one and current <= target_sqrt_p) or \
           (not zero_for_one and current >= target_sqrt_p):
            return total

    return float("inf")                        # ran out of book before target


#: Fraction of nominal depth deliberately ignored at the far end of a book.
#: The trailing ranges that make up this last sliver are too thin to trade but
#: span a wide price band, so including them makes price impact discontinuous
#: at full capacity. See `usable_segments`.
IGNORED_TAIL_DEPTH_FRACTION = 1e-3


def usable_segments(sqrt_p: float, segments: List[TickSegment],
                    zero_for_one: bool) -> Tuple[List[TickSegment], bool]:
    """Segments in the direction of travel, truncated at the end of real book.

    Returns `(ordered, truncated)`. `ordered` runs in the direction the price
    moves and stops after the last range holding meaningful liquidity;
    `truncated` says whether anything was dropped, i.e. whether consuming all
    of `ordered` means we hit the edge of our DATA rather than the edge of the
    pool.

    WHY THIS EXISTS. A Messari/Uniswap subgraph response does not always carry
    every boundary tick, and `liquidityNet` accumulated over a response with
    holes decays toward zero a range or two past the active one. The result is
    a book with a couple of genuine ranges followed by a long tail of ranges
    holding effectively nothing -- which is not a thin market, it is missing
    data wearing a thin market's clothes.

    The tail is dangerous precisely because it is empty: crossing a range with
    no liquidity costs no input, so a trade can traverse dozens of them for
    free and come to rest at a price far from anywhere real. Measured on the
    live weETH/WETH 5bp book: two ranges at L=4,790,207, then ~85 ranges at
    L between 0.03 and 0.66 spanning 24% of price. Selling 99.9% of the
    reported capacity moved the ratio -0.19%; selling 100% of it -- the last
    0.04 weETH -- crossed 84 further ticks and moved it -24.41%.

    That discontinuity drove the headline result. weETH flow in the cascade
    saturates at capacity, so whether a scenario landed exactly on the cliff
    or a few tokens short of it decided whether every weETH-collateralised
    account was marked down a quarter. Bad debt came out NON-MONOTONIC in
    shock size -- $373.28M at -30% against $32.34M at -45% -- with the whole
    $341M gap owed to this artifact. Worse, the mark applied to positions
    hundreds of times larger than the book: the largest account holds 401,299
    weETH against a reported capacity of 4,337.

    WHERE TO CUT. Not at a liquidity threshold. The obvious approach -- drop
    ranges thinner than some fraction of the deepest one -- needs a different
    fraction for every book, which is curve-fitting. Measured over the four
    live LST legs, the count of ranges above 1e-4 of the deepest was 2 for
    weETH, 6 for rETH, 222 for cbETH and 1069 for wstETH. No single constant
    separates a real book from a truncated one.

    So the cut is made in INPUT units, which are self-normalising: rank the
    ranges by depth and keep those holding `1 - IGNORED_TAIL_DEPTH_FRACTION`
    of the book's absorbable input (`_liquidity_floor`). What gets dropped is
    by construction a sliver of depth -- at most 0.1% -- spread over however
    many ranges it takes, which is exactly the pathology: those ranges cost
    nothing to cross but carry the price a long way. It also makes impact
    continuous in trade size, the property that actually matters: a trade at
    100% of capacity now lands near one at 99.9%, rather than 24 percentage
    points below it.

    The floor comes from the fetched book, not from the live price, so that
    consuming the book does not redefine what counts as book -- see the bug
    history in `_liquidity_floor`.

    Fees are ignored in the accumulation. They scale every range by the same
    factor, so they cancel out of the comparison.

    Note this REPLACES an earlier `liquidity <= 0` test. That test caught only
    exact zeros, which the truncation tail rarely produces -- the accumulated
    values here are 0.028 and 0.66, not 0. It also stopped at the first empty
    range, which was wrong in the other direction: a real book can have an
    empty range with genuine depth beyond it, and an rETH/WETH pool with one
    empty range at index 149 reported a capacity of zero, failing every rETH
    liquidation and manufacturing bad debt. Truncating by cumulative depth
    handles both: interior gaps are kept and crossed for free, while the dead
    tail past real depth is dropped.
    """
    ordered = sorted(segments, key=lambda s: s.sqrt_p_lower, reverse=zero_for_one)
    if zero_for_one:
        ahead = [s for s in ordered if s.sqrt_p_lower < sqrt_p]
    else:
        ahead = [s for s in ordered if s.sqrt_p_upper > sqrt_p]
    if not ahead:
        return [], False

    floor = _liquidity_floor(segments, zero_for_one)
    kept = [s for s in ahead if s.liquidity >= floor]
    return kept, len(kept) < len(ahead)


def _liquidity_floor(segments: List[TickSegment], zero_for_one: bool) -> float:
    """The least liquidity a range must hold to count as real book.

    Derived from the fetched segments alone -- deliberately NOT from the
    current price. Ranking the ranges by depth and cutting at a fixed
    cumulative share gives a floor that is a property of the BOOK, so it does
    not move as the price moves.

    BUG HISTORY: the first version of this cut positionally instead, keeping
    the shortest prefix of ranges ahead of the price that held 99.9% of the
    depth ahead. Both those quantities are measured from the live price, which
    made the cut ratchet: once a sale consumed the real ranges, the dead tail
    was all that lay ahead, so the tail became "the book" and the next sale
    promoted the next sliver of it. Selling the weETH leg's full capacity three
    times running gave -0.19%, then -5.72%, then -24.41% -- and the third sale
    moved the price 19 points while filling ZERO tokens, because a nominal
    capacity of ~1e-9 bought a free fall across the whole tail. Anchoring on
    the fetched book removes the ratchet by construction. Covered by
    test_book_truncation.py::test_repeated_capacity_sales_do_not_ratchet.
    """
    weighted = []
    for seg in segments:
        lo, hi = seg.sqrt_p_lower, seg.sqrt_p_upper
        if seg.liquidity <= 0 or lo <= 0 or hi <= lo:
            continue
        # input this range absorbs across its full width, in the traded token
        width = (1.0 / lo - 1.0 / hi) if zero_for_one else (hi - lo)
        weighted.append((seg.liquidity, seg.liquidity * width))
    if not weighted:
        return 0.0

    total = sum(w for _, w in weighted)
    if total <= 0:
        return 0.0

    # Deepest first, so the cut keeps the ranges that hold the depth and the
    # floor lands just under the thinnest of them.
    weighted.sort(key=lambda t: -t[0])
    keep_target = total * (1.0 - IGNORED_TAIL_DEPTH_FRACTION)
    running = 0.0
    for liquidity, w in weighted:
        running += w
        if running >= keep_target:
            return liquidity
    return weighted[-1][0]


def max_input_available(sqrt_p: float, segments: List[TickSegment],
                        zero_for_one: bool, fee_bps: float = 30.0) -> float:
    """Total input this book can absorb before it runs out.

    Walks the ranges holding real liquidity (`usable_segments` decides which)
    and accumulates. Interior gaps are crossed for free; the sparse tail past
    the last genuine range is not counted, because it holds no depth to sell
    into and counting it put a 24% discontinuity at exactly this function's
    return value.

    This exists because asking `input_to_reach_sqrt_p` for the very bottom of
    the book answers "infinity" whenever a zero-liquidity range sits anywhere
    in between, and a caller reading that as a capacity of ZERO gets the
    opposite of the truth. A real rETH/WETH pool with usable depth at the top
    and one empty range at index 149 reported a capacity of zero, which made
    every rETH liquidation in the model fail and manufactured bad debt out of
    nothing.
    """
    total = 0.0
    current = sqrt_p
    ordered, _ = usable_segments(sqrt_p, segments, zero_for_one)

    for seg in ordered:
        if zero_for_one and seg.sqrt_p_lower >= current:
            continue
        if not zero_for_one and seg.sqrt_p_upper <= current:
            continue
        if zero_for_one and current > seg.sqrt_p_upper:
            current = seg.sqrt_p_upper
        elif not zero_for_one and current < seg.sqrt_p_lower:
            current = seg.sqrt_p_lower
        if seg.liquidity <= 0:
            continue                    # interior gap: crossed for free
        far = seg.sqrt_p_lower if zero_for_one else seg.sqrt_p_upper
        if zero_for_one:
            amount = seg.liquidity * (1.0 / far - 1.0 / current)
        else:
            amount = seg.liquidity * (far - current)
        total += amount / (1 - fee_bps / 10_000)
        current = far
    return total


def price_from_sqrt_p(sqrt_p: float) -> float:
    return sqrt_p ** 2


def sqrt_p_from_price(price: float) -> float:
    return price ** 0.5