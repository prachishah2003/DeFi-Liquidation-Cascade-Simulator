"""
Tests for multi-venue routing.

The property that matters: splitting a sale across venues must produce
strictly less price impact than forcing all of it through one, and the split
must be the one arbitrage would enforce -- every venue ending at the same
price. If those two hold, the router is doing an aggregator's job rather than
a plausible-looking approximation of it.
"""

from uniswap_v3_math import TickSegment, sqrt_p_from_price
from cascade_sim import Pool
from routing import VenueRouter, LinearDepthVenue

MID = 2800.0


def book(mid, liquidity, steps=60, step=0.01):
    segs, price = [], mid
    for _ in range(steps):
        nxt = price * (1 - step)
        segs.append(TickSegment(sqrt_p_from_price(nxt), sqrt_p_from_price(price),
                                liquidity))
        price = nxt
    return segs


def pool(liquidity, mid=MID):
    return Pool("WETH", "USDC", sqrt_p_from_price(mid), book(mid, liquidity))


def test_single_pool_router_matches_the_bare_pool():
    """With one venue and no off-chain depth, routing must be a no-op."""
    direct = pool(30_000_000)
    routed = VenueRouter("WETH", pools=[pool(30_000_000)])

    qty = 4_000.0
    direct.sell("WETH", qty)
    routed.sell("WETH", qty)
    a, b = direct.collateral_price("WETH"), routed.collateral_price("WETH")
    print(f"  bare pool ${a:,.2f} vs single-venue router ${b:,.2f}")
    assert abs(a - b) / a < 1e-4, "routing changed a one-venue result"


def test_splitting_across_pools_reduces_impact():
    qty = 8_000.0
    one = VenueRouter("WETH", pools=[pool(20_000_000)])
    three = VenueRouter("WETH", pools=[pool(20_000_000), pool(20_000_000),
                                       pool(20_000_000)])
    one.sell("WETH", qty)
    three.sell("WETH", qty)
    p1, p3 = one.collateral_price("WETH"), three.collateral_price("WETH")
    print(f"  {qty:,.0f} WETH into 1 pool -> ${p1:,.2f} "
          f"({(p1/MID-1)*100:+.2f}%)")
    print(f"  {qty:,.0f} WETH into 3 pools -> ${p3:,.2f} "
          f"({(p3/MID-1)*100:+.2f}%)")
    assert p3 > p1, "splitting across venues must reduce impact"


def test_every_venue_ends_at_the_same_price():
    """The arbitrage condition the routing is derived from."""
    pools = [pool(30_000_000), pool(8_000_000), pool(50_000_000)]
    router = VenueRouter("WETH", pools=pools)
    router.sell("WETH", 10_000.0)
    prices = [p.collateral_price("WETH") for p in pools]
    spread = (max(prices) - min(prices)) / max(prices)
    print("  venue prices after routing: "
          + ", ".join(f"${p:,.2f}" for p in prices))
    assert spread < 1e-3, f"venues diverged by {spread:.2%} -- not an arb split"


def test_deeper_venue_absorbs_more():
    deep, thin = pool(60_000_000), pool(6_000_000)
    router = VenueRouter("WETH", pools=[deep, thin])
    quote = router.quote_sell("WETH", 9_000.0)
    by_qty = {f.venue: f.qty for f in quote.fills}
    qtys = list(by_qty.values())
    print("  fills: " + ", ".join(f"{k}={v:,.0f}" for k, v in by_qty.items()))
    assert max(qtys) > min(qtys) * 3, (
        "a 10x deeper venue should absorb far more of the order")


def test_off_chain_depth_absorbs_most_of_a_large_order():
    """The correction that matters most: a 0.3% pool is a small share of real
    WETH depth, so most of a large liquidation never touches it."""
    amm = pool(20_000_000)
    router = VenueRouter("WETH", pools=[amm],
                         linear_venues=[LinearDepthVenue("cex", 40_000_000)])
    quote = router.quote_sell("WETH", 20_000.0)
    fills = {f.venue: f.qty for f in quote.fills}
    offchain_share = fills.get("cex", 0) / sum(fills.values())
    print("  fills: " + ", ".join(f"{k}={v:,.0f}" for k, v in fills.items())
          + f"  (off-chain {offchain_share:.0%})")
    assert offchain_share > 0.5, "deep off-chain book should take the bulk"


def test_off_chain_depth_cuts_modelled_impact_sharply():
    qty = 20_000.0
    onchain_only = VenueRouter("WETH", pools=[pool(20_000_000)])
    with_cex = VenueRouter("WETH", pools=[pool(20_000_000)],
                           linear_venues=[LinearDepthVenue("cex", 40_000_000)])
    onchain_only.sell("WETH", qty)
    with_cex.sell("WETH", qty)
    a = onchain_only.collateral_price("WETH")
    b = with_cex.collateral_price("WETH")
    print(f"  one pool only : ${a:,.2f} ({(a/MID-1)*100:+.1f}%)")
    print(f"  pool + off-chain: ${b:,.2f} ({(b/MID-1)*100:+.1f}%)")
    assert b > a
    print(f"  -> ignoring off-chain depth overstates the move by "
          f"{((MID-a)/(MID-b) if MID > b else float('inf')):.1f}x")


def test_shock_repricies_every_venue():
    amm = pool(20_000_000)
    cex = LinearDepthVenue("cex", 40_000_000)
    router = VenueRouter("WETH", pools=[amm], linear_venues=[cex])
    router.set_collateral_price("WETH", MID * 0.85)
    assert abs(amm.collateral_price("WETH") - MID * 0.85) / MID < 1e-6
    assert abs(cex.price - MID * 0.85) / MID < 1e-6
    print(f"  exogenous shock repriced both venues to ${cex.price:,.2f}")


def test_router_reports_running_dry():
    router = VenueRouter("WETH", pools=[pool(1_000_000, MID)])
    quote = router.quote_sell("WETH", 5_000_000.0)
    print(f"  huge order: filled {quote.amount_in_filled:,.0f}, "
          f"ran_dry={quote.ran_dry}, reason={quote.stop_reason}")
    assert quote.ran_dry and quote.amount_in_filled < 5_000_000.0


def test_linear_venue_proceeds_are_the_integral():
    """Selling into a linear book must realise the average of start and end
    price, not the start price -- otherwise slippage is free."""
    venue = LinearDepthVenue("cex", 10_000_000, price=MID)
    qty = venue.q_full * 0.5                     # drives price to half
    proceeds = venue.proceeds(qty)
    expected = qty * (MID + MID * 0.5) / 2.0
    print(f"  proceeds ${proceeds:,.0f} vs closed form ${expected:,.0f}")
    assert abs(proceeds - expected) / expected < 1e-9


def test_pool_fee_is_respected_not_assumed():
    """Uniswap runs a separate pool per fee tier and they charge very
    different amounts. The engine defaulted to 30bp everywhere, so the 0.05%
    tier -- the deepest for WETH/USDC, and the one that takes most routed
    flow -- was charged six times its real fee."""
    cheap = pool(20_000_000)
    cheap.fee_bps = 5.0
    dear = pool(20_000_000)
    dear.fee_bps = 100.0

    qty = 2_000.0
    cheap_out = cheap.quote_sell("WETH", qty).amount_out
    dear_out = dear.quote_sell("WETH", qty).amount_out
    print(f"  same book, 5bp fee -> ${cheap_out:,.0f};  100bp fee -> ${dear_out:,.0f}")
    assert cheap_out > dear_out, "a cheaper pool must return more for the same sale"

    # roughly the fee difference, not something wilder
    ratio = (cheap_out - dear_out) / dear_out
    assert 0.005 < ratio < 0.02, f"fee difference of {ratio:.3%} is implausible"


def test_router_ladder_uses_each_pools_own_fee():
    cheap = pool(20_000_000)
    cheap.fee_bps = 5.0
    dear = pool(20_000_000)
    dear.fee_bps = 100.0
    router = VenueRouter("WETH", pools=[cheap, dear])
    quote = router.quote_sell("WETH", 4_000.0)
    by_venue = {f.venue: f.qty for f in quote.fills}
    cheap_qty = by_venue["amm0:WETH/USDC"]
    dear_qty = by_venue["amm1:WETH/USDC"]
    print(f"  identical depth, 5bp vs 100bp: cheap pool takes {cheap_qty:,.0f}, "
          f"dear pool {dear_qty:,.0f}")
    assert cheap_qty > dear_qty, (
        "the CHEAP venue must receive more of the order. Equalising post-trade "
        "price rather than fee-adjusted marginal proceeds sends more flow to "
        "the dearer pool -- exactly backwards, and what this test exists to "
        "catch.")


# ---------------------------------------------------------------------------
# two-hop venues: selling an asset that has no direct market
# ---------------------------------------------------------------------------

RATIO = 1.18          # wstETH per... rather, WETH per wstETH


def lst_leg(liquidity=8_000.0, ratio=RATIO):
    """A wstETH/WETH pool. Prices and liquidity are in WETH terms, so the
    numbers are small compared with the USDC pools above."""
    segs, price = [], ratio
    for _ in range(60):
        nxt = price * 0.995
        segs.append(TickSegment(sqrt_p_from_price(nxt), sqrt_p_from_price(price),
                                liquidity))
        price = nxt
    p = Pool("wstETH", "WETH", sqrt_p_from_price(ratio), segs)
    p.fee_bps = 1.0                       # the 0.01% tier, as in reality
    return p


def two_hop(weth_router=None, leg_liquidity=8_000.0):
    from routing import TwoHopVenue
    terminal = weth_router or VenueRouter("WETH", pools=[pool(40_000_000)])
    return TwoHopVenue("wstETH", lst_leg(leg_liquidity), terminal), terminal


def test_two_hop_price_is_ratio_times_base():
    venue, terminal = two_hop()
    expected = RATIO * terminal.collateral_price("WETH")
    got = venue.collateral_price("wstETH")
    print(f"  wstETH = {RATIO} WETH x ${terminal.collateral_price('WETH'):,.2f} "
          f"= ${got:,.2f}")
    assert abs(got - expected) / expected < 1e-9


def test_two_hop_selling_moves_both_legs():
    venue, terminal = two_hop()
    ratio_before = venue.leg.collateral_price("wstETH")
    weth_before = terminal.collateral_price("WETH")
    venue.sell("wstETH", 2_000.0)
    print(f"  ratio {ratio_before:.4f} -> {venue.leg.collateral_price('wstETH'):.4f}; "
          f"WETH ${weth_before:,.2f} -> ${terminal.collateral_price('WETH'):,.2f}")
    assert venue.leg.collateral_price("wstETH") < ratio_before, "hop one must move"
    assert terminal.collateral_price("WETH") < weth_before, (
        "hop two must move -- the WETH received has to be sold too")


def test_lst_liquidation_pushes_the_weth_price():
    """The contagion channel this exists to model: unwinding staked ether
    ends in ether being sold for dollars, in the same books a WETH
    liquidation uses."""
    shared = VenueRouter("WETH", pools=[pool(20_000_000)])
    from routing import TwoHopVenue
    venue = TwoHopVenue("wstETH", lst_leg(), shared)
    before = shared.collateral_price("WETH")
    venue.sell("wstETH", 3_000.0)
    after = shared.collateral_price("WETH")
    print(f"  selling 3,000 wstETH moved WETH ${before:,.2f} -> ${after:,.2f} "
          f"({(after/before-1)*100:+.3f}%)")
    assert after < before


def test_zero_slippage_was_the_old_behaviour():
    """What the model did before: no pool for the LST, so it sold at mark.
    The two-hop venue must be strictly worse, which is the whole point."""
    venue, _terminal = two_hop()
    qty = 2_000.0
    mark = venue.collateral_price("wstETH")
    at_mark = qty * mark
    routed = venue.quote_sell("wstETH", qty).amount_out
    shortfall = (at_mark - routed) / at_mark * 100
    print(f"  {qty:,.0f} wstETH at mark ${at_mark:,.0f} vs routed ${routed:,.0f} "
          f"({shortfall:.2f}% worse)")
    assert routed < at_mark, "routing through two pools cannot beat the mark"
    assert shortfall > 0.1


def test_thin_lst_leg_hurts_more():
    deep, _ = two_hop(leg_liquidity=40_000.0)
    thin, _ = two_hop(leg_liquidity=2_000.0)
    qty = 2_000.0
    deep_out = deep.quote_sell("wstETH", qty).amount_out
    thin_out = thin.quote_sell("wstETH", qty).amount_out
    print(f"  deep LST leg -> ${deep_out:,.0f};  thin LST leg -> ${thin_out:,.0f}")
    assert thin_out < deep_out


def test_repricing_moves_the_ratio_not_the_base():
    """A depeg must move the LST against ETH, leaving ETH alone."""
    venue, terminal = two_hop()
    weth_before = terminal.collateral_price("WETH")
    target = venue.collateral_price("wstETH") * 0.92          # 8% depeg
    venue.set_collateral_price("wstETH", target)
    print(f"  after an 8% depeg: wstETH ${venue.collateral_price('wstETH'):,.2f}, "
          f"WETH ${terminal.collateral_price('WETH'):,.2f} (unchanged)")
    assert abs(venue.collateral_price("wstETH") - target) / target < 1e-9
    assert abs(terminal.collateral_price("WETH") - weth_before) < 1e-9


def test_shock_order_puts_dependents_last():
    from routing import shock_order
    venue, terminal = two_hop()
    pools_map = {"wstETH": venue, "WETH": terminal}
    order = shock_order(pools_map)
    print(f"  shock order: {order}")
    assert order.index("WETH") < order.index("wstETH"), (
        "the intermediate must be repriced before anything quoted through it")


def test_cascade_applies_shocks_in_dependency_order():
    """End to end: shocking both WETH and wstETH must land each at exactly
    its target, not compound one into the other."""
    from routing import TwoHopVenue
    from cascade_sim import run_cascade
    terminal = VenueRouter("WETH", pools=[pool(40_000_000)])
    venue = TwoHopVenue("wstETH", lst_leg(), terminal)
    pools_map = {"WETH": terminal, "wstETH": venue}

    weth0 = terminal.collateral_price("WETH")
    lst0 = venue.collateral_price("wstETH")
    run_cascade([], pools_map, {"WETH": -0.15, "wstETH": -0.23}, verbose=False)

    weth1 = terminal.collateral_price("WETH")
    lst1 = venue.collateral_price("wstETH")
    print(f"  WETH   {(weth1/weth0-1)*100:+.2f}% (target -15.00%)")
    print(f"  wstETH {(lst1/lst0-1)*100:+.2f}% (target -23.00%)")
    assert abs(weth1 / weth0 - 0.85) < 1e-6
    assert abs(lst1 / lst0 - 0.77) < 1e-6, (
        "the LST compounded the WETH shock -- ordering is wrong")


def test_deepcopy_of_the_mapping_preserves_shared_venues():
    """A silent-failure hazard worth pinning.

    Every LST venue holds a reference to the SAME WETH router, which is what
    makes staked-ether liquidations push the ether price. Copying the
    mapping in one deepcopy keeps that sharing through the memo. Copying
    value by value gives each LST its own private WETH book -- the model
    still runs, the numbers still look plausible, and the contagion channel
    has quietly vanished.
    """
    import copy as _copy
    from routing import TwoHopVenue

    shared = VenueRouter("WETH", pools=[pool(20_000_000)])
    venues = {"WETH": shared,
              "wstETH": TwoHopVenue("wstETH", lst_leg(), shared),
              "cbETH": TwoHopVenue("cbETH", lst_leg(), shared)}

    good = _copy.deepcopy(venues)
    assert good["wstETH"].terminal is good["WETH"], "sharing lost in deepcopy"
    assert good["cbETH"].terminal is good["WETH"]

    bad = {k: _copy.deepcopy(v) for k, v in venues.items()}
    assert bad["wstETH"].terminal is not bad["WETH"], (
        "per-key copying unexpectedly preserved sharing -- this test no "
        "longer demonstrates the hazard it exists to document")
    print("  one deepcopy of the mapping keeps the shared WETH router; "
          "per-key copying silently detaches it")

    # and the consequence, measured
    before = good["WETH"].collateral_price("WETH")
    good["wstETH"].sell("wstETH", 2_000.0)
    shared_move = (good["WETH"].collateral_price("WETH") / before - 1) * 100
    before_b = bad["WETH"].collateral_price("WETH")
    bad["wstETH"].sell("wstETH", 2_000.0)
    detached_move = (bad["WETH"].collateral_price("WETH") / before_b - 1) * 100
    print(f"  selling 2,000 wstETH moves the shared WETH book "
          f"{shared_move:+.3f}%, the detached one {detached_move:+.3f}%")
    assert shared_move < 0 and detached_move == 0.0


# ---------------------------------------------------------------------------
# does the reconstructed book agree with what the pool actually holds?
# ---------------------------------------------------------------------------

def test_capacity_survives_an_empty_range_partway_down():
    """A gap in the book must not read as zero capacity.

    A live rETH/WETH pool had real depth at the top and one empty tick range
    at index 149. Asking what it costs to reach the very bottom answers
    infinity, and reading that as zero made the pool look completely
    illiquid -- which failed every rETH liquidation and produced bad debt out
    of nothing.
    """
    segs = book(MID, 5_000_000, steps=10) + [
        TickSegment(sqrt_p_from_price(MID * 0.5), sqrt_p_from_price(MID * 0.88), 0.0)
    ] + book(MID * 0.5, 5_000_000, steps=10)
    p = Pool("WETH", "USDC", sqrt_p_from_price(MID), segs)
    capacity = p.sell_capacity("WETH")
    print(f"  book with an empty range partway down: capacity {capacity:,.0f} WETH")
    assert capacity > 0, "an empty range must not zero out the depth above it"
    filled = p.quote_sell("WETH", capacity * 0.99).amount_in_filled
    assert filled > 0


def test_book_consistency_passes_on_a_sane_pool():
    p = Pool("WETH", "USDC", sqrt_p_from_price(MID), book(MID, 4_000_000))
    capacity = p.sell_capacity("WETH")
    payout = p.quote_sell("WETH", capacity).amount_out
    p.tvl_token0, p.tvl_token1 = capacity * 2, payout * 1.2
    implied, held, ratio = p.book_consistency("WETH")
    print(f"  sane pool: book implies {implied:,.0f} USDC out, pool holds "
          f"{held:,.0f} ({ratio:.2f}x)")
    assert ratio < 1.5


def test_book_consistency_catches_an_inflated_book():
    """The check that would have caught wstETH: a reconstructed book claiming
    it can pay out far more than the pool owns."""
    p = Pool("WETH", "USDC", sqrt_p_from_price(MID), book(MID, 4_000_000))
    capacity = p.sell_capacity("WETH")
    payout = p.quote_sell("WETH", capacity).amount_out
    p.tvl_token0, p.tvl_token1 = capacity, payout / 50.0     # pool holds 50x less
    implied, held, ratio = p.book_consistency("WETH")
    print(f"  inflated book: implies {implied:,.0f} USDC out against "
          f"{held:,.0f} held ({ratio:.0f}x) -- flagged")
    assert ratio > 1.5


def test_book_consistency_is_none_without_holdings():
    p = Pool("WETH", "USDC", sqrt_p_from_price(MID), book(MID, 4_000_000))
    assert p.book_consistency("WETH") is None, (
        "with no reported holdings the check must abstain, not invent a verdict")
    print("  no reported holdings -> no verdict, rather than a false pass")


def test_capacity_is_capped_by_the_pools_reserves():
    """The wstETH failure, in miniature.

    A tick book can claim far more depth than the pool holds when the
    boundary ticks that would collapse its liquidity are missing from the
    fetched data. The pool's reported reserves are ground truth: a swap
    cannot pay out more of the other token than the pool owns.
    """
    p = Pool("WETH", "USDC", sqrt_p_from_price(MID), book(MID, 4_000_000))
    unconstrained = p.book_capacity("WETH")

    # the pool actually holds only enough USDC for a fraction of that
    full_payout = p.quote_sell("WETH", unconstrained).amount_out
    p.tvl_token0, p.tvl_token1 = unconstrained, full_payout / 20.0

    capped = p.sell_capacity("WETH")
    print(f"  book alone claims {unconstrained:,.0f} WETH of depth; "
          f"reserves allow {capped:,.0f}")
    assert capped < unconstrained / 5, "the reserve constraint did not bind"

    payout = p.quote_sell("WETH", capped).amount_out
    print(f"  selling the capped amount pays out {payout:,.0f} USDC against "
          f"{p.tvl_token1:,.0f} held")
    assert payout <= p.tvl_token1 * 1.01, "payout still exceeds the reserves"


def test_router_respects_the_reserve_cap():
    p = Pool("WETH", "USDC", sqrt_p_from_price(MID), book(MID, 4_000_000))
    full = p.quote_sell("WETH", p.book_capacity("WETH")).amount_out
    p.tvl_token0, p.tvl_token1 = p.book_capacity("WETH"), full / 20.0

    router = VenueRouter("WETH", pools=[p])
    quote = router.quote_sell("WETH", p.book_capacity("WETH"))
    print(f"  router filled {quote.amount_in_filled:,.0f} of "
          f"{p.book_capacity('WETH'):,.0f} requested, ran_dry={quote.ran_dry}")
    assert quote.ran_dry, "the router ignored the reserve cap"
    assert quote.amount_out <= p.tvl_token1 * 1.01


def test_no_reserve_data_means_no_cap():
    """Absence of the constraint must not be treated as the constraint being
    satisfied -- but it also must not silently zero the pool out."""
    p = Pool("WETH", "USDC", sqrt_p_from_price(MID), book(MID, 4_000_000))
    assert p.reserve_limited_capacity("WETH") == float("inf")
    assert p.sell_capacity("WETH") == p.book_capacity("WETH")
    print("  with no reported reserves the book stands unclamped")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
    print(f"\nAll {len(tests)} checks passed.")
