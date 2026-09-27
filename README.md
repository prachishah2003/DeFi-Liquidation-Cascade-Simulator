# DeFi Liquidation Cascade Simulator

A live, data-driven model of how crypto lending liquidations feed on
themselves. Most liquidation-risk tools ask *"if price drops by X%, who
gets liquidated?"* treating the price drop as something that just
happens. This project models the part everyone else skips: liquidated
collateral gets **sold**, that sale moves the real market price, and that
extra price move can trigger the **next** wave of liquidations. It's a
feedback loop, not a single event — and this simulates it end-to-end using
real, live data from Aave, Compound, and Uniswap, rather than assumed or
historical-only numbers.

The data pipeline is checked against a real historical crash (Jan 31,
2026): it independently reconstructs Aave's liquidation volume for that
day and lands close to Aave's own published figure. That is a check on the
data plumbing. Whether the *cascade model* predicts the right magnitude is
a separate question, tested separately, and answered more cautiously --
see **Key results** below.

## What's in here

| Piece | What it does |
|---|---|
| Core engine | Endogenous cascade: liquidations sell into real Uniswap v3 liquidity, price impact feeds back into who's next |
| Live data pipeline | Pulls real open positions from Aave v3 + Compound v3, and real pool liquidity from Uniswap v3, via The Graph |
| Multi-collateral model | Properly blends health factors across every asset an account holds, instead of collapsing to one dominant asset |
| Cross-protocol | Combines Aave + Compound positions into one shared cascade through the same pool |
| Historical backtest | Validates the model against the real Jan 31, 2026 crash |
| Shock sweep | Tests the cascade across a range of shock sizes, charts the result |
| Oracle lag | Models Chainlink-style delayed price updates instead of assuming instant price knowledge |
| Systemic asset scoring | Ranks which collateral assets are most dangerous if shocked |
| VaR / Expected Shortfall | Formal statistical risk figures, computed through the actual cascade mechanism |
| **Liquidator economics** | **A liquidation happens only if the bonus covers slippage + gas + flash-loan fee. Positions nobody will touch stay open, and their shortfall becomes protocol bad debt** |
| **Multi-venue routing** | **Liquidations are split across every Uniswap fee tier plus off-chain depth, at the common price arbitrage enforces — not dumped into one pool** |
| **Factor shocks & depegs** | **Correlated moves via ETH betas, plus LST and stablecoin depeg scenarios — instead of shocking correlated assets one at a time** |
| **Two-hop execution** | **Staking tokens sell through LST/WETH and then WETH/USDC, sharing one WETH book — so an LST unwind moves the ether price** |
| Dashboard | Live interactive Streamlit app tying the above together |

## Setup

**1. Clone and create a virtual environment**
```bash
git clone <this-repo-url>
cd defi-liquidization
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get a free Graph API key**

This project reads live blockchain data via [The Graph](https://thegraph.com):
1. Go to **thegraph.com/studio** and sign in (a browser wallet like MetaMask works as the login, no funds needed, it's just used as an identity)
2. Create an API key (free tier: ~100k queries/month)
3. **Important:** open the key's **Security** settings and either select **"All subgraphs"**, or explicitly authorize these three IDs (queries fail with an auth error otherwise, even with a valid key):
   - Uniswap v3: `5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV`
   - Aave v3: `JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk`
   - Compound v3: `AwoxEZbiWLvv6e3QdvdMZw4WDURdGbvPfHmZRc8Dpfz9`
4. Export it in your shell:
```bash
export GRAPH_API_KEY=your_key_here
```
A freshly created key can take a few minutes to become active -- if you get `"API key not found"` immediately after creating one, wait a couple of minutes and retry.

## Running it

```bash
# Core cascade demo, synthetic data (no API key needed) -- start here
python3 "Demo cascade.py"

# The same thing, on real live data
python3 live_demo_cascade.py

# Proper multi-collateral accounts instead of the single-asset simplification
python3 live_demo_multi.py

# Aave + Compound combined, sharing one pool -- the real contagion story
python3 live_data_cross_protocol.py

# WETH + WBTC pools both live -- cross-asset price-impact contagion
python3 live_demo_multi_pools.py

# Validate against the real Jan 31, 2026 crash
python3 historical_backtest.py

# Cascade severity across a range of shock sizes, saves shock_sweep.png
python3 shock_sweep.py

# Instant-price vs. lagged-oracle cascade, side by side
python3 live_oracle_lag_compare.py

# Which collateral asset is most systemically dangerous?
python3 live_systematic_scoring.py

# Formal VaR / Expected Shortfall risk figures
python3 var_analysis.py

# Bad debt and liquidation volume across shock sizes
python3 bad_debt_sweep.py

# The largest shock the book absorbs, and what execution depth buys
python3 critical_shock.py

# Correlated crashes, LST depegs and stablecoin depegs on the real book
python3 scenario_runner.py

# Measure the impact-decay coefficient instead of assuming it
python3 estimate_kappa.py

# Check every live query still matches the current subgraph schema
python3 verify_schema.py

# Live interactive dashboard (opens in your browser)
streamlit run dashboard.py
```

Every live script prints its own diagnostics as it runs (how many accounts
were fetched, how many got filtered and why, etc.) — if something looks
off, that output is usually enough to tell you where.

## Key results

Every figure below was produced by a script in this repository, replayed
from the committed snapshot in `snapshots/`, and can be reproduced offline
with no API key — the command that generates each one is named beside it.
Two claims are deliberately kept apart, because conflating them is the
easiest way to overstate what this project has shown.

| result | produced by |
|---|---|
| Largest shock the book absorbs, by execution depth and κ | `critical_shock.py` |
| R₀ cross-check and its agreement with the cascade scan | `critical_shock.py` |
| LST and stablecoin depeg damage | `scenario_runner.py` |
| Measured impact decay κ | `estimate_kappa.py` |
| Bad debt and liquidation volume vs shock size | `bad_debt_sweep.py` |
| Cascade amplification vs shock size | `shock_sweep.py` |
| Every live query still matches the schema | `verify_schema.py` |

**What the data pipeline demonstrates (high confidence).** On the real Jan
31, 2026 crash, the live-data pipeline independently reconstructs Aave's
liquidation volume for that day and lands within the same ballpark as
Aave's own published ~$140M. This confirms the subgraph queries, the
decimal handling and the pagination are right. It says nothing about the
cascade model, because no model output is involved in producing it.

**What the cascade model demonstrates (lower confidence, stated as such).**
`historical_backtest.py` now compares the model's predicted WETH collateral
liquidated against the real WETH collateral liquidated that day, scaled by
the fraction of Aave's borrows the position sample actually captured. A
close ratio is encouraging, not proof: the model runs on *today's*
positions with a shock of the same magnitude, not on a block-pinned replay
of that day's book, and one event is an anecdote. An error distribution
across several historical crashes is the figure worth quoting, and it
isn't built yet.

**Cross-protocol risk is understated by single-protocol models.** Combining
Aave and Compound — both ultimately selling into the same market — produces
materially larger cascades than modeling either alone, because both
protocols' liquidations compete for the same on-chain liquidity.

**An LST depeg is far more dangerous than a much larger broad crash.** Run
against the live multi-asset book (90.7% of Aave borrows, κ=0 as measured,
60 gwei, and $100M/1% of off-chain depth — see the note under the table), a
liquid-staking-token discount of 8% on top of a −15% market puts *the same
number of accounts underwater* as that −15% market alone — and produces
**5.6× the bad debt**:

| scenario | accounts underwater | liquidated | bad debt |
|---|---|---|---|
| broad crash, −15% | 286 | 125 | $12.92M |
| broad crash, −30% | 352 | 170 | $13.81M |
| broad crash, −45% | 532 | 313 | $30.40M |
| **LST depeg (−8% vs ETH on a −15% market)** | **285** | **118** | **$72.39M** |
| USDC depeg to $0.88 | 277 | 119 | $12.08M |
| USDC depeg during a −25% crash | 319 | 146 | $12.46M |

**Every row assumes $100M of off-chain depth per 1% of price move.** That is
the most consequential assumption in this model and it is an order-of-magnitude
judgement, not a measurement: on these same books the shock that breaks the
system sits at −25.8% with on-chain execution only and at −49.8% with this much
off-chain depth. `scenario_runner.py` prints the figure with the table, and
`critical_shock.py` sweeps it.

**The mechanism, traced rather than assumed.** The three accounts that
dominate the depeg row hold LST collateral against **WETH debt** — leveraged
staking carry trades. They are delta-neutral to ether: their health factor is
*identical* (0.8657) at a −30% shock and at a −45% one, because the market
move cancels out of the collateral-to-debt ratio. A broad crash, however deep,
barely touches them. A depeg moves the one thing they are exposed to, and
their collateral stops covering their debt.

So the fragility does not sit where a risk dashboard would look for it. It is
not a function of how far ether falls; it is a property of a position
structure that a fall in ether leaves alone.

Counting every underwater account by how much of it a liquidator will take:

| | liquidated in full | only a partial slice pays | nothing pays, at any size |
|---|---|---|---|
| broad crash, −15% | 118 | 20 | 148 |
| LST depeg on a −15% market | 75 | 62 | 148 |

Look at the last column: **identical**. The depeg does not make more positions
untouchable. What it does is push them out of the first column and into the
second — three times as many accounts can now only be *partly* cleared, so
debt is left outstanding on positions a liquidator has already been paid to
work on. That is **position-level insolvency, not illiquidity**, and it is a
distinction worth keeping because the two have different remedies.

This is a sharper reading than the one it replaces. Before partial
liquidation the same effect showed up as accounts becoming wholly
unliquidatable, which was an artifact of the all-or-nothing rule rather than
a fact about the book.

Read the absolute counts in that breakdown with care. They depend heavily on
how much depth the reconstructed LST books hold, and those books are thin —
$13.0M of usable weETH depth, $5.4M wstETH, $0.3M cbETH, $0.2M rETH — and
each staking token sells into a single on-chain pool with no off-chain depth
behind it, while WETH gets three fee tiers plus $100M/1%. The *direction* is
the finding; the exact counts move with book coverage.

Account counts are therefore a misleading risk metric. Counting who is
underwater misses the question that matters, which is who can actually be
wound down.

**A stablecoin depeg on the debt side is a stabiliser, not an amplifier.**
USDC at $0.88 produces *less* bad debt than the baseline ($12.08M vs $12.92M),
and layered onto a −25% crash it partly offsets it. Debt denominated in a
broken dollar is cheaper to repay, so health factors improve. Every model in
this project priced stablecoin debt at exactly $1.00 until now, which
assumed away the single most disruptive thing that has happened to this
market — and assumed it in the wrong direction.

**Two independent methods put the threshold in the same place.** The
−25.8% figure above comes from brute force: run the full cascade at many
shock sizes and find where damage appears. `branching.py` finds the same
boundary from the opposite direction, as a branching ratio — R₀, the volume
of liquidation each wave triggers divided by the volume it sold. R₀ > 1
means each wave is larger than the last. **R₀ first exceeds 1 at −26%,
against −25.8% from the cascade scan — agreement to 0.2 percentage points**,
from a local generation ratio and a full simulation respectively. That is
evidence the threshold is a property of the book rather than an artifact of
either method.

That agreement was 1.2 points until the two methods were made to use the same
liquidation rule. Once the cascade started taking partial slices,
`liquidatable_volume` was still discarding any position whose *full*
repayment was refused — so R₀ read low for exactly the positions partial
liquidation rescues, and the cross-check was validating a different model
from the one it was checking.

**But R₀ measured at the start is not an all-clear.** At −26% the first
generation reads 0.141 — comfortably subcritical — while the worst point
along the path reaches **5.368**. Nothing about the market changed; the
cascade simply walked down into a denser part of the position ladder, where
each dollar of selling flips more accounts than it did higher up. So the intuitive use of R₀ — compute
it each block, watch it approach 1 — is weaker than it sounds: initial R₀
systematically understates danger, and `max_path_r0` is the figure that
matches reality. A low reading is a cheap screen, not a verdict.

**R₀ only means something where its denominator does.** Generation zero is
whatever is liquidatable at the shocked price, and just past the first
liquidation that is a handful of accounts. At −20% it is $1.2M against a
$1,229M book, and it triggers $5.7M: R₀ = 4.623. The arithmetic is right and
the reading is worthless — 0.1% of the book flipping 0.5% of it says nothing
about systemic amplification. Deep shocks produce small denominators too, for
the opposite reason: past a point the price move makes most liquidations
unprofitable, so little is liquidatable at all. `critical_shock.py` therefore
gates the verdict on generation zero clearing the same 1%-of-book threshold it
uses for "broken", and labels the rest **too small to rate** rather than
"contained" — because at depth those are the opposite claim. Ungated, that
single −20% row made R₀ appear to contradict the cascade scan by 6 points.

**Where you sell matters more than almost anything else.** Every earlier
version of this model sold every liquidated position into a single Uniswap
0.3% pool. Real liquidators route through an aggregator that splits across
every fee tier, and for WETH the largest venues are not on-chain at all.
Correcting this changes the headline number by more than an order of
magnitude: on a synthetic book, a −25% shock produces **−1.02pp of cascade
amplification through one pool versus −0.04pp routed realistically**, roughly
a 25× overstatement, and ignoring off-chain depth alone overstates the price
move about 8×. The split is not assumed — it is solved for, by finding the
single common price arbitrage forces every venue to, which is also the split
that minimises total slippage. Off-chain depth is a linear order book
parameterised by the notional that moves price 1%; it is the model's most
consequential assumption and `bad_debt_sweep.py` varies it explicitly.

**How large a crash the book absorbs, and what that buys.** The useful
figure for a risk team is not "how bad is a 40% crash" but the largest shock
the system takes before liquidations stop working. On a real Aave snapshot
(37% of protocol borrows, $1.23bn of WETH-collateralised debt, 60 gwei),
`critical_shock.py` scans for the smallest shock that leaves bad debt above
1% of the sampled book:

| execution depth assumed | survives a shock of |
|---|---|
| on-chain only (all three fee tiers) | **−25.8%** |
| + $25M/1% off-chain | −46.2% |
| + $50M/1% off-chain | −48.7% |
| + $100M/1% off-chain | −49.8% |
| + $300M/1% off-chain | −50.6% |

Execution depth is worth roughly **24.8 percentage points of shock tolerance**
across that range, with sharply diminishing returns past ~$50M/1%. That is a
more decision-useful way to price liquidity than quoting slippage in basis
points.

**The cascade is bistable, but only when execution is constrained.** With
on-chain venues alone the transition is a genuine phase change rather than a
curve: a −25.8% shock produces a −30.4% realised move and essentially no bad
debt, while a fraction of a point deeper produces at least −77.7% and
**$486M** — a fifth of a percentage
point of extra shock changing the outcome by three orders of magnitude. The
unravelling itself takes 13 rounds, with liquidations continuing throughout
(12, 4, 2, 8, 5, 6, 10, 6, 15, 9, 6, 3, 4) while the accounts nobody will
touch pile up from 4 to 213. Read that −77% as a floor rather than a figure:
the cascade drives the price to $606.80 and the fetched ±15,000-tick window
bottoms out at $606.73, so the model has no book below it and the true move
is further down — which makes the $486M a lower bound. Add
realistic off-chain depth and that discontinuity disappears: damage then
accumulates smoothly and the book simply tolerates much more. So reflexive
collapse is not an inherent property of the lending market; it is what
happens to a lending market whose liquidations have nowhere to go.

**A non-monotonicity that was also an artifact.** An earlier version of this
section pointed at `bad_debt_sweep.png` and argued that the
single-0.3%-pool series being flat near zero at −26% and −30% while sitting
at ~$505M at −18%, −22%, −35% and −40% was a real effect: the liquidator
economics gate, with deeper shocks deterring liquidators and so doing less
damage. The mechanism was traced from round-one liquidation counts and it
held up at the time.

It does not survive partial liquidation. With the all-or-nothing rule gone,
the same series ignites at −15% and stays at ~$506M for every deeper shock —
monotone, no dip. What looked like liquidators walking away from a deeper
crash was liquidators walking away from a *whole position* they would happily
have taken a slice of.

That is the third finding in this project to turn out to be a property of the
liquidation rule rather than of the book, after the κ substitution claim and
the bad-debt peak at −30%. The pattern is worth naming: **non-monotonicity in
this model has never once been real.** Every time it has appeared it has been
a discontinuity in the machinery — a boundary in the swap engine, a ratchet
in the truncation, an all-or-nothing threshold in the liquidator. It remains
a reason to go looking rather than a result to write down.

What the panel still shows, and this part is robust: where you sell dominates
everything. The realistic series stays at zero across every shock swept while
the single-pool series is catastrophic from −15% on. Note also the `[tick data
exhausted]` marks from −22% down — those points sit at the bottom of the
fetched window, so the $506M is a floor rather than a measurement.

**…and what happens if price impact is not permanent?** Letting a fraction κ
of each round's dislocation revert before the next wave — arbitrage, market
makers, passive liquidity refilling — moves the thresholds, but far less than
this project used to claim:

| κ (reverts per round) | on-chain only | + $50M/1% off-chain |
|---|---|---|
| 0.00 (fully permanent) | −25.8% | −48.7% |
| 0.25 | −26.8% | −48.7% |
| 0.50 | −27.8% | −48.7% |
| 0.75 | −34.0% | −48.7% |
| 0.90 | −38.9% | −48.7% |
| 1.00 (fully temporary) | −41.2% | −48.7% |

**Two claims previously made here are withdrawn.** The first was that a
quarter of the dislocation reverting would erase most of the fragility,
worth 19 points of shock tolerance. It is worth **one** point. Across the
entire κ range reversion buys 15.4 points, and κ=0.25 buys 1.0 of them — 6%,
not most. The second was that depth and reversion are *substitutes*: even
**full** reversion (−41.2%) does not reach what $50M/1% of execution depth
buys (−48.7%). They are not interchangeable.

Both were artifacts of all-or-nothing liquidation. Under that rule a high κ
let the book recover to the point where full close-factor repayments stopped
clearing their costs, and the cascade simply halted — which looked like a
rescue. With partial liquidation there is almost always *some* slice that
still pays, so the selling never stops; it grinds. Measured at a −30% shock:
κ=0 ends in 3 rounds having liquidated 32 accounts, κ=0.5 runs 28 rounds and
liquidates 122.

**Reversion buys time, not an end to the selling.** That is a less
comfortable conclusion than the one it replaces, and it is the second time in
this project that a headline finding turned out to be a property of the
liquidation rule rather than of the book.

Which row is the real one is therefore an empirical question, not a
modelling choice — so `estimate_kappa.py` measures it.

**Measured κ at cascade speed is ~0.00.** Every large trade is a natural
experiment: it pushes price, and price either stays pushed or comes back.
Across 53 large trades (≥$80k) on the 0.05% pool over 24 hours:

| horizon | ≈ elapsed | κ (median) | signal/noise | relevant? |
|---|---|---|---|---|
| 1 swap | 14s | **0.00** | 1754 | **cascade speed** |
| 3 swaps | 43s | 0.04 | 45 | too slow |
| 5 swaps | 72s | −0.01 | 13 | too slow |
| 10 swaps | 2.4m | 0.35 | 4.6 | too slow |
| 25 swaps | 6.0m | 0.43 | 2.6 | too slow |
| 50 swaps | 11.9m | 0.90 | 1.3 | too slow |

Impact *does* revert — about 90% of it — but over roughly ten minutes. A
cascade propagates in a block or two, 12 to 36 seconds, and over that window
nothing measurably comes back. **So κ=0 — which looked like this model's
most aggressive assumption — is approximately correct for the timescale that
matters, and the κ=0 row of the table above is the relevant one rather than
a worst case.** Quoting the ten-minute figure would understate risk by
roughly 20 percentage points of shock tolerance.

Two guards on that claim. Each measurement carries a signal-to-noise figure
— the trade's own dislocation against how far price drifts over the same
window anyway — because reversion measured over a window where the market
wanders further than the trade moved it is not a measurement. And ordinary
large trades are not forced liquidations: liquidations are publicly
predictable and arrive together, so informed counterparties can withdraw
exactly when they are needed. Real crisis κ is likely *lower* still, making
this an optimistic bound. The spread across individual events is wide (IQR
at 14s: −0.19 to 0.92), it is one pool over one calm day, and it should be
re-run across a volatile window before anyone leans on it.

**Gas is close to irrelevant at realistic position sizes.** Sweeping gas from
10 to 300 gwei moves peak bad debt by under 1% once venues are modelled
properly. The positions that matter are worth millions; a few thousand
dollars of gas does not decide whether they are liquidated. Gas only gates
dust — which the earlier synthetic runs, built on far smaller positions,
made look far more important than it is.

**Diversification dampens single-asset amplification.** Modeling true
multi-collateral accounts (vs. the single-dominant-asset approximation)
shows more total liquidations but *less* concentrated selling pressure on
any one asset.

**On the "cascade ceiling".** An earlier version of this README claimed that
pushing the shock far enough hits a real liquidity boundary in the pool.
That claim could not be supported at the time: the tick fetch was capped at
1,000 ticks with no pagination, so a book that merely ran out of *fetched
data* was indistinguishable from a pool that ran out of *liquidity*. The
engine now reports those two cases separately (`stop_reason` on every swap:
`filled` / `data_exhausted` / `liquidity_exhausted`), the tick fetch
paginates, and `shock_sweep.png` marks any point that ran off the end of
the data with an X so it cannot be read as a result. Re-run the sweep with
a wide `tick_window` before making any claim about a ceiling.

## Corrections log

Findings from a review of this code, and what was done about each. Kept in
the README rather than buried in commit messages, because several of them
invalidated numbers this README previously quoted.

| Finding | Status |
|---|---|
| The swap walk skipped the **active tick range** whenever the price sat inside it rather than on a boundary — i.e. after every exogenous shock, since shocks move the price mid-range. The walk then integrated the *next* range's liquidity across the skipped span, overstating price impact by ~12x on a concentrated book. | Fixed. Regression-tested against an independent reference implementation in `test_uniswap_v3_math.py`. **Every amplification figure produced before this fix was too large — re-run the sweep.** |
| "Positions liquidated" summed liquidation *events*, counting one account once per round it was hit. On a six-position test book this reported 10 for 6 accounts. Positions also never terminated: a flat 50% close factor makes debt decay geometrically without ever reaching zero. | Fixed. `summarize()` reports accounts, events and dollars separately; the close factor now follows Aave's real rule (100% below HF 0.95), so positions actually close. |
| Missing prices were silently defaulted — collateral to `0.0`, debt to `1.0`. Both defaults bias the same way, toward liquidation: unpriced collateral made healthy accounts look insolvent, unpriced debt booked non-stablecoin borrowings at a dollar. | Fixed. Raises `MissingPriceError`; `drop_unpriced()` filters affected positions and reports how many were dropped. |
| Position sampling took 500 accounts ordered by `openPositionCount` descending — the accounts with the *most positions*, which tracks sophistication rather than size or risk — with no pagination and no statement of what fraction of the protocol it represented. | Fixed. Risk-weighted per-market top-borrower sampling by default, cursor pagination, and a `CoverageReport` printed with every fetch stating what share of protocol borrows the sample captured. |
| Tick fetching was capped at `first: 1000` with no pagination, so a truncated book was indistinguishable from an exhausted pool. | Fixed. Cursor-paginated ticks; `stop_reason` distinguishes `data_exhausted` from `liquidity_exhausted`; the sweep chart marks untrustworthy points. |
| Historical liquidation volume was paginated with `skip`, which the Graph gateway caps at 5,000 records — silently under-counting a busy day, then comparing the short total against Aave's full-day figure. | Fixed. Cursor pagination on `id`, no page cap. |
| The README's headline validation claim described a data-pipeline check as though it validated the model. | Fixed above, and the backtest now makes the model-vs-reality comparison explicitly. |
| **Aave reports collateral in aTokens, and they were treated as separate assets.** `aEthWETH` *is* WETH — the subgraph prices them identically — but nothing normalised the symbol. Single-asset code filtering `collateral_asset == "WETH"` silently dropped every aToken position (most of why the WETH sample was 295 accounts at 37% coverage rather than the full book), and the factor model assigned `aEthwstETH` a beta of zero, treating staked ether as uncorrelated with ether. | Fixed. `normalize_asset_symbol()` maps chain-prefixed aTokens to their underlying, the count is reported on every fetch, and a schema check fails if any survive. |
| Accounts whose collateral had already been seized down to dust, but whose debt record lingered, were counted as live positions at health factor zero — 941 of them, manufacturing phantom insolvency and a meaningless "already underwater" headline. | Fixed. Positions below $100 of collateral are dropped as closed, and the count is reported. |
| **Every pool was charged a 0.30% fee.** Uniswap runs a separate pool per fee tier charging 5bp, 30bp or 100bp, but the swap engine defaulted to 30bp for all of them — so the 0.05% tier, the deepest WETH/USDC venue and the one taking most routed flow, was charged six times its real fee. | Fixed. `Pool.fee_bps` is read from the subgraph's `feeTier` and threaded through every quote, sale and routing calculation. |
| **Routing equalised post-trade price, which is wrong once fees differ.** A pool charging 100bp needs more input to reach any given price than one charging 5bp, because the fee comes out of the input rather than moving the price — so equalising price sent *more* flow to the dearer venue. Only visible once pools carried their real fees. | Fixed. Routing now equalises fee-adjusted marginal proceeds, which is what maximises the seller's output. On two identical-depth pools at 5bp and 100bp, the cheap one now takes 2,902 of a 4,000 WETH order instead of 1,098. |
| **The single-asset collapse kept an account's entire debt against only its largest collateral asset.** A diversified account holding $10.7M WETH + $12.8M wstETH + $7.7M WBTC against $20M USDC has a true health factor of 1.25; collapsed this way it recorded as 0.51 — falsely insolvent, carrying $7.2M of bad debt that does not exist. On live Aave data this produced $420M of phantom bad debt from a 4% shock and made every sweep result non-monotonic. | Fixed. `collapse="rescale"` (default) preserves the account's true blended health factor and therefore its real solvency condition; `"strict"` keeps only genuinely single-asset accounts; `"legacy"` reproduces the bug on purpose, and a test asserts that it still does. |
| **Price impact was discontinuous at exactly `sell_capacity`.** The reconstructed weETH/WETH book holds two genuinely deep ranges (L≈4,790,207) then ~85 ranges holding 0.03–0.66, spanning 24% of price. Crossing an empty range costs no input, so selling 99.9% of capacity moved the ratio −0.19% while selling 100% of it — the last 0.04 of 4,337 weETH — moved it **−24.41%**, crossing 84 further ticks. Cascade flow saturates at capacity, so whether a scenario landed on that boundary or a few tokens short decided whether every weETH-collateralised account was marked down a quarter. It was also marking a 401,299-weETH position to the last atom of a 4,337-weETH book. | Fixed. `usable_segments` cuts the walk at the end of real book, which also makes impact monotone in trade size. **This is what made bad debt non-monotonic in shock size — $373.28M at −30% against $32.34M at −45%. Both figures were artifacts and neither should be cited.** |
| **The first fix for that ratcheted.** Measuring "real depth" relative to what lay ahead of the *current* price meant that once a sale consumed the real ranges, the dead tail became the book and the next sale promoted the next sliver of it. Three successive full-capacity sales gave −0.19%, −5.72%, −24.41% — and the third moved the price 19 points while filling **zero** tokens. | Fixed. `_liquidity_floor` derives the cut from the fetched book rather than the live price, so consuming the book cannot redefine what counts as book. Both bugs are pinned by `test_book_truncation.py`, which fails on the pre-fix behaviour. |
| **The scenario table did not disclose its execution assumption.** `fetch_venue_router()` defaults to $100M/1% of off-chain depth and `scenario_runner.py` took that default silently, while disclosing coverage, κ and betas. On the same Uniswap books that one assumption moves the breaking shock from −26% to −49%. | Fixed. The footer now states the figure and both thresholds, so the table reads as conditional on it. |
| **R₀'s verdict had no materiality floor.** R₀ is generation one over generation zero, and just past the first liquidation generation zero is a handful of accounts. At −20% it was $1.2M against a $1,229M book triggering $5.7M: R₀ = 4.623, arithmetically correct and economically empty. That one row set the first supercritical shock to −20% and produced an apparent 6-point disagreement with the brute-force scan. | Fixed. The verdict is gated on generation zero clearing the same 1%-of-book threshold already used for "broken"; small-denominator rows are labelled *too small to rate*. Gated, the two methods agree to about a tenth of a point. |
| **Pool reserves never depleted.** `Pool.sell` moved the price and nothing else, so `reserve_limited_capacity` was re-derived from the pool's ORIGINAL balance on every call. The live wstETH leg returned a capacity of 1,591 wstETH, then 1,592, then 1,594, for as many rounds as the cascade cared to run — each sale paying out WETH the pool had already spent. | Fixed. A swap now settles both balances. Note what was and was not wrong: Uniswap's liquidity L genuinely does not change as price moves through a range, so leaving the tick book alone was correct; treating the *payout token* as inexhaustible was not. Pinned by `test_reserve_depletion.py`. |
| **Liquidation was all-or-nothing.** A liquidation whose full close-factor repayment did not clear its costs was refused outright, leaving the entire position open. Slippage grows faster than linearly in size, so a smaller slice of the same position frequently does pay — refusing the whole thing overstates bad debt, and overstates it worst on exactly the thin books this model is about. | Fixed. The liquidator takes the largest viable slice — largest rather than most profitable, because whatever margin the first one leaves, the next bids for. Between 20 and 62 accounts per scenario are now partly cleared instead of abandoned. Pinned by `test_partial_liquidation.py`. |
| `repaid_usd` was accumulated BEFORE the profitability check, so positions the liquidator refused still counted toward the round's repaid total. | Fixed while restructuring the above. |
| **The liquidation bonus was a flat 5% everywhere.** Aave charges 5% on blue-chip collateral and 7.5–10% on thinner assets — backwards from what the model assumed about where liquidation is attractive. | Fixed. Read per market by `fetch_market_risk_params`, deliberately as a SEPARATE query: snapshots are keyed on query text, so adding a field to the positions query would have invalidated every recorded snapshot and made the offline replay workflow unusable. Old snapshots miss it and fall back to 5% with a printed warning. Also corrected a misstatement — Aave v3's close factor is protocol-level, so there is no per-market close factor to read. Pinned by `test_market_risk_params.py`. |
| **R₀ and the cascade used different liquidation rules.** After partial liquidation landed, `liquidatable_volume` still discarded any position whose *full* repayment was refused, while the cascade went on to clear a profitable slice of it. R₀ read low for exactly the positions partial liquidation rescues, so the cross-check was validating a different model from the one it checked. | Fixed. Both use the same rule; agreement with the brute-force scan improved from 1.2 percentage points to 0.2. |
| **The round cap truncated cascades silently.** Working positions down in slices takes many more rounds than closing them whole, and `max_rounds=25` began to bind — a −30% shock at κ=0.5 needs 28. The effect was small ($31.44M against $31.49M converged) and completely invisible. | Fixed. The cap is 100, and any run that still reaches it prints a warning saying the result is truncated rather than converged. |
| **An interpretation was hardcoded into the reporting.** `critical_shock.py` printed "that number is almost entirely an artifact of assuming permanent impact" unconditionally — a conclusion baked into a print statement, which stayed on screen after it stopped being true. | Fixed. It now computes the κ gradient and describes what it actually finds. |
| **VaR and ES assumed normally distributed returns**, which understates precisely the tail this project is about, and `var_analysis.py` ran its cascade through a single 0.3% pool with no liquidator economics — the free-liquidation model the rest of the project had abandoned. | Fixed. Three estimators run side by side (normal, historical, Cornish-Fisher) with skew and excess kurtosis reported, the cascade runs on the Cornish-Fisher figures through the full router with economics, and the historical row warns when its tail rests on too few observations to be a quantile. Pinned by `test_var_methods.py`. |

**How that one stayed hidden, and what it says about the rest.** Two
properties of the old model concealed it. A cascade that liquidates
everything for free cannot distinguish a fake liquidation from a real one,
so a mispriced position looked exactly like a healthy one being wound down.
And with no bad-debt accounting, an account recorded as insolvent had
nowhere to show up. It surfaced within one run of adding liquidator
profitability — not because that feature looks for data errors, but because
refusing a liquidation forces the model to state what the position was
actually worth. Features that make a model *disagree with itself* are worth
more than features that add scope.

It also got worse before it got better: risk-weighted sampling deliberately
selects the largest accounts, which are the most diversified, which are
exactly the ones the collapse mangled. Better sampling made a latent bug
bite harder. Worth remembering before trusting any single improvement in
isolation.

**A retraction, kept deliberately.** An earlier version of this README
reported a "liquidation cliff" — liquidation volume peaking around −22% and
falling thereafter, so that a bigger crash liquidated less. That came from a
synthetic book, and it **does not reproduce on real data**: with live venues
liquidation volume rises monotonically to the deepest shock swept. The
mechanism behind it is real (unprofitable liquidations do get skipped, and
`bad_debt_sweep.py` still reports them), but the headline shape was an
artifact of a book far thinner than the real one. Corrected rather than
quietly dropped, because the difference between a result and an artifact of
your own test fixture is the whole game.

**A second retraction, same shape.** An earlier version of this README
reported bad debt *peaking* at a −30% shock and falling at −45%, and treated
that non-monotonicity as a finding about where the book is most fragile. It
was the capacity cliff in the corrections log above. Bad debt is monotone in
shock size on this book — $12.92M, $13.81M, $30.40M across −15%, −30% and
−45% — and so are the underwater count, the liquidated count and the
cascade's own contribution to the price move. The diagnosis took four
attempts, three of which were wrong: a phantom-bad-debt hypothesis, a
bad-debt-exclusion hypothesis, and a first fix that replaced the cliff with a
ratchet. Each was killed by measuring the thing it predicted rather than by
argument, which is the only reason to trust the fourth.

Closed since that list was written: multi-venue routing, two-hop LST
execution, correlated factor shocks with LST and stablecoin depegs, and a
measured rather than assumed κ.

Still open, in rough priority order: widening the tick window until no
headline result rests on the edge of fetched data, since the deepest cascade
currently bottoms out against it; a liquidity-consuming book, so depth
thins where it has already been hit; partial liquidation, since all-or-
nothing per round overstates bad debt at thin depth; per-market close
factors and liquidation bonuses instead of Aave-wide defaults; a block-level
clock so oracle latency is quantified in time rather than abstract rounds;
and a true block-pinned historical replay across several crashes.

## Known limitations (stated deliberately, not hidden)

- **Price impact now covers the staking tokens too, but that path is newly built.** `routing.TwoHopVenue` sells an LST through its LST/WETH pool and then sells the resulting WETH through the shared WETH venues, so staked-ether liquidations push the ether price — a contagion channel the model previously could not represent at all. Pools are discovered by token symbol and ranked by TVL rather than hardcoded. Every other collateral asset is still priced at fetch time and sells at its mark with no slippage, which understates its contribution; `scenario_runner.py` now prints exactly which assets those are on every run.
- **κ is measured, but from one calm day on one pool.** `estimate_kappa.py` puts it near zero at cascade speed, which is what the model now uses. That rests on 53 events over 24 hours in a 2.7% price range, with a wide spread across events, and ordinary large trades stand in for forced liquidations. Re-running it across a genuinely volatile window is the obvious next step, and the figure should be treated as provisional until then.
- **Reconstructed LST books are thin, and may be truncated rather than merely illiquid.** Usable depth as fetched: WETH $125.9M, weETH $13.0M, wstETH $5.4M, cbETH $0.3M, rETH $0.2M. The wstETH book still implies it could pay out 464× the WETH the pool actually holds, so its slippage is a lower bound and its capacity should not be quoted — `live_data.py` prints that warning on every run. Any result that depends on repricing a staking token is dominated by this, and it is the single most valuable thing to fix next.
- **Tick-window edge effects bound the deepest results.** Where a cascade drives the price to the edge of the fetched window, the model reports a floor rather than an outcome — see the −77% figure above, which sits $0.07 from the bottom of the book. Widen `tick_window` before quoting any such number.
- **The off-chain depth figure is a judgement, not a measurement, and it is worth 23 percentage points of critical shock.** Read every scenario number as conditional on it.
- **The tick book is static in L, which is right; its reserves no longer are.** Uniswap's liquidity L does not change as price moves through a range, so the tick book is correct to stay put. What a swap does change is the pool's holdings, and those now deplete. What remains unmodelled is LPs withdrawing under stress, which real books do exactly when hit hardest.
- **Off-chain depth is a single linear parameter.** `LinearDepthVenue` collapses every CEX, market maker and unmodelled on-chain venue into one number: the notional that moves price 1%. Real books are neither linear nor uniformly deep, and depth evaporates precisely during the crashes this model is about. Treat it as a dial to sweep, not a measurement.
- **Liquidator cost parameters are assumptions, not measurements.** Gas units (450k), gas price, the 9bp flash-loan fee and the $25 minimum-profit floor are all order-of-magnitude defaults. `bad_debt_sweep.py` sweeps gas across three regimes precisely because the cliff location depends on them; treat the *shape* of the result as the finding and the exact cliff percentage as indicative.
- **Partial liquidation assumes competition, not a monopolist.** The model takes the largest slice that clears its costs, reasoning that any margin the first liquidator leaves, the next bids for. A single liquidator facing no competition would take the profit-maximising slice instead, which is smaller — leaving more debt outstanding and more bad debt.
- **MakerDAO is not included.** It uses a structurally different vault/CDP model rather than the pooled-market pattern Aave and Compound share, so it doesn't fit the current data-fetching code as-is.
- Single-asset mode restricts debt to stablecoins (USDC/USDT/DAI) for a clean 1:1 pricing assumption.
- The historical backtest compares real historical price/liquidation data against the model run on **today's** live positions (not a true block-pinned historical replay) — a documented approximation, not a limitation hidden from the reader.
- **VaR still assumes tomorrow is drawn from the same distribution as the last 90 days.** Normality is gone — three estimators run side by side and the fat-tailed ones are used — but volatility clusters, so the day after a crash is not drawn from the calm-period distribution. None of the three models that. A GARCH-style conditional volatility model is the honest next step.
- **The liquidation bonus is read per market; the close factor cannot be.** Aave v3's close factor is protocol-level — 50%, rising to 100% below the health-factor threshold — so there is no per-market value to fetch. An earlier version of this list claimed both were per-market defaults; only the bonus was.
- **The LST venues have no off-chain depth and the WETH venue does.** WETH sells into three fee tiers *plus* $100M/1% of assumed off-chain depth; each staking token sells into a single on-chain pool and nothing else. Now that reserves deplete, the wstETH pool's WETH side drains to zero in round one and every LST venue reports dry. Real wstETH also trades on Curve, Balancer and centralised venues, so the model overstates how fast LST collateral becomes unsellable. Giving the LST legs their own sweepable depth parameter is the clearest next fix.

## Project structure

```
cascade_sim.py                  Core single-asset cascade engine + health factor
liquidator.py                    Liquidator profitability + bad-debt accounting
routing.py                       Multi-venue execution (fee tiers + off-chain depth)
branching.py                     R0 branching ratio and cascade-path tracing
market_impact.py                 Temporary vs permanent impact (kappa decay)
factor_model.py                  ETH-factor betas, LST and stablecoin depeg scenarios
snapshot.py                      Record/replay cache for reproducible runs
uniswap_v3_math.py               Uniswap v3 swap math (price impact)
chart_style.py                   Shared, CVD-validated chart tokens
multi_asset.py                   Multi-collateral/multi-debt cascade engine
live_data.py                     Live data fetchers (Aave, Compound, Uniswap via The Graph)
oracle_lag.py                    Oracle-staleness cascade variant
systematic_scoring.py            Systemic asset ranking logic
var_analysis.py                  VaR / Expected Shortfall analysis

Demo cascade.py                  Synthetic-data demo (no API key needed)
live_demo_cascade.py             Live single-asset cascade
live_demo_multi.py               Live multi-collateral cascade
live_demo_multi_pools.py         Live cascade with WETH + WBTC pools
live_data_cross_protocol.py      Live Aave + Compound combined cascade
live_oracle_lag_compare.py       Live instant-oracle vs. lagged-oracle comparison
live_systematic_scoring.py       Live systemic asset ranking
historical_backtest.py           Backtest against the real Jan 31, 2026 crash
shock_sweep.py                   Cascade severity across a range of shock sizes
bad_debt_sweep.py                Bad debt and liquidation volume vs. shock size
critical_shock.py                Largest shock the book absorbs, by execution depth
scenario_runner.py               Factor and depeg scenarios on the multi-asset book
estimate_kappa.py                Measures impact decay from real swap data
verify_schema.py                 Checks every live query against the current schema
dashboard.py                     Interactive live Streamlit dashboard

test_liquidator.py               Liquidator economics + bad-debt tests
test_routing.py                  Multi-venue routing and split-optimality tests
test_market_impact.py            Impact-decay arithmetic and cascade behaviour
test_branching.py                R0 and its agreement with full cascades
test_factor_model.py             Correlated shocks and depeg arithmetic
test_snapshot.py                 Record/replay round-trip tests
test_uniswap_v3_math.py          Swap-engine tests, incl. the active-tick regression
test_sampling.py                 Pagination / sampling / coverage tests (no network)
test_live_data.py                Tests for the live-data parsing/math logic
test_multi_asset.py              Tests for the multi-collateral cascade logic
conftest.py                      Lets pytest import the suites from the repo root
.github/workflows/tests.yml      CI: every suite on Python 3.9, 3.11 and 3.12
requirements.txt
LICENSE
```

## License

Add a `LICENSE` file at the project root if you want one, the `LICENSE`
file inside `.venv/` belongs to a bundled dependency, not this project.

---

*Built as a master's project. Not aimed at publication, but validated
against real market data throughout rather than left as a theoretical
exercise.*
