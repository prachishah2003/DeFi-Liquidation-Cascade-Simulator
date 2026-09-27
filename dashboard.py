"""
Live DeFi liquidation cascade dashboard.

Run with:  streamlit run dashboard.py

Requires GRAPH_API_KEY set (same as every other script in this project).
Ties together live position/pool data, the endogenous cascade engine, the
multi-asset extension, a VaR/Expected Shortfall risk summary, and (in
multi-asset mode) the systemic asset ranking -- behind a single what-if
shock slider.

Deliberately NOT included here (kept as standalone scripts instead):
oracle-lag comparison (its value is in the round-by-round narrative, which
reads better as a log than a widget) and the WBTC cross-pool contagion demo
(a small, context-dependent number better suited to a report paragraph).
See live_oracle_lag_compare.py and live_demo_multi_pools.py for those.
"""

import os
import copy

import streamlit as st
import pandas as pd

from live_data import fetch_aave_positions, fetch_aave_positions_multi, \
    fetch_uniswap_pool, WETH_USDC_POOL
from cascade_sim import run_cascade, health_factor, summarize
from multi_asset import run_cascade_multi, multi_health_factor
from var_analysis import fetch_daily_returns, compute_var_es
from systematic_scoring import score_systemic_assets

st.set_page_config(page_title="DeFi Liquidation Cascade Monitor", layout="wide")


# ---------------------------------------------------------------------------
# Cached data fetch -- avoids re-hitting the subgraph on every slider tick.
# Cascade runs themselves are NOT cached (they're fast, local, and depend
# on the shock slider) -- only the live fetch is.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Fetching live pool liquidity...")
def load_pool():
    return fetch_uniswap_pool(WETH_USDC_POOL, tick_window=15000)


@st.cache_data(ttl=300, show_spinner="Fetching live Aave positions...")
def load_positions_single():
    all_positions = fetch_aave_positions(first=500, verbose=False)
    return [p for p in all_positions if p.collateral_asset == "WETH"]


@st.cache_data(ttl=300, show_spinner="Fetching live Aave positions (multi-asset)...")
def load_positions_multi():
    return fetch_aave_positions_multi(first=500, verbose=False)


@st.cache_data(ttl=3600, show_spinner="Fetching 90 days of price history for volatility...")
def load_daily_returns():
    return fetch_daily_returns(days=90)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Scenario controls")

if not os.environ.get("GRAPH_API_KEY"):
    st.error("GRAPH_API_KEY is not set. Export it in the terminal you launched "
             "`streamlit run` from, then restart the app.")
    st.stop()

model_choice = st.sidebar.radio(
    "Position model",
    ["Single-asset (WETH-only, most validated)", "Multi-collateral (proper blended HF)"],
    help="Single-asset collapses each account to its dominant collateral/debt "
         "asset. Multi-collateral models every asset an account holds, but "
         "only WETH's price impact is simulated (see project notes)."
)
use_multi = model_choice.startswith("Multi")

shock_pct = st.sidebar.slider(
    "WETH shock size (%)", min_value=-40, max_value=0, value=-12, step=1,
    help="Trustworthy up to about -35% for this pool -- beyond that, real "
         "on-chain liquidity for this pool runs out (see project notes). "
         "Also used for the systemic asset ranking below."
)
if shock_pct <= -36:
    st.sidebar.warning("Shocks beyond ~-35% may hit a real liquidity boundary "
                        "in this pool -- results past that point are unreliable, "
                        "not just extreme.")

run_button = st.sidebar.button("Run cascade", type="primary")

if st.sidebar.button("Refresh live data (clears cache)"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("DeFi Liquidation Cascade Monitor")
st.caption("Live Aave v3 positions + Uniswap v3 WETH/USDC liquidity. "
           "Endogenous cascade: liquidations sell into real AMM depth, "
           "which moves price, which triggers more liquidations.")

pool = load_pool()
current_price = pool.collateral_price("WETH")

if use_multi:
    positions, static_prices = load_positions_multi()
    prices_now = dict(static_prices)
    prices_now["WETH"] = current_price
    hf_fn = multi_health_factor
else:
    positions = load_positions_single()
    prices_now = {"WETH": current_price}
    hf_fn = health_factor

col1, col2, col3 = st.columns(3)
col1.metric("WETH price (live)", f"${current_price:,.2f}")
col2.metric("Positions loaded", f"{len(positions)}")

hfs = [(p, hf_fn(p, prices_now)) for p in positions]
underwater_now = [p for p, hf in hfs if hf < 1.0]
col3.metric("Already underwater", f"{len(underwater_now)}",
            delta=None if not underwater_now else "at current prices",
            delta_color="inverse")

st.subheader("Positions closest to liquidation")
worst = sorted(hfs, key=lambda x: x[1])[:15]
if use_multi:
    rows = [{
        "Account": p.position_id,
        "Health Factor": round(hf, 3),
        "Collateral": "+".join(p.collateral.keys()),
        "Debt": "+".join(p.debt.keys()),
        "Status": "UNDERWATER" if hf < 1.0 else "",
    } for p, hf in worst]
else:
    rows = [{
        "Account": p.position_id,
        "Health Factor": round(hf, 3),
        "Collateral": p.collateral_asset,
        "Debt": p.debt_asset,
        "Status": "UNDERWATER" if hf < 1.0 else "",
    } for p, hf in worst]
df = pd.DataFrame(rows)
st.dataframe(
    df.style.apply(
        lambda row: ["background-color: #ffe0e0" if row["Status"] == "UNDERWATER" else ""] * len(row),
        axis=1
    ),
    width="stretch", hide_index=True,
)

st.divider()


# ---------------------------------------------------------------------------
# Shared scenario runner -- used by both the manual "Run cascade" button
# and the automatic VaR/ES section below, so there's one code path for
# "what does a shock of size X do."
# ---------------------------------------------------------------------------

def run_shock_scenario(pct: float):
    sim_pool = copy.deepcopy(pool)
    sim_positions = copy.deepcopy(positions)

    if use_multi:
        logs = run_cascade_multi(sim_positions, {"WETH": sim_pool}, static_prices,
                                  initial_shock={"WETH": pct / 100},
                                  max_rounds=20, verbose=False)
    else:
        logs = run_cascade(sim_positions, {"WETH": sim_pool},
                            initial_shock={"WETH": pct / 100},
                            max_rounds=20, verbose=False)

    final_price = sim_pool.collateral_price("WETH")
    realized_pct = (final_price / current_price - 1) * 100
    summary = summarize(logs)
    ran_dry = any(l.pool_ran_dry for l in logs)
    weth_sold = summary.collateral_sold.get("WETH", 0.0)
    dollar_liquidated = weth_sold * current_price

    return {
        "logs": logs, "final_price": final_price, "realized_pct": realized_pct,
        "accounts_liquidated": summary.accounts_liquidated,
        "liquidation_events": summary.liquidation_events,
        "debt_repaid_usd": summary.debt_repaid_usd,
        "ran_dry": ran_dry, "dollar_liquidated": dollar_liquidated,
    }


# ---------------------------------------------------------------------------
# Run the manual scenario (slider-driven)
# ---------------------------------------------------------------------------

if run_button:
    st.subheader(f"Cascade result: {shock_pct:+d}% WETH shock")
    result = run_shock_scenario(shock_pct)
    logs = result["logs"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shock applied", f"{shock_pct:+d}%")
    m2.metric("Realized move", f"{result['realized_pct']:+.2f}%",
              delta=f"{result['realized_pct'] - shock_pct:+.2f}pp amplification")
    m3.metric("Accounts liquidated", f"{result['accounts_liquidated']}",
              help=f"{result['liquidation_events']} liquidation events -- one "
                   f"account can be liquidated in several consecutive rounds, so "
                   f"events always run ahead of accounts.")
    m4.metric("Cascade rounds", f"{len(logs)}")

    summary_obj = summarize(logs)
    if summary_obj.pools_data_exhausted:
        st.error("The cascade walked off the end of the fetched tick data "
                 f"({sorted(set(summary_obj.pools_data_exhausted))}). This is a "
                 "DATA limit, not a market one -- the real pool may well have "
                 "more liquidity below this point. Widen `tick_window` in "
                 "`load_pool()` and re-run before reading anything into it.")
    elif result["ran_dry"]:
        st.warning("Pool liquidity was genuinely exhausted before the cascade "
                   "resolved -- the fetched book covers the full range and the "
                   "pool still had nothing left to absorb the selling. This "
                   "result is a lower bound on the move, and a real finding "
                   "about how concentrated this pool's liquidity is.")

    if logs:
        price_path = pd.DataFrame({
            "Round": [0] + [l.round_num for l in logs],
            "WETH price": [current_price * (1 + shock_pct / 100)]
                          + [l.prices_after["WETH"] for l in logs],
            "Liquidated this round": [0] + [len(l.liquidated) for l in logs],
        })
        c1, c2 = st.columns(2)
        with c1:
            st.caption("WETH price through the cascade")
            st.line_chart(price_path.set_index("Round")["WETH price"])
        with c2:
            st.caption("Positions liquidated per round")
            st.bar_chart(price_path.set_index("Round")["Liquidated this round"])
    else:
        st.info("No positions went underwater at this shock size.")

    st.divider()


# ---------------------------------------------------------------------------
# Risk summary: VaR / Expected Shortfall
# ---------------------------------------------------------------------------

st.subheader("Risk summary (Value-at-Risk / Expected Shortfall)")
st.caption("Real ~90-day historical volatility, normal-distribution VaR/ES "
           "shock sizes, run through the actual cascade engine for a dollar "
           "figure -- not just a raw statistical estimate.")

try:
    returns = load_daily_returns()
    var_shock, es_shock, sigma = compute_var_es(returns, confidence=0.95)
    var_result = run_shock_scenario(var_shock * 100)
    es_result = run_shock_scenario(es_shock * 100)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Daily volatility (90d)", f"{sigma:.2%}")
    v2.metric("95% VaR shock", f"{var_shock:+.2%}")
    v3.metric("VaR: $ liquidated", f"${var_result['dollar_liquidated']:,.0f}",
              help="Model's prediction if a 1-in-20-day-bad move happens today.")
    v4.metric("ES: $ liquidated", f"${es_result['dollar_liquidated']:,.0f}",
              help=f"Average of the worst 5% of days ({es_shock:+.2%} shock) -- "
                   f"the tail-risk figure, not just the threshold.")
    st.caption("Caveat: assumes normally-distributed returns, which understates "
               "real crypto fat tails -- treat this as a lower bound on true "
               "tail risk, not a conservative upper one.")
except Exception as e:
    st.warning(f"Couldn't compute VaR/ES this session: {e}")

st.divider()


# ---------------------------------------------------------------------------
# Systemic asset ranking (multi-asset mode only -- needs the full asset
# picture that single-asset mode collapses away)
# ---------------------------------------------------------------------------

if use_multi:
    st.subheader("Systemic asset ranking")
    st.caption(f"Each asset shocked {shock_pct:+d}% in isolation (same slider as "
               f"above) -- which assets flip the most accounts underwater, and "
               f"which carry the most debt at risk when they do. These can "
               f"disagree -- a narrowly-held asset can still be more dangerous "
               f"in dollar terms than a widely-held one.")

    results = score_systemic_assets(positions, prices_now, shock_pct=shock_pct / 100)
    if results:
        rdf = pd.DataFrame(results)
        rdf["flip_rate"] = (rdf["flip_rate"] * 100).round(1).astype(str) + "%"
        rdf["debt_value_at_risk"] = rdf["debt_value_at_risk"].map(lambda v: f"${v:,.0f}")
        rdf = rdf.rename(columns={
            "asset": "Asset", "accounts_holding": "Holders",
            "accounts_flipped": "Flipped", "flip_rate": "Flip rate",
            "debt_value_at_risk": "Debt at risk",
        })

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Sorted by accounts flipped (breadth)")
            st.dataframe(rdf.sort_values("Flipped", ascending=False).head(10),
                         width="stretch", hide_index=True)
        with c2:
            # sort by the raw numeric column before formatting was applied
            raw = pd.DataFrame(results).sort_values("debt_value_at_risk", ascending=False).head(10)
            raw["debt_value_at_risk"] = raw["debt_value_at_risk"].map(lambda v: f"${v:,.0f}")
            raw["flip_rate"] = (raw["flip_rate"] * 100).round(1).astype(str) + "%"
            raw = raw.rename(columns={
                "asset": "Asset", "accounts_holding": "Holders",
                "accounts_flipped": "Flipped", "flip_rate": "Flip rate",
                "debt_value_at_risk": "Debt at risk",
            })
            st.caption("Sorted by debt at risk (depth)")
            st.dataframe(raw, width="stretch", hide_index=True)
    else:
        st.info("No scoreable assets found in this batch.")

    st.divider()
else:
    st.caption("Switch to Multi-collateral mode in the sidebar to see the "
               "systemic asset ranking.")
    st.divider()


st.caption(
    "Scope notes: price-impact modeling covers WETH only (one live pool). "
    "Other collateral assets (multi-asset mode) are priced at fetch-time "
    "and held static through the cascade -- systemic ranking above measures "
    "direct price exposure, not second-order cross-asset propagation "
    "(see live_demo_multi_pools.py for that). Single-asset debt is "
    "restricted to stablecoins (USDC/USDT/DAI). See project writeup for "
    "full methodology and validated backtest results."
)