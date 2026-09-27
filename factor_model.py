"""
Correlated shocks, and the two depegs that actually happened.

`systematic_scoring.py` ranks collateral assets by shocking each one in
isolation, holding every other price fixed. For a book whose collateral is
mostly ETH and ETH derivatives, that is not a conservative simplification --
it is close to meaningless. wstETH, weETH, cbETH and rETH are all claims on
staked ether. They do not move independently of ETH; they move almost exactly
WITH it. A "wstETH-only" shock is not a scenario any market can produce.

So shocks here are applied to a FACTOR, and each asset moves by its loading
on that factor:

    return_i = beta_i * factor_return + idiosyncratic_i

For an ETH-beta book this collapses most of the collateral onto one number,
which is the correct structure: diversification across four flavours of
staked ether is not diversification.

THE TWO SCENARIOS THAT MATTER
-----------------------------
Both are idiosyncratic moves layered on top of the factor, and both have
happened:

  LST depeg -- a liquid staking token trades below the ether it represents.
      stETH traded around 0.94 ETH in June 2022 as Celsius and 3AC unwound.
      The collateral does not just fall with ETH; it falls RELATIVE to ETH,
      and every position collateralised in it takes both hits at once.

  Stablecoin depeg -- the DEBT side breaks. USDC traded near $0.88 in March
      2023 after Silicon Valley Bank failed. Every model in this project
      until now priced stablecoin debt at exactly $1.00, which assumes away
      the single most disruptive thing that has happened to this market.
      A depeg makes debt CHEAPER to repay, so health factors improve and
      liquidations become less likely -- the opposite of the naive intuition,
      and a reason to model it rather than assert it.

BETAS
-----
The defaults below are structural, not estimated: an LST's beta to ETH is
approximately 1 by construction, because it is a claim on ETH. `estimate_betas`
regresses real returns when price history is available, and that is what
should be used; the defaults exist so a scenario still runs without it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Beta to the ETH factor. LSTs are ~1.0 by construction (each is a claim on
# staked ether); BTC's ~0.8 is a rough long-run figure and the one number
# here most worth replacing with a regression. Stablecoins are ~0 -- they do
# not track ETH, they break for their own reasons, which is what the depeg
# scenarios are for.
DEFAULT_ETH_BETAS = {
    # Ether and every claim on it. Beta 1 is structural, not estimated: these
    # tokens ARE ether, plus a staking yield and a redemption discount.
    "WETH": 1.00, "ETH": 1.00,
    "wstETH": 1.00, "stETH": 1.00, "cbETH": 1.00, "rETH": 1.00,
    "weETH": 1.00, "osETH": 1.00, "rsETH": 1.00, "ezETH": 1.00, "ETHx": 1.00,

    # Bitcoin and its wrappers. ~0.8 is a long-run rule of thumb and the
    # number here most deserving of a regression.
    "WBTC": 0.80, "cbBTC": 0.80, "tBTC": 0.80, "LBTC": 0.80, "eBTC": 0.80,
    "FBTC": 0.80, "BTCb": 0.80, "BTC.b": 0.80,

    # DeFi governance tokens: higher beta than ETH, since they are levered
    # bets on the same activity. 1.3 is a stated assumption, not a fit --
    # replace it with estimate_betas() the moment price history is available.
    "AAVE": 1.30, "CRV": 1.30, "UNI": 1.30, "MKR": 1.30, "SNX": 1.30,
    "LDO": 1.30, "BAL": 1.30, "ENS": 1.30, "1INCH": 1.30, "FXS": 1.30,
    "LINK": 1.20,

    # Dollar stablecoins: ~0 beta to ETH by design. They do not track the
    # market, they break for their own reasons -- which is what the depeg
    # scenarios exist to model.
    "USDC": 0.0, "USDT": 0.0, "DAI": 0.0, "GHO": 0.0, "LUSD": 0.0,
    "sUSD": 0.0, "FRAX": 0.0, "PYUSD": 0.0, "USDS": 0.0, "USDe": 0.0,
    "sUSDe": 0.0, "eUSDe": 0.0, "syrupUSDT": 0.0, "sDAI": 0.0,

    # Not dollar-pegged and not ETH-correlated: a euro stablecoin and
    # tokenised gold. Zero beta to the ETH factor is right; both still carry
    # their own FX or commodity risk, which this model does not represent.
    "EURC": 0.0, "XAUt": 0.0,
}

LST_SYMBOLS = {"wstETH", "stETH", "cbETH", "rETH", "weETH", "osETH", "rsETH",
               "ezETH", "ETHx"}
STABLE_SYMBOLS = {"USDC", "USDT", "DAI", "GHO", "LUSD", "sUSD", "FRAX",
                  "PYUSD", "USDS", "USDe", "sUSDe", "eUSDe", "sDAI"}


@dataclass
class FactorShock:
    """A market-wide move plus named idiosyncratic overrides."""
    name: str
    factor_return: float                                  # e.g. -0.30
    idiosyncratic: Dict[str, float] = field(default_factory=dict)
    betas: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ETH_BETAS))
    description: str = ""

    def asset_returns(self, assets) -> Dict[str, float]:
        """Return for each asset: beta * factor + its own idiosyncratic move."""
        out = {}
        for asset in assets:
            beta = self.betas.get(asset, 0.0)
            out[asset] = beta * self.factor_return + self.idiosyncratic.get(asset, 0.0)
        return out

    def shocked_prices(self, prices: Dict[str, float]) -> Dict[str, float]:
        returns = self.asset_returns(prices.keys())
        return {a: p * (1 + returns.get(a, 0.0)) for a, p in prices.items()}


def estimate_betas(returns_by_asset: Dict[str, List[float]],
                   factor_asset: str = "WETH") -> Dict[str, float]:
    """OLS beta of each asset's returns on the factor's, no intercept.

    Assets with too little overlapping history keep their structural default
    rather than being assigned a beta fitted to a handful of points.
    """
    factor = returns_by_asset.get(factor_asset)
    if not factor:
        raise ValueError(f"no returns for the factor asset {factor_asset!r}")

    betas = dict(DEFAULT_ETH_BETAS)
    for asset, series in returns_by_asset.items():
        n = min(len(series), len(factor))
        if n < 20:
            continue
        x, y = factor[-n:], series[-n:]
        denominator = sum(v * v for v in x)
        if denominator <= 0:
            continue
        betas[asset] = sum(a * b for a, b in zip(x, y)) / denominator
    return betas


# ---------------------------------------------------------------------------
# Named scenarios
# ---------------------------------------------------------------------------

def market_crash(factor_return: float = -0.30) -> FactorShock:
    return FactorShock(
        name=f"broad crash ({factor_return:+.0%} ETH factor)",
        factor_return=factor_return,
        description="Everything falls together by its beta. The baseline "
                    "against which the depeg scenarios should be read.")


def lst_depeg(factor_return: float = -0.15, depeg: float = -0.08) -> FactorShock:
    return FactorShock(
        name=f"LST depeg ({depeg:+.0%} vs ETH, on a {factor_return:+.0%} market)",
        factor_return=factor_return,
        idiosyncratic={s: depeg for s in LST_SYMBOLS},
        description="Staked-ETH tokens trade below the ether they represent, "
                    "as stETH did around 0.94 in June 2022. Positions "
                    "collateralised in LSTs take the market move and the "
                    "discount at once.")


def stablecoin_depeg(symbol: str = "USDC", depeg: float = -0.12,
                     factor_return: float = 0.0) -> FactorShock:
    return FactorShock(
        name=f"{symbol} depeg to ${1 + depeg:.2f}",
        factor_return=factor_return,
        idiosyncratic={symbol: depeg},
        description=f"{symbol} breaks its peg, as it did to ~$0.88 in March "
                    f"2023. Where {symbol} is DEBT this makes the debt cheaper "
                    f"to repay and health factors improve; where it is "
                    f"COLLATERAL the opposite. Which effect dominates depends "
                    f"entirely on the book, which is why it is worth running.")


def stable_depeg_in_a_crash(symbol: str = "USDC", depeg: float = -0.12,
                            factor_return: float = -0.25) -> FactorShock:
    return FactorShock(
        name=f"{symbol} depeg during a {factor_return:+.0%} crash",
        factor_return=factor_return,
        idiosyncratic={symbol: depeg},
        description="The combination, which is how it would actually happen: "
                    "a stablecoin breaks while collateral is already falling.")


def default_scenarios() -> List[FactorShock]:
    return [
        market_crash(-0.15),
        market_crash(-0.30),
        market_crash(-0.45),
        lst_depeg(),
        stablecoin_depeg(),
        stable_depeg_in_a_crash(),
    ]


def compare_isolated_vs_factor(asset: str, prices: Dict[str, float],
                               shock: float = -0.20,
                               betas: Optional[Dict[str, float]] = None):
    """Show what shocking one asset alone misses.

    Returns (isolated_prices, factor_prices) for the same nominal move, so a
    caller can run both through the cascade and compare. For an LST the two
    are wildly different: in isolation nothing else moves, whereas any factor
    move large enough to take an LST down that far takes ETH, every other LST
    and most of BTC with it.
    """
    betas = betas or DEFAULT_ETH_BETAS
    isolated = dict(prices)
    if asset in isolated:
        isolated[asset] = isolated[asset] * (1 + shock)

    beta = betas.get(asset, 1.0) or 1.0
    equivalent_factor = shock / beta
    factor = FactorShock(name=f"factor move implying {asset} {shock:+.0%}",
                         factor_return=equivalent_factor, betas=betas)
    return isolated, factor.shocked_prices(prices)
