"""
Temporary vs permanent price impact.

Every cascade in this project so far treats a liquidator's price impact as
PERMANENT: collateral is dumped, the price falls, and it stays fallen for the
rest of the cascade. That is the most aggressive assumption available, and it
is wrong in a specific, quantifiable way.

Market impact decomposes into two parts. The permanent component reflects
information -- a forced seller reveals something, and the market reprices.
The temporary component is the cost of demanding liquidity faster than it is
supplied, and it decays: arbitrageurs buy the dislocation against other
venues, market makers re-quote, and passive liquidity refills the book. On
Ethereum that happens within a block or two, which is fast relative to a
cascade that plays out over many blocks.

So the question is not "does the price recover" but "how much of it recovers
before the next wave of liquidations is triggered?" That is one parameter:

    kappa = the fraction of the cascade-induced dislocation that reverts
            between rounds

    kappa = 0   every basis point of impact is permanent -- the assumption
                every earlier version of this model made
    kappa = 1   impact is entirely temporary; the market is back at its
                post-shock fundamental before the next round, so there is no
                reflexive feedback at all and the cascade cannot amplify

Reality sits between, and the honest output is a RANGE over kappa rather than
a point estimate. This is the Almgren-Chriss temporary/permanent split in the
form a cascade model can use, and it is the same quantity equity-execution
desks calibrate as impact decay.

WHAT IT REVERTS TOWARD
----------------------
The post-shock fundamental, not the pre-shock price. An exogenous -20% shock
is the market genuinely repricing; that does not bounce back. What bounces
back is the EXTRA dislocation the forced selling caused on top of it. Mixing
those two up would quietly undo the shock itself and make every cascade look
harmless.

A NOTE ON WHAT IS BEING MODELLED
--------------------------------
The pools here carry a static tick book: a sale moves the price but does not
consume the segments, so depth at any given price is always the same. The
model therefore already assumes liquidity refills at each price level; kappa
governs only whether the PRICE comes back. That is a simplification -- real
books thin out exactly when they are hit hardest -- and it means these
figures understate stress at high kappa. Stated here rather than buried.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ImpactDecay:
    """How much cascade-induced price dislocation reverts between rounds."""

    kappa: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.kappa <= 1.0:
            raise ValueError(f"kappa must be in [0, 1], got {self.kappa}")

    @property
    def enabled(self) -> bool:
        return self.kappa > 0.0

    def revert(self, current_price: float, reference_price: float) -> float:
        """Price after one round of reversion toward `reference_price`."""
        deviation = current_price - reference_price
        return reference_price + deviation * (1.0 - self.kappa)

    def describe(self) -> str:
        if self.kappa == 0:
            return "impact fully permanent (kappa=0)"
        if self.kappa == 1:
            return "impact fully temporary (kappa=1) -- no reflexive feedback"
        return f"{self.kappa:.0%} of dislocation reverts per round (kappa={self.kappa})"


# Plausible bracket for a sweep. Ethereum blocks are ~12s and arbitrage
# against deeper venues is fast, so on a per-round basis a large kappa is not
# exotic -- but it is an assumption, which is the entire reason to sweep it.
DEFAULT_KAPPA_GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
