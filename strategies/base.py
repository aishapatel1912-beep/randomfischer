"""Strategy contracts for Gabagool-style paired UP/DOWN trading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from bot import MarketWorker


@dataclass(frozen=True)
class GabagoolDecision:
    """One executable leg of a delta-neutral YES/NO accumulation cycle."""

    side: str
    price: float
    size: float
    trigger_price: float
    reason: str
    projected_pair_cost: float
    first_leg: bool


class GabagoolStrategyProtocol(Protocol):
    async def evaluate(self, worker: "MarketWorker") -> Optional[GabagoolDecision]:
        ...

    async def execute(self, worker: "MarketWorker", decision: GabagoolDecision) -> None:
        ...


# Compatibility aliases.  The bot imports these names today; keeping them
# avoids forcing unrelated code to change at the same time as the strategy.
MomentumDecision = GabagoolDecision
MomentumStrategyProtocol = GabagoolStrategyProtocol
