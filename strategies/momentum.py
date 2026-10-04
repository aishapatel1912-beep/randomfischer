"""Inventory-aware Gabagool-style paired accumulation for Polymarket crypto UP/DOWN markets.

Core behavior:
  * First position: only take a genuinely cheap leg.
  * Once inventory exists: the underweight leg has absolute priority.
  * A normal hedge must preserve the normal pair-cost ceiling.
  * A stressed hedge may use a slightly looser rescue ceiling so a cheap first
    leg cannot strand the worker indefinitely with naked inventory.
  * Endgame inventory is much tighter than normal inventory.
  * The execution layer can unwind the overweight leg when buying the hedge is
    no longer economically sensible.

This is a public-mechanics approximation of a Gabagool-style method; it is not
private Gabagool22 source code.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

from strategies.base import GabagoolDecision
from strategies.momentum_execution import execute_momentum_decision
from utils.momentum_risk import is_market_locked

if TYPE_CHECKING:
    from bot import MarketWorker


_EPS = 1e-9


class GabagoolStrategy:
    @staticmethod
    def _leg(worker: "MarketWorker", side: str) -> Tuple[float, float]:
        inv = worker.inventory
        return float(inv.shares(side)), float(inv.avg_cost(side))

    @staticmethod
    def _select_first_side(
        yes_ask: float,
        no_ask: float,
        threshold: float,
    ) -> Optional[Tuple[str, float]]:
        candidates = []
        if 0 < yes_ask <= threshold + _EPS:
            candidates.append(("YES", yes_ask))
        if 0 < no_ask <= threshold + _EPS:
            candidates.append(("NO", no_ask))
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])

    @staticmethod
    def _round_up_tick(price: float) -> float:
        return math.ceil((price - _EPS) * 100.0) / 100.0

    async def evaluate(self, worker: "MarketWorker") -> Optional[GabagoolDecision]:
        from bot import OrderState, is_locked_price

        if worker.order_state == OrderState.PENDING or worker._order_lock.locked():
            return None
        if not worker.active_market or not worker.can_enter():
            return None

        cfg = worker.worker_config
        seconds_left = worker.market_seconds_left()
        # The minimum-time gate applies to opening fresh risk, not to a hedge
        # that reduces an already-existing imbalance. This distinction is
        # essential in the final minute.
        pre_yes_shares = float(worker.inventory.shares("YES"))
        pre_no_shares = float(worker.inventory.shares("NO"))
        pre_imbalance = abs(pre_yes_shares - pre_no_shares)
        if seconds_left < cfg.gabagool_min_time_to_resolution and pre_imbalance <= _EPS:
            return None

        yes_ask = worker.best_ask("YES")
        no_ask = worker.best_ask("NO")
        if is_market_locked(yes_ask, no_ask, is_locked=is_locked_price):
            worker.log_locked_market_skip(yes_ask, no_ask)
            return None
        if yes_ask <= 0 or no_ask <= 0:
            return None

        yes_shares, yes_avg = self._leg(worker, "YES")
        no_shares, no_avg = self._leg(worker, "NO")
        imbalance = yes_shares - no_shares
        abs_imbalance = abs(imbalance)
        endgame = seconds_left <= cfg.gabagool_endgame_seconds

        # Endgame rule: once the market is close to resolution, never open a
        # fresh first leg and never intentionally increase an existing skew.
        if endgame and abs_imbalance <= _EPS:
            return None

        if yes_shares <= _EPS and no_shares <= _EPS:
            picked = self._select_first_side(
                yes_ask, no_ask, cfg.gabagool_initial_entry_threshold,
            )
            if picked is None:
                return None
            side, ask = picked
            first_leg = True
            opposite_shares = 0.0
            opposite_avg = 0.0
            target_price = cfg.gabagool_initial_entry_threshold
            reason = "first_cheap_leg"
        else:
            first_leg = False

            # Inventory is the primary constraint. If we are even slightly
            # outside the soft band, ONLY the underweight side is eligible.
            if imbalance > _EPS:
                side = "NO"
                reason = "inventory_hedge_no"
            elif imbalance < -_EPS:
                side = "YES"
                reason = "inventory_hedge_yes"
            else:
                side = "YES" if yes_ask <= no_ask else "NO"
                reason = "balanced_cheap_leg"

            ask = yes_ask if side == "YES" else no_ask
            opposite = "NO" if side == "YES" else "YES"
            opposite_shares, opposite_avg = self._leg(worker, opposite)

            # Normal hedging is strict. Once the worker is outside the soft
            # inventory band, use the rescue ceiling instead of simply giving
            # up and leaving the original leg naked forever.
            if opposite_shares > _EPS and opposite_avg > 0:
                pair_ceiling = (
                    cfg.gabagool_hedge_max_pair_cost
                    if abs_imbalance > cfg.gabagool_inventory_soft_limit + _EPS
                    else cfg.gabagool_max_pair_cost
                )
                target_price = pair_ceiling - opposite_avg
            else:
                target_price = cfg.gabagool_initial_entry_threshold

        if target_price <= 0 or ask <= 0:
            return None

        size = worker.entry_order_size([side])
        if size is None or size <= 0:
            return None

        current_shares = yes_shares if side == "YES" else no_shares
        other_shares = no_shares if side == "YES" else yes_shares
        projected_unpaired = abs((current_shares + size) - other_shares)

        unpaired_limit = cfg.gabagool_endgame_max_unpaired if endgame else cfg.gabagool_max_unpaired_shares
        if projected_unpaired > unpaired_limit + _EPS:
            return None

        executable_price = self._round_up_tick(ask + cfg.gabagool_price_buffer)
        executable_price = min(executable_price, target_price)
        executable_price = round(executable_price, 2)
        if executable_price + _EPS < ask:
            return None

        current_avg = yes_avg if side == "YES" else no_avg
        projected_avg = (
            (current_shares * current_avg) + (size * executable_price)
        ) / max(current_shares + size, _EPS)

        projected_pair_cost = 0.0
        if opposite_shares > _EPS and opposite_avg > 0:
            projected_pair_cost = projected_avg + opposite_avg
            pair_ceiling = (
                cfg.gabagool_hedge_max_pair_cost
                if abs_imbalance > cfg.gabagool_inventory_soft_limit + _EPS
                else cfg.gabagool_max_pair_cost
            )
            if projected_pair_cost > pair_ceiling + _EPS:
                return None

        return GabagoolDecision(
            side=side,
            price=executable_price,
            size=size,
            trigger_price=ask,
            reason=reason,
            projected_pair_cost=round(projected_pair_cost, 6),
            first_leg=first_leg,
        )

    async def execute(self, worker: "MarketWorker", decision: GabagoolDecision) -> None:
        await execute_momentum_decision(worker, decision)


MomentumStrategy = GabagoolStrategy
