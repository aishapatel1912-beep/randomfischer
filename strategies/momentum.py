"""Inventory-aware Gabagool-style paired accumulation for Polymarket crypto UP/DOWN markets.

Core behavior:
  * First position: only take a high-confidence leg at or above the configured 90c entry floor.
  * Once the first leg fills: only buy the opposite hedge, at the maximum price
    that keeps the completed YES+NO pair below the configured pair ceiling.
  * A hedge may rest below the current ask; the bot does not chase a falling knife.
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
        max_entry: float,
    ) -> Optional[Tuple[str, float]]:
        """Choose the stronger side, not the cheaper side.

        The old 49c rule entered whichever token was cheap. That is exactly
        the falling-knife failure mode this strategy is meant to avoid. The
        new rule enters only when a side is already priced at/above the high-
        confidence floor (90c by default).
        """
        candidates = []
        if threshold <= yes_ask <= max_entry + _EPS:
            candidates.append(("YES", yes_ask))
        if threshold <= no_ask <= max_entry + _EPS:
            candidates.append(("NO", no_ask))
        if not candidates:
            return None
        # Prefer the stronger/highest-probability side.
        return max(candidates, key=lambda x: x[1])

    @staticmethod
    def _round_up_tick(price: float) -> float:
        return math.ceil((price - _EPS) * 100.0) / 100.0

    @staticmethod
    def _hedge_ceiling(worker: "MarketWorker", opposite_avg: float, stressed: bool) -> float:
        """Maximum executable price for the underweight hedge.

        Inventory reduction is not enough by itself: the hedge must also be
        economically acceptable.  The configured pair ceiling is capped by
        an explicit maximum loss per share on the completed pair.
        """
        cfg = worker.worker_config
        rescue_ceiling = cfg.gabagool_hedge_max_pair_cost if stressed else cfg.gabagool_max_pair_cost
        loss_ceiling = 1.0 - cfg.gabagool_max_hedge_loss_per_share
        pair_ceiling = min(rescue_ceiling, loss_ceiling)
        return max(0.0, pair_ceiling - max(0.0, opposite_avg))

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

        # Once a matched pair has already locked in the configured gross
        # margin, STOP starting another accumulation cycle in this market.
        # If inventory is imbalanced, the hedge branch below is still allowed
        # because it reduces risk rather than adding a new cycle.
        if abs_imbalance <= _EPS and worker.gabagool_profit_secured():
            return None

        # Endgame rule: once the market is close to resolution, never open a
        # fresh first leg and never intentionally increase an existing skew.
        if endgame and abs_imbalance <= _EPS:
            return None

        if yes_shares <= _EPS and no_shares <= _EPS:
            # FIRST LEG: buy only a high-confidence side (90c by default).
            # We do not buy the cheap side anymore.
            # Leave at least one cent of room for the opposite token. With a
            # 98c pair ceiling, the first leg can therefore be no higher than
            # 97c; at 90c it needs an 8c-or-better hedge to lock the 2c edge.
            max_first_entry = min(0.99, cfg.gabagool_max_pair_cost - 0.01)
            picked = self._select_first_side(
                yes_ask, no_ask, cfg.gabagool_initial_entry_threshold,
                max_first_entry,
            )
            if picked is None:
                return None
            side, ask = picked
            if cfg.gabagool_falling_knife_enabled and worker.is_falling_knife(side, ask):
                print(
                    f"🛑 [GABAGOOL 90C BLOCK] {worker.asset_type.upper()} "
                    f"{worker.window_slug} | {side}={round(ask*100)}c "
                    f"is falling; waiting for stabilization"
                )
                return None
            first_leg = True
            opposite_shares = 0.0
            opposite_avg = 0.0
            target_price = 0.99
            reason = "first_high_confidence_leg"
        else:
            first_leg = False

            # Once a first leg exists, ONLY the opposite side may be bought.
            # Never add to the already-overweight leg.
            if imbalance > _EPS:
                side = "NO"
                reason = "hedge_yes_with_no"
            elif imbalance < -_EPS:
                side = "YES"
                reason = "hedge_no_with_yes"
            else:
                # A balanced pair is already complete. Do not start another
                # cycle unless the previous pair has been fully secured.
                return None

            ask = yes_ask if side == "YES" else no_ask
            opposite = "NO" if side == "YES" else "YES"
            opposite_shares, opposite_avg = self._leg(worker, opposite)
            if opposite_shares <= _EPS or opposite_avg <= 0:
                return None

            # The hedge is NOT chased. Calculate the maximum price we can pay
            # for the opposite side and still lock the configured gross edge.
            stressed = abs_imbalance > cfg.gabagool_inventory_soft_limit + _EPS
            target_price = self._hedge_ceiling(worker, opposite_avg, stressed)
            if target_price < 0.01 - _EPS:
                return None

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

        if first_leg:
            # Cross the current ask for the high-confidence first leg.
            executable_price = self._round_up_tick(ask + cfg.gabagool_price_buffer)
            executable_price = min(executable_price, target_price)
            executable_price = round(executable_price, 2)
            if executable_price + _EPS < ask:
                return None
        else:
            # Hedge leg: place a resting GTC at the safe price. If the ask is
            # already there, it can fill immediately; if not, we wait for the
            # opposite side to become cheap enough. This is the key anti-chase
            # protection against the falling-knife problem.
            executable_price = round(target_price, 2)
            if executable_price < 0.01:
                return None

        current_avg = yes_avg if side == "YES" else no_avg
        projected_avg = (
            (current_shares * current_avg) + (size * executable_price)
        ) / max(current_shares + size, _EPS)

        projected_pair_cost = 0.0
        if opposite_shares > _EPS and opposite_avg > 0:
            projected_pair_cost = projected_avg + opposite_avg
            stressed = abs_imbalance > cfg.gabagool_inventory_soft_limit + _EPS
            pair_ceiling = min(
                cfg.gabagool_hedge_max_pair_cost if stressed else cfg.gabagool_max_pair_cost,
                1.0 - cfg.gabagool_max_hedge_loss_per_share,
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
