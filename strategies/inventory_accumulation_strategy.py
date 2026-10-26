"""Sequential 10–20¢ first leg, then opposite leg to 98¢ combined pair cost."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from strategies.inventory_accumulation_execution import fok_buy_with_retries
from utils.falling_knife_filter import FallingKnifeState, evaluate_falling_knife, record_buy_on_side
from utils.inventory_balance import BuyApproval
from utils.inventory_cost import InventorySnapshot, matched_combined_average
from utils.inventory_strategy_store import (
    AccumulationPhase,
    MarketCycleRecord,
    inventory_strategy_store,
)
from utils.profitability_gate import check_executable_depth
from utils.sequential_pair_strategy import (
    SequentialDecision,
    entry_cutoff_active,
    evaluate_opposite_with_entry_state,
    mark_side_filled,
    max_opposite_gross_price,
    pick_first_side,
    side_already_entered,
)

if TYPE_CHECKING:
    from bot import MarketWorker

_LOG = logging.getLogger("inventory_accumulation")


def _opposite(side: str) -> str:
    return "NO" if side.upper() == "YES" else "YES"


class InventoryAccumulationStrategy:
    def _key(self, worker: "MarketWorker") -> str:
        return worker.worker_config.key

    def _slippage_reserve(self, worker: "MarketWorker") -> float:
        cfg = worker.worker_config
        return max(cfg.max_slippage, cfg.spread_capture_slippage_buffer)

    def _knife_state(self, cycle: Optional[MarketCycleRecord]) -> FallingKnifeState:
        if cycle is None:
            return FallingKnifeState()
        return FallingKnifeState.from_metadata(cycle.metadata.get("falling_knife"))

    def _persist_knife_state(self, cycle: Optional[MarketCycleRecord], state: FallingKnifeState) -> None:
        if cycle is None:
            return
        cycle.metadata["falling_knife"] = state.metadata_blob()
        inventory_strategy_store.upsert(cycle)

    def _log_decision(
        self,
        worker: "MarketWorker",
        cycle: Optional[MarketCycleRecord],
        *,
        reason: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        cfg = worker.worker_config
        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        secs = worker.market_seconds_left()
        payload = {
            "event": "sequential_pair_decision",
            "market_id": worker.canonical_market_id() or "",
            "seconds_left": secs,
            "yes_executable": worker.best_ask("YES"),
            "no_executable": worker.best_ask("NO"),
            "phase": cycle.phase.value if cycle else "",
            "yes_shares": snap.yes_shares,
            "no_shares": snap.no_shares,
            "combined_avg": matched_combined_average(snap),
            "entry_cutoff": entry_cutoff_active(secs, cfg.no_buys_last_minutes),
            "dry_run": worker.is_dry_run(),
            "reason": reason,
            "ts": time.time(),
        }
        if extra:
            payload.update(extra)
        worker.dashboard["last_gate_reason"] = reason
        worker.dashboard["entry_cutoff_active"] = payload["entry_cutoff"]
        _LOG.info(json.dumps(payload))
        if reason.startswith("REJECT") or reason.startswith("SKIP") or reason.startswith("WAIT"):
            worker.dashboard["entry_block_reason"] = reason

    def _publish_inventory_dashboard(self, worker: "MarketWorker") -> None:
        cfg = worker.worker_config
        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        combined = matched_combined_average(snap)
        worker.dashboard["matched_pairs"] = round(snap.matched_pairs, 4)
        worker.dashboard["unpaired_yes"] = round(snap.unpaired_yes, 4)
        worker.dashboard["unpaired_no"] = round(snap.unpaired_no, 4)
        worker.dashboard["combined_avg_cost"] = round(combined, 4) if combined else 0.0
        worker.dashboard["combined_avg_cost_c"] = round(combined * 100, 2) if combined else 0.0
        worker.dashboard["combined_cost_target_c"] = round(cfg.target_combined_pair_cost * 100, 2)
        worker.dashboard["market_exposure_usd"] = round(snap.total_exposure_usd(), 4)

    def _entry_cutoff(self, worker: "MarketWorker", cycle: Optional[MarketCycleRecord]) -> bool:
        cfg = worker.worker_config
        if not entry_cutoff_active(worker.market_seconds_left(), cfg.no_buys_last_minutes):
            return False
        if cycle and cycle.phase == AccumulationPhase.WAITING_FOR_OPPOSITE_ENTRY:
            cycle.phase = AccumulationPhase.ENTRY_WINDOW_CLOSED
            cycle.metadata["close_reason"] = "UNMATCHED_POSITION_AT_CUTOFF"
            inventory_strategy_store.upsert(cycle)
            self._log_decision(worker, cycle, reason="UNMATCHED_POSITION_AT_CUTOFF")
        self._log_decision(worker, cycle, reason="REJECT_ENTRY_CUTOFF")
        return True

    def _market_ready(self, worker: "MarketWorker") -> Optional[str]:
        from bot import is_locked_price

        cfg = worker.worker_config
        if not cfg.inventory_accumulation_enabled:
            return "inventory_accumulation_disabled"
        if getattr(worker, "_ws_market_data_stale", False):
            return "REJECT_STALE_ORDERBOOK"
        if not worker.active_market:
            return "no_active_market"
        if not worker.canonical_market_id():
            return "missing_condition_id"
        if worker.market_seconds_left() <= 0:
            return "market_closed"
        if not worker.can_enter():
            return "outside_entry_window"
        y = float(worker.best_ask("YES"))
        n = float(worker.best_ask("NO"))
        if y <= 0 and n <= 0:
            return "missing_quotes"
        if y > 0 and is_locked_price(y):
            return "locked_outcome_price"
        if n > 0 and is_locked_price(n):
            return "locked_outcome_price"
        age = worker.max_quote_age_seconds()
        if age is not None and age > cfg.max_quote_age_seconds:
            return "REJECT_STALE_ORDERBOOK"
        return None

    def _side_entered(self, worker: "MarketWorker", cycle: MarketCycleRecord, side: str) -> bool:
        cfg = worker.worker_config
        if side_already_entered(cycle.metadata, side):
            return True
        if cfg.max_entries_per_side_per_market <= 0:
            return False
        return worker.inventory.shares(side) >= cfg.min_order_size - 1e-9

    def _sync_cycle_from_inventory(self, worker: "MarketWorker", cycle: MarketCycleRecord) -> None:
        from bot import MIN_SHARES

        cfg = worker.worker_config
        inv = worker.inventory
        entries = cycle.metadata.setdefault("entries", {})
        if inv.yes_shares >= MIN_SHARES - 1e-9 and not entries.get("YES", {}).get("filled"):
            entries["YES"] = {"filled": True, "avg_price": inv.avg_cost("YES"), "qty": inv.yes_shares}
        if inv.no_shares >= MIN_SHARES - 1e-9 and not entries.get("NO", {}).get("filled"):
            entries["NO"] = {"filled": True, "avg_price": inv.avg_cost("NO"), "qty": inv.no_shares}
        if inv.matched_pairs >= MIN_SHARES - 1e-9:
            cycle.phase = AccumulationPhase.PAIR_COMPLETED
            cycle.initial_hedge_complete = True
            cycle.target_achieved = True
        elif cycle.first_leg_side and self._side_entered(worker, cycle, cycle.first_leg_side):
            cycle.phase = AccumulationPhase.WAITING_FOR_OPPOSITE_ENTRY
        inventory_strategy_store.upsert(cycle)

    async def _record_fill(
        self,
        worker: "MarketWorker",
        side: str,
        qty: float,
        px: float,
        oid: Optional[str],
        *,
        cycle: MarketCycleRecord,
    ) -> None:
        if oid and not worker.record_fill_idempotent(oid, side, qty, px):
            return
        worker.inventory.record_buy(side, qty, px)
        worker.entry_count += 1
        worker.log_trade(side, px, "BUY", size=qty, order_id=oid)
        mark_side_filled(cycle.metadata, side, fill_price=px, qty=qty, order_id=oid or "")
        state = self._knife_state(cycle)
        record_buy_on_side(state, side, paired=False)
        self._persist_knife_state(cycle, state)
        inventory_strategy_store.upsert(cycle)

    def _falling_knife_blocks(
        self, worker: "MarketWorker", side: str, ask: float, cycle: MarketCycleRecord,
    ) -> Optional[str]:
        cfg = worker.worker_config
        if not cfg.falling_knife_filter_enabled:
            return None
        state = self._knife_state(cycle)
        block, reason, state = evaluate_falling_knife(
            side=side,
            ask=ask,
            history=worker.ask_history_for_knife(side),
            state=state,
            enabled=True,
            window_seconds=cfg.price_drop_window_seconds,
            fast_window_seconds=cfg.gabagool_falling_knife_fast_window_seconds,
            max_price_drop_fraction=cfg.max_price_drop_percent / 100.0,
            fast_drop_fraction=cfg.gabagool_falling_knife_fast_drop,
            cooldown_seconds=cfg.falling_knife_cooldown_seconds,
            recovery_cents=cfg.gabagool_falling_knife_recovery_cents,
            stabilization_window_seconds=cfg.stabilization_window_seconds,
            stabilization_tolerance_percent=cfg.stabilization_tolerance_percent,
            max_consecutive_buys=cfg.max_entries_per_side_per_market,
        )
        self._persist_knife_state(cycle, state)
        return reason if block else None

    async def _try_first_entry(self, worker: "MarketWorker", cycle: MarketCycleRecord, cid: str) -> None:
        if self._entry_cutoff(worker, cycle):
            return
        from bot import OrderState, MIN_SHARES

        cfg = worker.worker_config
        block = self._market_ready(worker)
        if block:
            worker.dashboard["entry_block_reason"] = block
            return

        for side in ("YES", "NO"):
            if self._side_entered(worker, cycle, side):
                self._log_decision(worker, cycle, reason="REJECT_FIRST_SIDE_ALREADY_ENTERED", extra={"side": side})
                return

        size = worker.entry_order_size(["YES", "NO"])
        if not size or size < cfg.min_order_size:
            worker.dashboard["entry_block_reason"] = "invalid_order_size"
            return

        yes_exe = worker.executable_buy("YES", float(size))
        no_exe = worker.executable_buy("NO", float(size))
        sel = pick_first_side(
            yes_exe,
            no_exe,
            min_price=cfg.cheap_side_min_price,
            max_price=cfg.cheap_side_max_price,
        )
        worker.dashboard["cheap_entry_candidate_side"] = sel.side or ""
        if sel.decision != SequentialDecision.BUY or not sel.side:
            self._log_decision(worker, cycle, reason=sel.reason, extra=sel.log_fields())
            return

        side = sel.side
        exe = yes_exe if side == "YES" else no_exe
        if not exe:
            return
        depth_err = check_executable_depth(exe, float(size), min_orderbook_depth=cfg.min_orderbook_depth)
        if depth_err:
            self._log_decision(worker, cycle, reason=depth_err, extra={"side": side})
            return
        knife = self._falling_knife_blocks(worker, side, sel.executable_price, cycle)
        if knife:
            self._log_decision(worker, cycle, reason=knife, extra={"side": side})
            return

        plan = worker.plan_accumulation_buy(
            side, float(size), purpose="cheap_entry", allow_first_leg_entry=True,
        )
        if plan.approval != BuyApproval.APPROVED:
            return

        async with worker._order_lock:
            if self._entry_cutoff(worker, cycle):
                return
            if worker.order_state != OrderState.IDLE:
                self._log_decision(worker, cycle, reason="REJECT_PENDING_ORDER_CONFLICT")
                return
            worker.order_state = OrderState.PENDING
            cycle.phase = AccumulationPhase.FIRST_ENTRY_PENDING
            inventory_strategy_store.upsert(cycle)
            qty, px, oid = await fok_buy_with_retries(
                worker,
                side,
                plan.approved_qty,
                purpose="cheap_entry",
                max_price_cap=cfg.cheap_side_max_price,
                max_retries=cfg.max_hedge_retries,
                retry_delay_ms=cfg.hedge_retry_delay_ms,
            )
            worker.order_state = OrderState.IDLE
            if qty <= MIN_SHARES - 1e-9:
                cycle.phase = AccumulationPhase.WAITING_FOR_FIRST_ENTRY
                inventory_strategy_store.upsert(cycle)
                worker.dashboard["entry_block_reason"] = "first_entry_fok_failed"
                return

            cycle.first_leg_side = side
            cycle.first_fill_at = time.time()
            await self._record_fill(worker, side, qty, px, oid, cycle=cycle)
            cycle.phase = AccumulationPhase.WAITING_FOR_OPPOSITE_ENTRY
            inventory_strategy_store.upsert(cycle)
            max_opp = max_opposite_gross_price(px, cfg.target_combined_pair_cost)
            self._log_decision(
                worker,
                cycle,
                reason="first_entry_filled",
                extra={
                    "side": side,
                    "fill_price": px,
                    "qty": qty,
                    "max_opposite_price": max_opp,
                    "target_combined_c": cfg.target_combined_pair_cost * 100,
                },
            )
            print(
                f"✅ [FIRST LEG] {worker.asset_type.upper()} {worker.window_slug} | "
                f"{side} x{qty:.2f} @ {px:.4f} | max opposite={max_opp:.4f} "
                f"(target pair={cfg.target_combined_pair_cost:.2f})"
            )

    async def _try_opposite_entry(self, worker: "MarketWorker", cycle: MarketCycleRecord) -> None:
        if cycle.phase == AccumulationPhase.PAIR_COMPLETED:
            return
        if self._entry_cutoff(worker, cycle):
            return
        from bot import OrderState, MIN_SHARES

        cfg = worker.worker_config

        first_side = cycle.first_leg_side
        if not first_side:
            self._sync_cycle_from_inventory(worker, cycle)
            first_side = cycle.first_leg_side
        if not first_side:
            return

        opp = _opposite(first_side)
        if self._side_entered(worker, cycle, opp):
            if worker.inventory.matched_pairs >= MIN_SHARES - 1e-9:
                cycle.phase = AccumulationPhase.PAIR_COMPLETED
                inventory_strategy_store.upsert(cycle)
                self._log_decision(worker, cycle, reason="PAIR_COMPLETED")
            return

        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        hedge_qty = min(
            snap.unpaired_yes if opp == "NO" else snap.unpaired_no,
            worker.inventory.headroom(opp, cfg.max_position_size),
            cfg.max_order_size,
        )
        if hedge_qty < cfg.min_order_size - 1e-9:
            return

        exe = worker.executable_buy(opp, hedge_qty)
        if not exe:
            worker.dashboard["entry_block_reason"] = "opposite_no_liquidity"
            return
        exe_px, filled = exe
        depth_err = check_executable_depth(exe, hedge_qty, min_orderbook_depth=cfg.min_orderbook_depth)
        if depth_err:
            self._log_decision(worker, cycle, reason=depth_err, extra={"side": opp})
            return

        ev = evaluate_opposite_with_entry_state(
            snap,
            cycle.metadata,
            first_leg_side=first_side,
            opposite_side=opp,
            opposite_qty=hedge_qty,
            opposite_executable=exe_px,
            target_combined_pair_cost=cfg.target_combined_pair_cost,
            max_combined_pair_cost=cfg.max_combined_pair_cost,
            min_net_profit_per_pair=cfg.min_net_profit_per_pair,
            slippage_reserve=self._slippage_reserve(worker),
        )
        worker.dashboard["max_opposite_price_c"] = round((ev.max_opposite_price or 0) * 100, 2)
        worker.dashboard["projected_gross_combined_c"] = round((ev.projected_gross_combined or 0) * 100, 2)
        if ev.decision == SequentialDecision.WAIT:
            self._log_decision(worker, cycle, reason=ev.reason, extra=ev.log_fields())
            return
        if ev.decision != SequentialDecision.BUY:
            self._log_decision(worker, cycle, reason=ev.reason, extra=ev.log_fields())
            return

        knife = self._falling_knife_blocks(worker, opp, exe_px, cycle)
        if knife:
            self._log_decision(worker, cycle, reason=knife, extra={"side": opp})
            return

        plan = worker.plan_accumulation_buy(
            opp, hedge_qty, purpose="initial_hedge", allow_first_leg_entry=False,
        )
        if plan.approval != BuyApproval.APPROVED:
            return

        first_row = cycle.metadata.get("entries", {}).get(first_side, {})
        first_avg = float(first_row.get("avg_price") or (snap.yes_avg() if first_side == "YES" else snap.no_avg()))
        max_opp_px = min(
            exe_px + cfg.max_slippage,
            max_opposite_gross_price(first_avg, cfg.max_combined_pair_cost),
        )

        async with worker._order_lock:
            if self._entry_cutoff(worker, cycle):
                return
            if worker.order_state != OrderState.IDLE:
                self._log_decision(worker, cycle, reason="REJECT_PENDING_ORDER_CONFLICT")
                return
            worker.order_state = OrderState.PENDING
            cycle.phase = AccumulationPhase.OPPOSITE_ENTRY_PENDING
            inventory_strategy_store.upsert(cycle)
            qty, px, oid = await fok_buy_with_retries(
                worker,
                opp,
                plan.approved_qty,
                purpose="initial_hedge",
                max_price_cap=max_opp_px,
                max_retries=cfg.max_hedge_retries,
                retry_delay_ms=cfg.hedge_retry_delay_ms,
            )
            worker.order_state = OrderState.IDLE
            if qty <= MIN_SHARES - 1e-9:
                cycle.phase = AccumulationPhase.WAITING_FOR_OPPOSITE_ENTRY
                inventory_strategy_store.upsert(cycle)
                return

            await self._record_fill(worker, opp, qty, px, oid, cycle=cycle)
            cycle.phase = AccumulationPhase.PAIR_COMPLETED
            cycle.initial_hedge_complete = True
            cycle.target_achieved = True
            combined = matched_combined_average(
                InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
            )
            cycle.metadata["pair_combined_avg"] = combined
            inventory_strategy_store.upsert(cycle)
            self._log_decision(
                worker,
                cycle,
                reason="PAIR_COMPLETED",
                extra={"combined_avg": combined, "side": opp, "fill_price": px},
            )
            comb_s = f"{combined:.4f}" if combined is not None else "n/a"
            print(
                f"✅ [OPPOSITE LEG] {worker.asset_type.upper()} {worker.window_slug} | "
                f"{opp} x{qty:.2f} @ {px:.4f} | combined={comb_s}"
            )

    async def on_market_tick(self, worker: "MarketWorker") -> None:
        from bot import OrderState, asset_cooldown, is_trading_allowed, log_cooldown_block

        cfg = worker.worker_config
        wkey = self._key(worker)
        self._publish_inventory_dashboard(worker)

        cid = worker.canonical_market_id()
        if not cid:
            return

        has_position = worker.inventory.yes_shares > 0 or worker.inventory.no_shares > 0
        cycle = inventory_strategy_store.cycle_for_market(wkey, cid)
        if cycle is None and (has_position or cfg.inventory_accumulation_enabled):
            cycle = inventory_strategy_store.get_or_create_cycle(wkey, cid)
        if cycle is None:
            return

        self._sync_cycle_from_inventory(worker, cycle)

        if cycle.phase in (AccumulationPhase.PAIR_COMPLETED, AccumulationPhase.ENTRY_WINDOW_CLOSED):
            worker.dashboard["trade_phase"] = cycle.phase.value
            return

        if worker.order_state == OrderState.PENDING or worker._order_lock.locked():
            return

        if not is_trading_allowed():
            worker.dashboard["entry_block_reason"] = "trading_schedule_blocked"
            return
        if asset_cooldown.is_entry_blocked(worker.asset_type, worker.window_slug) and not has_position:
            log_cooldown_block(worker.asset_type, worker.window_slug, "INV", 0)
            worker.dashboard["entry_block_reason"] = "asset_cooldown"
            return

        if cycle.phase in (
            AccumulationPhase.WAITING_FOR_FIRST_ENTRY,
            AccumulationPhase.FIRST_ENTRY_PENDING,
        ):
            if not has_position:
                await self._try_first_entry(worker, cycle, cid)
        elif cycle.phase in (
            AccumulationPhase.WAITING_FOR_OPPOSITE_ENTRY,
            AccumulationPhase.OPPOSITE_ENTRY_PENDING,
        ):
            await self._try_opposite_entry(worker, cycle)

        worker.dashboard["trade_phase"] = cycle.phase.value

    async def on_binance_tick(self, worker: "MarketWorker") -> None:
        await self.on_market_tick(worker)

    async def evaluate(self, worker: "MarketWorker"):
        return None

    async def execute(self, worker: "MarketWorker", decision) -> None:
        return None


MomentumStrategy = InventoryAccumulationStrategy
GabagoolStrategy = InventoryAccumulationStrategy
SpreadCaptureStrategy = InventoryAccumulationStrategy
LateEntryStrategy = InventoryAccumulationStrategy
BinanceTriggeredFokStrategy = InventoryAccumulationStrategy
BinanceFokStrategy = InventoryAccumulationStrategy
