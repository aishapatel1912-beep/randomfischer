"""Inventory accumulation with profitability-first gates and falling-knife protection."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple

from strategies.inventory_accumulation_execution import fok_buy_with_retries, fok_paired_buy
from utils.falling_knife_filter import (
    FallingKnifeState,
    evaluate_falling_knife,
    record_buy_on_side,
)
from utils.profitability_gate import (
    check_executable_depth,
    evaluate_first_leg_gate,
)
from utils.inventory_balance import (
    BuyApproval,
    log_buy_plan,
    plan_paired_buy,
    plan_single_leg_buy,
    underrepresented_side,
)
from utils.inventory_cost import (
    InventorySnapshot,
    PurchaseDecision,
    evaluate_dca_purchase,
    evaluate_initial_hedge,
    evaluate_paired_dca_purchase,
    matched_combined_average,
    pick_cheap_side,
)
from utils.inventory_strategy_store import (
    AccumulationPhase,
    MarketCycleRecord,
    inventory_strategy_store,
)

if TYPE_CHECKING:
    from bot import MarketWorker

_LOG = logging.getLogger("inventory_accumulation")


def _opposite(side: str) -> str:
    return "NO" if side.upper() == "YES" else "YES"


class InventoryAccumulationStrategy:
    def _key(self, worker: "MarketWorker") -> str:
        return worker.worker_config.key

    def _publish_inventory_dashboard(self, worker: "MarketWorker") -> None:
        cfg = worker.worker_config
        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        combined = matched_combined_average(snap)
        worker.dashboard["matched_pairs"] = round(snap.matched_pairs, 4)
        worker.dashboard["unpaired_yes"] = round(snap.unpaired_yes, 4)
        worker.dashboard["unpaired_no"] = round(snap.unpaired_no, 4)
        worker.dashboard["combined_avg_cost"] = round(combined, 4) if combined else 0.0
        worker.dashboard["combined_avg_cost_c"] = round(combined * 100, 2) if combined else 0.0
        worker.dashboard["combined_cost_target_c"] = round(cfg.combined_cost_target * 100, 2)
        worker.dashboard["market_exposure_usd"] = round(snap.total_exposure_usd(), 4)

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

    def _log_gate_decision(
        self,
        worker: "MarketWorker",
        *,
        side: str,
        qty: float,
        purpose: str,
        allowed: bool,
        reason: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        cfg = worker.worker_config
        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        payload = {
            "market": worker.canonical_market_id() or "",
            "purpose": purpose,
            "side": side,
            "qty": qty,
            "yes_ask": worker.best_ask("YES"),
            "no_ask": worker.best_ask("NO"),
            "yes_shares": snap.yes_shares,
            "no_shares": snap.no_shares,
            "matched": snap.matched_pairs,
            "unmatched_yes": snap.unpaired_yes,
            "unmatched_no": snap.unpaired_no,
            "allowed": allowed,
            "reason": reason,
            "ts": time.time(),
        }
        if extra:
            payload.update(extra)
        worker.dashboard["last_gate_reason"] = reason
        _LOG.info(json.dumps({"event": "accumulation_gate", **payload}))
        if not allowed:
            worker.dashboard["entry_block_reason"] = reason

    def _falling_knife_blocks(
        self, worker: "MarketWorker", side: str, ask: float, cycle: Optional[MarketCycleRecord],
    ) -> Optional[str]:
        cfg = worker.worker_config
        if not cfg.falling_knife_filter_enabled:
            return None
        state = self._knife_state(cycle)
        hist = worker.ask_history_for_knife(side)
        block, reason, state = evaluate_falling_knife(
            side=side,
            ask=ask,
            history=hist,
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
            max_consecutive_buys=cfg.max_consecutive_buys_per_outcome,
        )
        self._persist_knife_state(cycle, state)
        if block:
            worker.dashboard["falling_knife_blocked"] = True
            worker.dashboard["falling_knife_side"] = side
            return reason
        return None

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

    def _risk_blocks_buy(
        self, worker: "MarketWorker", side: str, qty: float, price: float,
        *, for_initial_hedge: bool = False,
    ) -> Optional[str]:
        cfg = worker.worker_config
        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        projected_cost = snap.total_exposure_usd() + qty * price
        if projected_cost > cfg.max_market_exposure + 1e-9:
            return "max_market_exposure"
        if worker.inventory.shares(side) + qty > cfg.max_position_size + 1e-9:
            return "max_position_size"
        if not for_initial_hedge and cfg.allow_unmatched_accumulation:
            if side == "YES" and snap.unpaired_yes + qty > cfg.max_unmatched_yes + 1e-9:
                return "max_unmatched_yes"
            if side == "NO" and snap.unpaired_no + qty > cfg.max_unmatched_no + 1e-9:
                return "max_unmatched_no"
        if qty < cfg.min_order_size - 1e-9:
            return "below_min_order_size"
        if not worker.validate_order_size(side, qty):
            return "invalid_order_size"
        return None

    async def _record_fill(
        self,
        worker: "MarketWorker",
        side: str,
        qty: float,
        px: float,
        oid: Optional[str],
        *,
        cycle: Optional[MarketCycleRecord] = None,
        paired: bool = False,
    ) -> None:
        if oid and not worker.record_fill_idempotent(oid, side, qty, px):
            return
        worker.inventory.record_buy(side, qty, px)
        worker.entry_count += 1
        worker.log_trade(side, px, "BUY", size=qty, order_id=oid)
        if cycle is not None:
            state = self._knife_state(cycle)
            record_buy_on_side(state, side, paired=paired)
            self._persist_knife_state(cycle, state)

    async def _try_cheap_entry(
        self, worker: "MarketWorker", cycle: Optional[MarketCycleRecord], cid: str,
    ) -> None:
        from bot import OrderState, MIN_SHARES

        cfg = worker.worker_config
        wkey = self._key(worker)
        if cycle and cycle.condition_id == cid and cycle.phase != AccumulationPhase.IDLE:
            return
        if worker.market_seconds_left() <= 0:
            worker.dashboard["entry_block_reason"] = "market_closed"
            return

        size = worker.entry_order_size(["YES", "NO"])
        if not size or size < cfg.min_order_size:
            worker.dashboard["entry_block_reason"] = "invalid_order_size"
            return

        yes_exe = worker.executable_buy("YES", float(size))
        no_exe = worker.executable_buy("NO", float(size))
        if yes_exe:
            worker.dashboard["yes_executable_c"] = round(yes_exe[0] * 100, 1)
        if no_exe:
            worker.dashboard["no_executable_c"] = round(no_exe[0] * 100, 1)
        side, ev = pick_cheap_side(
            yes_exe,
            no_exe,
            cheap_entry_price=cfg.cheap_entry_price,
            max_entry_price=cfg.max_entry_price,
            yes_shares=worker.inventory.yes_shares,
            no_shares=worker.inventory.no_shares,
        )
        worker.dashboard["cheap_entry_candidate_side"] = side or ""
        if not side or ev.decision != PurchaseDecision.BUY:
            worker.dashboard["entry_block_reason"] = ev.reason
            return

        opp = _opposite(side)
        opp_exe = worker.executable_buy(opp, float(size))
        opp_px = opp_exe[0] if opp_exe else worker.best_ask(opp)
        depth_err = check_executable_depth(
            yes_exe if side == "YES" else no_exe,
            float(size),
            min_orderbook_depth=cfg.min_orderbook_depth,
        )
        if depth_err:
            self._log_gate_decision(
                worker, side=side, qty=float(size), purpose="cheap_entry", allowed=False, reason=depth_err,
            )
            return
        first_leg = evaluate_first_leg_gate(
            side=side,
            qty=float(size),
            leg_executable_price=ev.executable_price,
            opposite_executable_price=float(opp_px or 0),
            fee_rate=cfg.spread_capture_fee_rate,
            max_combined_cost=cfg.max_combined_cost,
            min_net_profit_per_pair=cfg.min_net_profit_per_pair,
            slippage_reserve=self._slippage_reserve(worker),
            max_entry_price=cfg.max_entry_price,
        )
        if not first_leg.allowed:
            self._log_gate_decision(
                worker,
                side=side,
                qty=float(size),
                purpose="cheap_entry",
                allowed=False,
                reason=first_leg.reason,
                extra=first_leg.log_fields(),
            )
            return
        knife = self._falling_knife_blocks(worker, side, ev.executable_price, cycle)
        if knife:
            self._log_gate_decision(
                worker, side=side, qty=float(size), purpose="cheap_entry", allowed=False, reason=knife,
            )
            return

        plan = worker.plan_accumulation_buy(
            side, float(size), purpose="cheap_entry", allow_first_leg_entry=True,
        )
        if plan.approval != BuyApproval.APPROVED:
            return

        block = self._risk_blocks_buy(worker, side, plan.approved_qty, ev.executable_price)
        if block:
            worker.dashboard["entry_block_reason"] = block
            return

        async with worker._order_lock:
            if worker.order_state != OrderState.IDLE:
                return
            plan = worker.plan_accumulation_buy(
                side, float(size), purpose="cheap_entry", allow_first_leg_entry=True,
            )
            if plan.approval != BuyApproval.APPROVED:
                return
            worker.order_state = OrderState.PENDING
            qty, px, oid = await fok_buy_with_retries(
                worker,
                side,
                plan.approved_qty,
                purpose="cheap_entry",
                max_price_cap=cfg.max_entry_price,
                max_retries=cfg.max_hedge_retries,
                retry_delay_ms=cfg.hedge_retry_delay_ms,
            )
            worker.order_state = OrderState.IDLE
            if qty <= MIN_SHARES - 1e-9:
                worker.dashboard["entry_block_reason"] = "cheap_entry_fok_failed"
                return

            rec = inventory_strategy_store.start_cycle(
                wkey,
                cid,
                first_leg_side=side,
                hedge_delay_seconds=cfg.initial_hedge_delay_seconds,
            )
            await self._record_fill(worker, side, qty, px, oid, cycle=rec)
            worker.dashboard["trade_phase"] = rec.phase.value
            worker.dashboard["entry_block_reason"] = ""
            print(
                f"✅ [CHEAP ENTRY] {worker.asset_type.upper()} {worker.window_slug} | "
                f"{side} x{qty:.2f} @ {px:.4f} | hedge in {cfg.initial_hedge_delay_seconds}s"
            )

    async def _try_initial_hedge(self, worker: "MarketWorker", cycle: MarketCycleRecord) -> None:
        from bot import OrderState, MIN_SHARES

        cfg = worker.worker_config
        if cycle.initial_hedge_complete:
            return
        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        if snap.matched_pairs > MIN_SHARES - 1e-9 and snap.unpaired_yes < MIN_SHARES and snap.unpaired_no < MIN_SHARES:
            cycle.initial_hedge_complete = True
            cycle.phase = AccumulationPhase.ACCUMULATING
            inventory_strategy_store.upsert(cycle)
            return

        if cycle.first_leg_side == "YES":
            hedge_side = "NO"
            hedge_qty = snap.unpaired_yes
        else:
            hedge_side = "YES"
            hedge_qty = snap.unpaired_no
        if hedge_qty <= MIN_SHARES - 1e-9:
            return

        hedge_qty = min(
            hedge_qty,
            worker.inventory.headroom(hedge_side, cfg.max_position_size),
            cfg.max_order_size,
        )
        if hedge_qty < cfg.min_order_size:
            return

        exe = worker.executable_buy(hedge_side, hedge_qty)
        if not exe:
            cycle.hedge_retries += 1
            inventory_strategy_store.upsert(cycle)
            worker.dashboard["entry_block_reason"] = "hedge_no_liquidity"
            return

        exe_px, _ = exe
        depth_err = check_executable_depth(exe, hedge_qty, min_orderbook_depth=cfg.min_orderbook_depth)
        if depth_err:
            worker.dashboard["entry_block_reason"] = depth_err
            return
        knife = self._falling_knife_blocks(worker, hedge_side, exe_px, cycle)
        if knife:
            self._log_gate_decision(
                worker, side=hedge_side, qty=hedge_qty, purpose="initial_hedge", allowed=False, reason=knife,
            )
            return

        limits = evaluate_initial_hedge(
            snap,
            hedge_side,
            hedge_qty,
            exe_px,
            max_slippage=cfg.max_slippage,
            limit_price=exe_px + cfg.max_slippage,
            max_combined_cost=cfg.max_combined_cost,
            min_net_profit_per_pair=cfg.min_net_profit_per_pair,
            slippage_reserve=self._slippage_reserve(worker),
        )
        if limits.decision != PurchaseDecision.BUY:
            worker.dashboard["entry_block_reason"] = limits.reason
            self._log_gate_decision(
                worker,
                side=hedge_side,
                qty=hedge_qty,
                purpose="initial_hedge",
                allowed=False,
                reason=limits.reason,
                extra={"projected_combined": limits.resulting_combined},
            )
            return

        plan = worker.plan_accumulation_buy(
            hedge_side, hedge_qty, purpose="initial_hedge", allow_first_leg_entry=False,
        )
        if plan.approval != BuyApproval.APPROVED:
            return
        hedge_qty = plan.approved_qty

        block = self._risk_blocks_buy(
            worker, hedge_side, hedge_qty, exe_px, for_initial_hedge=True,
        )
        if block:
            worker.dashboard["entry_block_reason"] = block
            return

        async with worker._order_lock:
            if worker.order_state != OrderState.IDLE:
                return
            plan = worker.plan_accumulation_buy(
                hedge_side, hedge_qty, purpose="initial_hedge", allow_first_leg_entry=False,
            )
            if plan.approval != BuyApproval.APPROVED:
                return
            hedge_qty = plan.approved_qty
            worker.order_state = OrderState.PENDING
            held_avg = snap.yes_avg() if hedge_side == "NO" else snap.no_avg()
            max_hedge_px = min(exe_px + cfg.max_slippage, max(0.01, cfg.max_combined_cost - held_avg))
            qty, px, oid = await fok_buy_with_retries(
                worker,
                hedge_side,
                hedge_qty,
                purpose="initial_hedge",
                max_price_cap=max_hedge_px if max_hedge_px > 0 else None,
                max_retries=cfg.max_hedge_retries,
                retry_delay_ms=cfg.hedge_retry_delay_ms,
            )
            worker.order_state = OrderState.IDLE
            if qty <= MIN_SHARES - 1e-9:
                cycle.hedge_retries += 1
                inventory_strategy_store.upsert(cycle)
                return

            await self._record_fill(worker, hedge_side, qty, px, oid, cycle=cycle)
            cycle.initial_hedge_complete = True
            cycle.phase = AccumulationPhase.ACCUMULATING
            combined = matched_combined_average(
                InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
            )
            cycle.metadata["initial_hedge_combined"] = combined
            inventory_strategy_store.upsert(cycle)
            comb_s = f"{combined:.4f}" if combined is not None else "n/a"
            print(
                f"✅ [INITIAL HEDGE] {worker.asset_type.upper()} {worker.window_slug} | "
                f"{hedge_side} x{qty:.2f} @ {px:.4f} | combined={comb_s}"
            )

    async def _try_dca(self, worker: "MarketWorker", cycle: MarketCycleRecord) -> None:
        from bot import OrderState, MIN_SHARES

        cfg = worker.worker_config
        if not cycle.initial_hedge_complete:
            return
        if cycle.metadata.get("paired_dca_paused"):
            worker.dashboard["entry_block_reason"] = "paired_dca_paused_rebalance_required"
            await self._try_dca_rebalance(worker, cycle)
            return

        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        if snap.matched_pairs <= MIN_SHARES - 1e-9:
            return

        eff = worker.effective_inventory()
        size = min(cfg.max_order_size, cfg.order_size_max)
        market = worker.active_market.get("question", "") if worker.active_market else ""

        if eff.share_imbalance_effective > cfg.max_share_imbalance + 1e-9:
            await self._try_dca_rebalance(worker, cycle)
            return

        paired_plan = plan_paired_buy(
            snap=snap,
            eff=eff,
            requested_qty=size,
            min_order_size=cfg.min_order_size,
            max_share_imbalance=cfg.max_share_imbalance,
            balance_inventory_before_buy=cfg.balance_inventory_before_buy,
        )
        if paired_plan.approval != BuyApproval.APPROVED:
            worker.dashboard["entry_block_reason"] = paired_plan.reason
            log_buy_plan(paired_plan.yes_plan, market=market, purpose="paired_dca")
            return

        qty = paired_plan.qty
        yes_exe = worker.executable_buy("YES", qty)
        no_exe = worker.executable_buy("NO", qty)
        if not yes_exe or not no_exe:
            worker.dashboard["entry_block_reason"] = "paired_dca_no_liquidity"
            return
        yes_px, no_px = yes_exe[0], no_exe[0]
        if check_executable_depth(yes_exe, qty, min_orderbook_depth=cfg.min_orderbook_depth):
            worker.dashboard["entry_block_reason"] = "REJECT_INSUFFICIENT_DEPTH"
            return
        if check_executable_depth(no_exe, qty, min_orderbook_depth=cfg.min_orderbook_depth):
            worker.dashboard["entry_block_reason"] = "REJECT_INSUFFICIENT_DEPTH"
            return
        for s, px in (("YES", yes_px), ("NO", no_px)):
            knife = self._falling_knife_blocks(worker, s, px, cycle)
            if knife:
                worker.dashboard["entry_block_reason"] = knife
                return

        ev = evaluate_paired_dca_purchase(
            snap,
            qty,
            yes_px,
            no_px,
            combined_cost_target=cfg.combined_cost_target,
            max_combined_cost=cfg.max_combined_cost,
            min_net_profit_per_pair=cfg.min_net_profit_per_pair,
            slippage_reserve=self._slippage_reserve(worker),
        )
        if ev.decision != PurchaseDecision.BUY:
            worker.dashboard["entry_block_reason"] = ev.reason
            return

        block_y = self._risk_blocks_buy(worker, "YES", qty, yes_px)
        block_n = self._risk_blocks_buy(worker, "NO", qty, no_px)
        if block_y or block_n:
            worker.dashboard["entry_block_reason"] = block_y or block_n
            return

        async with worker._order_lock:
            if worker.order_state != OrderState.IDLE:
                return
            eff = worker.effective_inventory()
            paired_plan = plan_paired_buy(
                snap=InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate),
                eff=eff,
                requested_qty=qty,
                min_order_size=cfg.min_order_size,
                max_share_imbalance=cfg.max_share_imbalance,
                balance_inventory_before_buy=cfg.balance_inventory_before_buy,
            )
            if paired_plan.approval != BuyApproval.APPROVED:
                worker.dashboard["entry_block_reason"] = paired_plan.reason
                return
            worker.order_state = OrderState.PENDING
            yq, yp, nq, np, y_oid, n_oid = await fok_paired_buy(
                worker,
                paired_plan.qty,
                purpose="paired_dca",
                max_retries=cfg.max_hedge_retries,
                retry_delay_ms=cfg.hedge_retry_delay_ms,
            )
            worker.order_state = OrderState.IDLE
            if yq <= MIN_SHARES - 1e-9:
                return
            await self._record_fill(worker, "YES", yq, yp, y_oid, cycle=cycle, paired=False)
            if nq <= MIN_SHARES - 1e-9:
                cycle.metadata["paired_dca_paused"] = True
                inventory_strategy_store.upsert(cycle)
                return
            await self._record_fill(worker, "NO", nq, np, n_oid, cycle=cycle, paired=True)
            if ev.target_achieved:
                cycle.target_achieved = True
                cycle.phase = AccumulationPhase.TARGET_ACHIEVED
            inventory_strategy_store.upsert(cycle)
            combined = matched_combined_average(
                InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
            )
            comb_s = f"{combined:.4f}" if combined is not None else "n/a"
            print(
                f"📉 [PAIRED DCA] {worker.asset_type.upper()} {worker.window_slug} | "
                f"YES x{yq:.2f} @ {yp:.4f} + NO x{nq:.2f} @ {np:.4f} | combined={comb_s}"
            )

    async def _try_dca_rebalance(self, worker: "MarketWorker", cycle: MarketCycleRecord) -> None:
        from bot import OrderState, MIN_SHARES

        cfg = worker.worker_config
        snap = InventorySnapshot.from_ledger(worker.inventory, fee_rate=cfg.spread_capture_fee_rate)
        eff = worker.effective_inventory()
        side = underrepresented_side(eff)
        if not side:
            return

        gap = abs(eff.effective_yes - eff.effective_no)
        size = min(cfg.max_order_size, cfg.order_size_max, gap)
        plan = plan_single_leg_buy(
            snap=snap,
            eff=eff,
            side=side,
            requested_qty=size,
            min_order_size=cfg.min_order_size,
            max_share_imbalance=cfg.max_share_imbalance,
            max_unmatched_shares=cfg.max_unmatched_shares,
            balance_inventory_before_buy=cfg.balance_inventory_before_buy,
            allow_unmatched_accumulation=cfg.allow_unmatched_accumulation,
            purpose="dca_rebalance",
            allow_first_leg_entry=False,
        )
        market = worker.active_market.get("question", "") if worker.active_market else ""
        log_buy_plan(plan, market=market, purpose="dca_rebalance")
        if plan.approval != BuyApproval.APPROVED:
            worker.dashboard["entry_block_reason"] = plan.reason
            return

        exe = worker.executable_buy(side, plan.approved_qty)
        if not exe:
            worker.dashboard["entry_block_reason"] = "rebalance_no_liquidity"
            return
        exe_px, _ = exe
        knife = self._falling_knife_blocks(worker, side, exe_px, cycle)
        if knife:
            worker.dashboard["entry_block_reason"] = knife
            return
        ev = evaluate_dca_purchase(
            snap,
            side,
            plan.approved_qty,
            exe_px,
            combined_cost_target=cfg.combined_cost_target,
            max_combined_cost=cfg.max_combined_cost,
            require_improvement=False,
            min_net_profit_per_pair=cfg.min_net_profit_per_pair,
            slippage_reserve=self._slippage_reserve(worker),
        )
        if ev.decision != PurchaseDecision.BUY:
            worker.dashboard["entry_block_reason"] = ev.reason
            return

        block = self._risk_blocks_buy(worker, side, plan.approved_qty, exe_px)
        if block:
            worker.dashboard["entry_block_reason"] = block
            return

        async with worker._order_lock:
            if worker.order_state != OrderState.IDLE:
                return
            plan = worker.plan_accumulation_buy(side, size, purpose="dca_rebalance")
            if plan.approval != BuyApproval.APPROVED:
                return
            worker.order_state = OrderState.PENDING
            qty, px, oid = await fok_buy_with_retries(
                worker,
                side,
                plan.approved_qty,
                purpose="dca_rebalance",
                max_retries=cfg.max_hedge_retries,
                retry_delay_ms=cfg.hedge_retry_delay_ms,
            )
            worker.order_state = OrderState.IDLE
            if qty <= MIN_SHARES - 1e-9:
                return
            await self._record_fill(worker, side, qty, px, oid, cycle=cycle)
            eff_after = worker.effective_inventory()
            if eff_after.share_imbalance_effective <= cfg.max_share_imbalance + 1e-9:
                cycle.metadata.pop("paired_dca_paused", None)
            inventory_strategy_store.upsert(cycle)
            print(
                f"⚖️ [REBALANCE] {worker.asset_type.upper()} {worker.window_slug} | "
                f"{side} x{qty:.2f} @ {px:.4f} | imb={eff_after.share_imbalance_effective:.2f}"
            )

    async def on_market_tick(self, worker: "MarketWorker") -> None:
        from bot import OrderState, MIN_SHARES, asset_cooldown, is_trading_allowed, log_cooldown_block

        cfg = worker.worker_config
        wkey = self._key(worker)
        self._publish_inventory_dashboard(worker)

        block = self._market_ready(worker)
        has_position = worker.inventory.yes_shares > 0 or worker.inventory.no_shares > 0
        if block and not has_position:
            worker.dashboard["entry_block_reason"] = block
            return

        if not is_trading_allowed():
            worker.dashboard["entry_block_reason"] = "trading_schedule_blocked"
            return
        if asset_cooldown.is_entry_blocked(worker.asset_type, worker.window_slug) and not has_position:
            log_cooldown_block(worker.asset_type, worker.window_slug, "INV", 0)
            worker.dashboard["entry_block_reason"] = "asset_cooldown"
            return

        cid = worker.canonical_market_id()
        if not cid:
            return

        cycle = inventory_strategy_store.active_cycle(wkey)
        if cycle and cycle.condition_id != cid:
            cycle = None

        if worker.order_state == OrderState.PENDING or worker._order_lock.locked():
            return

        if cycle is None and not has_position:
            if block:
                return
            await self._try_cheap_entry(worker, cycle, cid)
            return

        if cycle is None and has_position:
            snap = InventorySnapshot.from_ledger(
                worker.inventory, fee_rate=cfg.spread_capture_fee_rate,
            )
            if snap.unpaired_yes > MIN_SHARES - 1e-9:
                recovered_leg = "YES"
            elif snap.unpaired_no > MIN_SHARES - 1e-9:
                recovered_leg = "NO"
            elif worker.inventory.yes_shares > worker.inventory.no_shares + 1e-9:
                recovered_leg = "YES"
            elif worker.inventory.no_shares > worker.inventory.yes_shares + 1e-9:
                recovered_leg = "NO"
            else:
                recovered_leg = "YES"
            cycle = inventory_strategy_store.start_cycle(
                wkey, cid, first_leg_side=recovered_leg,
                hedge_delay_seconds=0,
            )
            cycle.initial_hedge_complete = worker.inventory.matched_pairs > 0
            cycle.phase = AccumulationPhase.ACCUMULATING if cycle.initial_hedge_complete else AccumulationPhase.FIRST_LEG_OPEN
            inventory_strategy_store.upsert(cycle)

        if cycle.phase in (AccumulationPhase.FIRST_LEG_OPEN, AccumulationPhase.HEDGE_DUE):
            now = time.time()
            if now >= cycle.hedge_deadline_at:
                cycle.phase = AccumulationPhase.HEDGE_DUE
                inventory_strategy_store.upsert(cycle)
                await self._try_initial_hedge(worker, cycle)
            else:
                worker.dashboard["entry_block_reason"] = "awaiting_initial_hedge_delay"
                worker.dashboard["hedge_seconds_left"] = round(cycle.hedge_deadline_at - now, 1)

        if cycle.phase in (AccumulationPhase.ACCUMULATING, AccumulationPhase.TARGET_ACHIEVED):
            if worker.market_seconds_left() > 0 or has_position:
                await self._try_dca(worker, cycle)

        worker.dashboard["trade_phase"] = cycle.phase.value

    async def on_binance_tick(self, worker: "MarketWorker") -> None:
        """Fallback tick to process hedge deadline and DCA if Polymarket WS is quiet."""
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
