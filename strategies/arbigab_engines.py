"""Three-engine Arbigab-style strategy layer.

This module reconstructs the *publicly described mechanics* of Arbigab:
Momentum, Market Making, and Spread Capture.  It is not private Arbigab source.
The engine layer deliberately delegates order placement/fill accounting to the
existing MarketWorker so the user's CLOB, inventory and settlement code stays
intact.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from bot import MarketWorker

_EPS = 1e-9


@dataclass(frozen=True)
class EconomicPosition:
    yes_shares: float
    no_shares: float
    yes_avg: float
    no_avg: float
    invested: float
    position_delta: float
    pnl_if_yes: float
    pnl_if_no: float
    mark_pnl: float
    matched_pairs: float
    locked_profit: float

    @property
    def unpaired(self) -> float:
        return abs(self.yes_shares - self.no_shares)

    @property
    def combined_avg(self) -> float:
        if self.yes_shares <= _EPS or self.no_shares <= _EPS:
            return 0.0
        return self.yes_avg + self.no_avg


@dataclass(frozen=True)
class EngineDecision:
    engine: str
    side: str = ""
    action: str = "BUY"
    price: float = 0.0
    size: float = 0.0
    reason: str = ""
    score: float = 0.0
    second_side: str = ""
    second_action: str = ""
    second_price: float = 0.0


def economic_position(worker: "MarketWorker") -> EconomicPosition:
    inv = worker.inventory
    ys = float(inv.shares("YES"))
    ns = float(inv.shares("NO"))
    ya = float(inv.avg_cost("YES")) if ys > _EPS else 0.0
    na = float(inv.avg_cost("NO")) if ns > _EPS else 0.0
    invested = ys * ya + ns * na
    # Resolution economics: YES wins => YES shares pay $1, NO pays $0.
    # DOWN/NO wins => inverse.  This is the core risk/accounting view.
    pnl_yes = ys - invested
    pnl_no = ns - invested
    y_bid = worker.best_bid("YES")
    n_bid = worker.best_bid("NO")
    mark_value = ys * max(y_bid, 0.0) + ns * max(n_bid, 0.0)
    mark_pnl = mark_value - invested
    matched = min(ys, ns)
    locked = max(0.0, matched * (1.0 - ya - na)) if matched > _EPS else 0.0
    return EconomicPosition(
        yes_shares=ys, no_shares=ns, yes_avg=ya, no_avg=na,
        invested=invested, position_delta=ys - ns,
        pnl_if_yes=pnl_yes, pnl_if_no=pnl_no, mark_pnl=mark_pnl,
        matched_pairs=matched, locked_profit=locked,
    )


def _tick_up(p: float) -> float:
    return round(math.ceil((p - _EPS) * 100.0) / 100.0, 2)


def _tick_down(p: float) -> float:
    return round(math.floor((p + _EPS) * 100.0) / 100.0, 2)


class BinancePreemptiveCancel:
    """Shared adverse-selection gate for every worker.

    A sharp Binance move cancels the resting token most likely to be filled
    toxically before new quotes are considered.  The signal is deliberately
    conservative: stale Binance data never triggers a cancellation.
    """

    @staticmethod
    def exposed_side(worker: "MarketWorker") -> Optional[str]:
        signal = worker.binance_signal()
        if signal is None or not signal.is_fresh:
            return None
        # Positive Binance move makes resting NO/Down bids vulnerable;
        # negative move makes resting YES/Up bids vulnerable.
        if signal.price_delta <= -worker.worker_config.preemptive_cancel_price_delta:
            return "YES"
        if signal.price_delta >= worker.worker_config.preemptive_cancel_price_delta:
            return "NO"
        if signal.momentum <= -worker.worker_config.preemptive_cancel_momentum:
            return "YES"
        if signal.momentum >= worker.worker_config.preemptive_cancel_momentum:
            return "NO"
        if signal.imbalance <= -worker.worker_config.preemptive_cancel_imbalance:
            return "YES"
        if signal.imbalance >= worker.worker_config.preemptive_cancel_imbalance:
            return "NO"
        return None

    async def run(self, worker: "MarketWorker") -> bool:
        if not worker.worker_config.preemptive_cancel_enabled:
            return False
        side = self.exposed_side(worker)
        if not side:
            worker.dashboard["preemptive_cancel_active"] = False
            return False
        worker.dashboard["preemptive_cancel_active"] = True
        worker.dashboard["preemptive_cancel_side"] = side
        cancelled = await worker.cancel_engine_orders(side=side)
        if cancelled:
            print(
                f"🚨 [PREEMPTIVE CANCEL] {worker.asset_type.upper()} {worker.window_slug} | "
                f"Binance adverse move -> cancelled {side} resting exposure ({cancelled})"
            )
        return cancelled > 0


class MomentumEngine:
    """Binance-to-Polymarket latency engine."""

    async def evaluate(self, worker: "MarketWorker") -> Optional[EngineDecision]:
        cfg = worker.worker_config
        if not cfg.momentum_engine_enabled or not worker.can_enter():
            return None
        signal = worker.binance_signal()
        if signal is None or not signal.is_fresh or not signal.price_ready:
            return None
        if abs(signal.price_delta) < cfg.momentum_min_delta:
            return None
        if signal.age_seconds > cfg.momentum_signal_max_age:
            return None

        side = "YES" if signal.price_delta > 0 else "NO"
        score = abs(signal.price_delta)
        pos = economic_position(worker)
        # Do not add directional risk when the same side is already materially
        # overweight.  A later risk pass can still reduce that inventory.
        if (side == "YES" and pos.position_delta >= cfg.momentum_max_position_delta) or (
            side == "NO" and pos.position_delta <= -cfg.momentum_max_position_delta
        ):
            return None
        ask = worker.best_ask(side)
        bid = worker.best_bid(side)
        if ask <= 0 or bid <= 0:
            return None
        if worker.is_falling_knife(side, ask):
            return None
        size = worker.engine_order_size(side)
        if size <= 0:
            return None
        mode = cfg.momentum_execution_mode.lower()
        if mode == "single_maker":
            px = _tick_down(bid)
            if px <= 0 or px >= ask:
                return None
            return EngineDecision("momentum", side, "BUY", px, size, f"binance_delta={signal.price_delta:+.5%}", score)
        if mode == "dual_hybrid":
            contra = "NO" if side == "YES" else "YES"
            contra_bid = worker.best_bid(contra)
            contra_px = _tick_down(contra_bid)
            if contra_px <= 0:
                return None
            return EngineDecision(
                "momentum", side, "BUY", ask, size,
                f"binance_delta={signal.price_delta:+.5%}", score,
                contra, "BUY", contra_px,
            )
        px = ask
        action = "BUY"
        return EngineDecision("momentum", side, action, px, size, f"binance_delta={signal.price_delta:+.5%}", score)

    async def execute(self, worker: "MarketWorker", d: EngineDecision) -> bool:
        cfg = worker.worker_config
        mode = cfg.momentum_execution_mode.lower()
        order_type = "FOK" if mode == "single_taker" else "GTC"
        ok, oid, filled = await worker.place_order_raw(d.side, d.price, d.size, order_type=order_type, action="BUY")
        if not ok:
            return False
        if oid and oid != "dry-run" and order_type == "GTC":
            worker.track_engine_order(oid, d.side, d.size, d.price, "BUY", "momentum")
        fill_size = d.size if filled or oid == "dry-run" else 0.0
        if oid and oid != "dry-run" and order_type == "FOK" and not filled:
            fill_size, fill_price = await worker.poll_order_fill(oid, d.size, d.price)
        else:
            fill_price = d.price
        if fill_size > worker.min_fill_delta():
            worker.inventory.record_buy(d.side, fill_size, fill_price)
            worker.log_trade(d.side, fill_price, "BUY", size=fill_size)
        if d.second_side:
            ok2, oid2, filled2 = await worker.place_order_raw(
                d.second_side, d.second_price, d.size, order_type="GTC", action=d.second_action or "BUY"
            )
            if ok2 and oid2 and oid2 != "dry-run":
                worker.track_engine_order(oid2, d.second_side, d.size, d.second_price, d.second_action or "BUY", "momentum")
            if filled2 or oid2 == "dry-run":
                worker.inventory.record_buy(d.second_side, d.size, d.second_price)
        return fill_size > worker.min_fill_delta()


class SpreadCaptureEngine:
    """Risk-neutral UP+DOWN capture with executable-price confirmation."""

    async def evaluate(self, worker: "MarketWorker") -> Optional[EngineDecision]:
        cfg = worker.worker_config
        if not cfg.spread_capture_enabled or not worker.can_enter():
            return None
        yb, nb = worker.best_bid("YES"), worker.best_bid("NO")
        ya, na = worker.best_ask("YES"), worker.best_ask("NO")
        if min(yb, nb, ya, na) <= 0:
            return None
        # Public description uses combined bid as the trigger.  Execution must
        # still be checked against executable asks; otherwise the bot can see a
        # theoretical arb that disappears when it actually buys.
        bid_sum = yb + nb
        ask_sum = ya + na
        if bid_sum >= cfg.spread_capture_trigger_sum:
            return None
        if ask_sum > cfg.spread_capture_max_entry_sum + _EPS:
            return None
        size = worker.engine_order_size("YES")
        if size <= 0 or worker.engine_order_size("NO") <= 0:
            return None
        pos = economic_position(worker)
        if pos.unpaired > cfg.spread_capture_max_unpaired:
            return None
        return EngineDecision(
            "spread", "YES", "BUY", ya, size,
            f"bid_sum={bid_sum:.4f} ask_sum={ask_sum:.4f}",
            max(0.0, 1.0 - ask_sum), "NO", "BUY", na,
        )

    async def execute(self, worker: "MarketWorker", d: EngineDecision) -> bool:
        ok1, oid1, filled1 = await worker.place_order_raw(d.side, d.price, d.size, order_type="FOK", action="BUY")
        if not ok1:
            return False
        s1 = d.size if filled1 or oid1 == "dry-run" else 0.0
        p1 = d.price
        if oid1 and oid1 != "dry-run" and not filled1:
            s1, p1 = await worker.poll_order_fill(oid1, d.size, d.price)
        if s1 <= worker.min_fill_delta():
            return False
        worker.inventory.record_buy(d.side, s1, p1)
        worker.log_trade(d.side, p1, "BUY", size=s1)
        # Only submit the second leg after the first has actually filled. This
        # prevents an accidental one-sided "arb" from becoming new inventory.
        ok2, oid2, filled2 = await worker.place_order_raw(d.second_side, d.second_price, s1, order_type="FOK", action="BUY")
        if not ok2:
            return True
        s2 = s1 if filled2 or oid2 == "dry-run" else 0.0
        p2 = d.second_price
        if oid2 and oid2 != "dry-run" and not filled2:
            s2, p2 = await worker.poll_order_fill(oid2, s1, d.second_price)
        if s2 > worker.min_fill_delta():
            worker.inventory.record_buy(d.second_side, s2, p2)
            worker.log_trade(d.second_side, p2, "BUY", size=s2)
        return s2 > worker.min_fill_delta()


class MarketMakingEngine:
    """Two-sided resting quotes, guarded by Binance preemptive cancellation."""

    async def evaluate_and_quote(self, worker: "MarketWorker") -> bool:
        cfg = worker.worker_config
        if not cfg.market_making_enabled or not worker.can_enter():
            return False
        if BinancePreemptiveCancel.exposed_side(worker):
            return False
        changed = await worker.refresh_market_maker_quotes()
        return changed


class ArbigabEngineRouter:
    def __init__(self) -> None:
        self.momentum = MomentumEngine()
        self.spread = SpreadCaptureEngine()
        self.market_making = MarketMakingEngine()
        self.cancel = BinancePreemptiveCancel()

    async def on_tick(self, worker: "MarketWorker") -> bool:
        worker.dashboard["engine"] = "risk"
        # Cancellation is first and unconditional: it protects every engine.
        await self.cancel.run(worker)
        if worker.order_state.name == "PENDING" or worker._order_lock.locked():
            return False
        # Existing inventory-risk controller remains the highest-priority
        # economic action. It may hedge or boundedly unwind, but never chases a
        # falling knife just to make share counts look pretty.
        if await worker._manage_gabagool_inventory():
            return True
        # A locked-in pair should not be used as justification to start another
        # directional cycle in the same market.
        pos = economic_position(worker)
        if pos.locked_profit >= worker.worker_config.gabagool_min_profit_margin * max(pos.matched_pairs, 1.0):
            if pos.unpaired <= worker.worker_config.gabagool_inventory_soft_limit:
                return await self.market_making.evaluate_and_quote(worker)

        spread = await self.spread.evaluate(worker)
        if spread:
            worker.dashboard["engine"] = "spread_capture"
            async with worker._order_lock:
                return await self.spread.execute(worker, spread)
        momentum = await self.momentum.evaluate(worker)
        if momentum:
            worker.dashboard["engine"] = "momentum"
            async with worker._order_lock:
                return await self.momentum.execute(worker, momentum)
        worker.dashboard["engine"] = "market_making" if worker.worker_config.market_making_enabled else "idle"
        return await self.market_making.evaluate_and_quote(worker)
