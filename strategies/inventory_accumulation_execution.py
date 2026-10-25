"""FOK execution helpers for inventory accumulation."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional, Tuple

from utils.clob_helpers import clamp_buy_price
from utils.momentum_risk import is_order_fully_filled

if TYPE_CHECKING:
    from bot import MarketWorker


async def _simulate_fok_fill(worker: "MarketWorker", side: str, limit: float, size: float):
    from bot import MIN_FILL_DELTA

    cfg = worker.worker_config
    timeout_sec = max(0.0, cfg.fill_timeout_ms / 1000.0)
    poll_sec = max(0.05, cfg.fill_poll_ms / 1000.0)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() <= deadline:
        exe = worker.executable_buy(side, size)
        if exe and exe[0] <= limit + 1e-9 and exe[1] >= size - MIN_FILL_DELTA:
            return exe[1], exe[0]
        await asyncio.sleep(poll_sec)
    return 0.0, 0.0


async def fok_buy(
    worker: "MarketWorker",
    side: str,
    limit_price: float,
    size: float,
    *,
    purpose: str,
) -> Tuple[float, float, Optional[str]]:
    from bot import MIN_FILL_DELTA

    if worker.is_dry_run():
        qty, px = await _simulate_fok_fill(worker, side, limit_price, size)
        return qty, px, "dry-run" if qty > MIN_FILL_DELTA else None

    placed = await worker.place_accumulation_fok(side, limit_price, size, purpose=purpose)
    order_id, immediate = placed[0], placed[1]
    if not order_id:
        return 0.0, 0.0, None
    if immediate and immediate > MIN_FILL_DELTA:
        worker._untrack_order(order_id)
        fill_px = await worker.resolve_fill_price(order_id, size, limit_price)
        return float(immediate), fill_px, order_id

    cfg = worker.worker_config
    timeout_sec = cfg.fill_timeout_ms / 1000.0
    poll_sec = cfg.fill_poll_ms / 1000.0
    started = time.monotonic()
    cumulative = 0.0
    fill_price = limit_price
    while time.monotonic() - started < timeout_sec:
        cumulative, fill_price = await worker.poll_order_fill(order_id, size, limit_price)
        if is_order_fully_filled(size, cumulative, MIN_FILL_DELTA):
            worker._untrack_order(order_id)
            return cumulative, fill_price, order_id
        await asyncio.sleep(poll_sec)
    try:
        cumulative, fill_price = await worker.cancel_order_confirmed(order_id, size, limit_price)
    except TimeoutError:
        return cumulative, fill_price, order_id
    worker._untrack_order(order_id)
    return cumulative, fill_price, order_id


def buy_limit_for_executable(
    worker: "MarketWorker",
    side: str,
    size: float,
    *,
    max_price_cap: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Return (limit_price, executable_avg) or None."""
    cfg = worker.worker_config
    exe = worker.executable_buy(side, size)
    if not exe:
        return None
    avg, filled = exe
    slip = cfg.max_slippage
    cap = avg + slip
    if max_price_cap is not None:
        cap = min(cap, max_price_cap + slip)
    if max_price_cap is not None and avg > max_price_cap + 1e-9:
        return None
    return clamp_buy_price(cap), avg


async def fok_buy_with_retries(
    worker: "MarketWorker",
    side: str,
    size: float,
    *,
    purpose: str,
    max_price_cap: Optional[float] = None,
    max_retries: int,
    retry_delay_ms: int,
) -> Tuple[float, float, Optional[str]]:
    from bot import MIN_FILL_DELTA

    attempts = 0
    while attempts <= max_retries:
        worker.refresh_order_book(side)
        limits = buy_limit_for_executable(worker, side, size, max_price_cap=max_price_cap)
        if not limits:
            return 0.0, 0.0, None
        limit, _exe = limits
        qty, px, oid = await fok_buy(worker, side, limit, size, purpose=purpose)
        if qty > MIN_FILL_DELTA and px > 0:
            return qty, px, oid
        attempts += 1
        if attempts > max_retries:
            break
        await asyncio.sleep(max(0.05, retry_delay_ms / 1000.0))
    return 0.0, 0.0, None


async def fok_paired_buy(
    worker: "MarketWorker",
    qty: float,
    *,
    purpose: str,
    max_price_cap_yes: Optional[float] = None,
    max_price_cap_no: Optional[float] = None,
    max_retries: int,
    retry_delay_ms: int,
) -> Tuple[float, float, float, float, Optional[str], Optional[str]]:
    """Submit YES then NO FOK legs (not atomic). Returns (yes_qty, yes_px, no_qty, no_px, yes_oid, no_oid)."""
    from bot import MIN_FILL_DELTA

    yes_qty, yes_px, yes_oid = await fok_buy_with_retries(
        worker,
        "YES",
        qty,
        purpose="paired_dca_yes",
        max_price_cap=max_price_cap_yes,
        max_retries=max_retries,
        retry_delay_ms=retry_delay_ms,
    )
    if yes_qty <= MIN_FILL_DELTA:
        return 0.0, 0.0, 0.0, 0.0, None, None

    no_qty, no_px, no_oid = await fok_buy_with_retries(
        worker,
        "NO",
        yes_qty,
        purpose="paired_dca_no",
        max_price_cap=max_price_cap_no,
        max_retries=max_retries,
        retry_delay_ms=retry_delay_ms,
    )
    if no_qty <= MIN_FILL_DELTA:
        print(
            f"⚠️ [PAIRED RECOVERY] {worker.asset_type.upper()} {worker.window_slug} | "
            f"YES leg filled x{yes_qty:.2f} @ {yes_px:.4f} but NO FOK failed — "
            f"pausing paired DCA until inventory rebalanced"
        )
        return yes_qty, yes_px, 0.0, 0.0, yes_oid, None
    return yes_qty, yes_px, no_qty, no_px, yes_oid, no_oid
