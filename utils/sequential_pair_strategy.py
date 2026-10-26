"""Sequential cheap-side-first pair strategy (10–20¢ leg, then opposite to 98¢ combined)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from utils.inventory_cost import InventorySnapshot, PurchaseDecision, matched_combined_average
from utils.profitability_gate import (
    check_executable_depth,
    check_pair_economics,
    projected_net_profit_per_pair,
)

_EPS = 1e-9


class SequentialDecision(str, Enum):
    BUY = "BUY"
    SKIP = "SKIP"
    WAIT = "WAIT"


@dataclass(frozen=True)
class SequentialEvaluation:
    decision: SequentialDecision
    reason: str
    side: Optional[str] = None
    executable_price: float = 0.0
    max_opposite_price: Optional[float] = None
    projected_gross_combined: Optional[float] = None
    projected_net_profit: Optional[float] = None

    def log_fields(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "side": self.side,
            "executable_price": self.executable_price,
            "max_opposite_price": self.max_opposite_price,
            "projected_gross_combined": self.projected_gross_combined,
            "projected_net_profit": self.projected_net_profit,
        }


def entry_cutoff_active(seconds_left: float, no_buys_last_minutes: int) -> bool:
    try:
        left = float(seconds_left)
        mins = int(no_buys_last_minutes)
    except (TypeError, ValueError):
        return True
    if mins < 0 or not math.isfinite(left):
        return True
    return left <= mins * 60.0 + _EPS


def in_cheap_side_band(price: float, *, min_price: float, max_price: float) -> bool:
    px = float(price)
    if px <= 0 or not math.isfinite(px):
        return False
    return float(min_price) - _EPS <= px <= float(max_price) + _EPS


def pick_first_side(
    yes_exe: Optional[Tuple[float, float]],
    no_exe: Optional[Tuple[float, float]],
    *,
    min_price: float,
    max_price: float,
) -> SequentialEvaluation:
    """Choose one side in [min_price, max_price]; lower executable ask wins; YES on tie."""
    candidates: list[tuple[str, float, float]] = []
    for side, exe in (("YES", yes_exe), ("NO", no_exe)):
        if not exe:
            continue
        avg, liq = float(exe[0]), float(exe[1])
        if in_cheap_side_band(avg, min_price=min_price, max_price=max_price):
            candidates.append((side, avg, liq))
    if not candidates:
        return SequentialEvaluation(
            SequentialDecision.SKIP,
            "SKIP_NO_SIDE_IN_10_TO_20_CENT_RANGE",
        )
    candidates.sort(key=lambda row: (row[1], -row[2], 0 if row[0] == "YES" else 1))
    side, avg, _ = candidates[0]
    return SequentialEvaluation(
        SequentialDecision.BUY,
        "first_side_selected",
        side=side,
        executable_price=avg,
    )


def gross_combined_pair_cost(first_leg_avg: float, opposite_executable: float) -> float:
    return float(first_leg_avg) + float(opposite_executable)


def max_opposite_gross_price(first_leg_avg: float, target_combined_pair_cost: float) -> float:
    return float(target_combined_pair_cost) - float(first_leg_avg)


def side_entry_record(metadata: Dict[str, Any], side: str) -> Optional[dict]:
    entries = metadata.get("entries") or {}
    row = entries.get(side.upper())
    return row if isinstance(row, dict) else None


def side_already_entered(metadata: Dict[str, Any], side: str) -> bool:
    row = side_entry_record(metadata, side)
    return bool(row and row.get("filled"))


def mark_side_filled(metadata: Dict[str, Any], side: str, *, fill_price: float, qty: float, order_id: str) -> None:
    entries = dict(metadata.get("entries") or {})
    entries[side.upper()] = {
        "filled": True,
        "avg_price": float(fill_price),
        "qty": float(qty),
        "order_id": order_id or "",
    }
    metadata["entries"] = entries


def evaluate_opposite_entry(
    snap: InventorySnapshot,
    *,
    first_leg_side: str,
    first_leg_avg: float,
    opposite_side: str,
    opposite_qty: float,
    opposite_executable: float,
    target_combined_pair_cost: float,
    max_combined_pair_cost: float,
    min_net_profit_per_pair: float,
    slippage_reserve: float,
) -> SequentialEvaluation:
    if opposite_executable <= 0 or opposite_qty <= _EPS:
        return SequentialEvaluation(SequentialDecision.SKIP, "REJECT_INVALID_QTY_OR_PRICE", side=opposite_side)
    max_px = max_opposite_gross_price(first_leg_avg, target_combined_pair_cost)
    gross = gross_combined_pair_cost(first_leg_avg, opposite_executable)
    if opposite_executable > max_px + _EPS:
        return SequentialEvaluation(
            SequentialDecision.WAIT,
            "WAITING_FOR_OPPOSITE_PRICE",
            side=opposite_side,
            executable_price=opposite_executable,
            max_opposite_price=max_px,
            projected_gross_combined=gross,
        )
    after = snap.with_buy(opposite_side, opposite_qty, opposite_executable)
    resulting = matched_combined_average(after)
    if resulting is None:
        return SequentialEvaluation(SequentialDecision.SKIP, "REJECT_NO_MATCHED_PAIR", side=opposite_side)
    econ = check_pair_economics(
        resulting,
        max_combined_cost=max_combined_pair_cost,
        min_net_profit_per_pair=min_net_profit_per_pair,
        slippage_reserve=slippage_reserve,
    )
    if not econ.allowed:
        if gross > max_combined_pair_cost + _EPS:
            return SequentialEvaluation(
                SequentialDecision.WAIT,
                "REJECT_COMBINED_COST_TOO_HIGH",
                side=opposite_side,
                executable_price=opposite_executable,
                max_opposite_price=max_px,
                projected_gross_combined=gross,
                projected_net_profit=econ.projected_net_profit,
            )
        return SequentialEvaluation(
            SequentialDecision.SKIP,
            econ.reason,
            side=opposite_side,
            executable_price=opposite_executable,
            max_opposite_price=max_px,
            projected_gross_combined=gross,
            projected_net_profit=econ.projected_net_profit,
        )
    net_p = projected_net_profit_per_pair(resulting, slippage_reserve=slippage_reserve)
    return SequentialEvaluation(
        SequentialDecision.BUY,
        "opposite_entry_allowed",
        side=opposite_side,
        executable_price=opposite_executable,
        max_opposite_price=max_px,
        projected_gross_combined=gross,
        projected_net_profit=net_p,
    )


def evaluate_opposite_with_entry_state(
    snap: InventorySnapshot,
    metadata: Dict[str, Any],
    *,
    first_leg_side: str,
    opposite_side: str,
    opposite_qty: float,
    opposite_executable: float,
    target_combined_pair_cost: float,
    max_combined_pair_cost: float,
    min_net_profit_per_pair: float,
    slippage_reserve: float,
) -> SequentialEvaluation:
    if side_already_entered(metadata, opposite_side):
        return SequentialEvaluation(
            SequentialDecision.SKIP,
            "REJECT_OPPOSITE_SIDE_ALREADY_ENTERED",
            side=opposite_side,
        )
    first_row = side_entry_record(metadata, first_leg_side)
    first_avg = float(first_row["avg_price"]) if first_row and first_row.get("avg_price") else (
        snap.yes_avg() if first_leg_side == "YES" else snap.no_avg()
    )
    if first_avg <= _EPS:
        return SequentialEvaluation(SequentialDecision.SKIP, "missing_first_leg_fill_price")
    return evaluate_opposite_entry(
        snap,
        first_leg_side=first_leg_side,
        first_leg_avg=first_avg,
        opposite_side=opposite_side,
        opposite_qty=opposite_qty,
        opposite_executable=opposite_executable,
        target_combined_pair_cost=target_combined_pair_cost,
        max_combined_pair_cost=max_combined_pair_cost,
        min_net_profit_per_pair=min_net_profit_per_pair,
        slippage_reserve=slippage_reserve,
    )
