"""Sequential 10–20¢ / 98¢ combined pair strategy tests."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from utils.inventory_cost import InventorySnapshot
from utils.inventory_strategy_store import AccumulationPhase, InventoryStrategyStore
from utils.sequential_pair_strategy import (
    SequentialDecision,
    entry_cutoff_active,
    evaluate_opposite_with_entry_state,
    gross_combined_pair_cost,
    max_opposite_gross_price,
    pick_first_side,
    side_already_entered,
    mark_side_filled,
)
from strategies.inventory_accumulation_strategy import InventoryAccumulationStrategy


class SequentialLogicTests(unittest.TestCase):
    def test_yes15_no78_allows_opposite(self):
        meta = {"entries": {"YES": {"filled": True, "avg_price": 0.15}}}
        snap = InventorySnapshot(yes_shares=5, no_shares=0, yes_cost=0.75, no_cost=0)
        ev = evaluate_opposite_with_entry_state(
            snap,
            meta,
            first_leg_side="YES",
            opposite_side="NO",
            opposite_qty=5,
            opposite_executable=0.78,
            target_combined_pair_cost=0.98,
            max_combined_pair_cost=0.98,
            min_net_profit_per_pair=0.0,
            slippage_reserve=0.0,
        )
        self.assertEqual(ev.decision, SequentialDecision.BUY)
        self.assertAlmostEqual(gross_combined_pair_cost(0.15, 0.78), 0.93)

    def test_yes15_no85_waits(self):
        meta = {"entries": {"YES": {"filled": True, "avg_price": 0.15}}}
        snap = InventorySnapshot(yes_shares=5, no_shares=0, yes_cost=0.75, no_cost=0)
        ev = evaluate_opposite_with_entry_state(
            snap,
            meta,
            first_leg_side="YES",
            opposite_side="NO",
            opposite_qty=5,
            opposite_executable=0.85,
            target_combined_pair_cost=0.98,
            max_combined_pair_cost=0.98,
            min_net_profit_per_pair=0.0,
            slippage_reserve=0.0,
        )
        self.assertEqual(ev.decision, SequentialDecision.WAIT)
        self.assertEqual(ev.reason, "WAITING_FOR_OPPOSITE_PRICE")

    def test_yes15_no83_gross_98(self):
        self.assertAlmostEqual(gross_combined_pair_cost(0.15, 0.83), 0.98)
        self.assertAlmostEqual(max_opposite_gross_price(0.15, 0.98), 0.83)

    def test_no12_yes80_opposite_logic(self):
        meta = {"entries": {"NO": {"filled": True, "avg_price": 0.12}}}
        snap = InventorySnapshot(yes_shares=0, no_shares=5, yes_cost=0, no_cost=0.60)
        ev = evaluate_opposite_with_entry_state(
            snap,
            meta,
            first_leg_side="NO",
            opposite_side="YES",
            opposite_qty=5,
            opposite_executable=0.80,
            target_combined_pair_cost=0.98,
            max_combined_pair_cost=0.98,
            min_net_profit_per_pair=0.0,
            slippage_reserve=0.0,
        )
        self.assertEqual(ev.decision, SequentialDecision.BUY)

    def test_both_in_band_picks_cheaper_yes_tie(self):
        sel = pick_first_side((0.14, 100.0), (0.14, 100.0), min_price=0.10, max_price=0.20)
        self.assertEqual(sel.side, "YES")
        self.assertEqual(sel.decision, SequentialDecision.BUY)

    def test_both_in_band_picks_lower_no(self):
        sel = pick_first_side((0.18, 100.0), (0.11, 100.0), min_price=0.10, max_price=0.20)
        self.assertEqual(sel.side, "NO")

    def test_neither_in_band_skips(self):
        sel = pick_first_side((0.25, 100.0), (0.30, 100.0), min_price=0.10, max_price=0.20)
        self.assertEqual(sel.reason, "SKIP_NO_SIDE_IN_10_TO_20_CENT_RANGE")

    def test_one_entry_per_side_metadata(self):
        meta = {}
        mark_side_filled(meta, "YES", fill_price=0.15, qty=5, order_id="a")
        self.assertTrue(side_already_entered(meta, "YES"))
        self.assertFalse(side_already_entered(meta, "NO"))

    def test_entry_cutoff_at_120_seconds(self):
        self.assertTrue(entry_cutoff_active(120, 2))
        self.assertFalse(entry_cutoff_active(120.1, 2))

    def test_no_buys_below_120_seconds(self):
        self.assertTrue(entry_cutoff_active(119.9, 2))
        self.assertTrue(entry_cutoff_active(0, 2))

    def test_fees_can_reject_at_83c_gross(self):
        meta = {"entries": {"YES": {"filled": True, "avg_price": 0.15}}}
        snap = InventorySnapshot(yes_shares=5, no_shares=0, yes_cost=0.75, no_cost=0, fee_rate=0.05)
        ev = evaluate_opposite_with_entry_state(
            snap,
            meta,
            first_leg_side="YES",
            opposite_side="NO",
            opposite_qty=5,
            opposite_executable=0.83,
            target_combined_pair_cost=0.98,
            max_combined_pair_cost=0.98,
            min_net_profit_per_pair=0.02,
            slippage_reserve=0.005,
        )
        self.assertNotEqual(ev.decision, SequentialDecision.BUY)


class StrategyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cutoff_blocks_before_fok_submit(self):
        strat = InventoryAccumulationStrategy()
        worker = MagicMock()
        worker.worker_config.no_buys_last_minutes = 2
        worker.market_seconds_left.return_value = 119
        worker.dashboard = {}
        worker.is_dry_run.return_value = True
        worker.best_ask.return_value = 0.15
        worker.canonical_market_id.return_value = "cid-1"
        store = InventoryStrategyStore(":memory:")
        cycle = store.get_or_create_cycle("btc:5m", "cid-1")
        self.assertTrue(strat._entry_cutoff(worker, cycle))
        with patch(
            "strategies.inventory_accumulation_execution.fok_buy_with_retries",
            new_callable=AsyncMock,
        ) as mock_fok:
            await strat._try_first_entry(worker, cycle, "cid-1")
            mock_fok.assert_not_called()


if __name__ == "__main__":
    unittest.main()
