import unittest

from utils.inventory_cost import evaluate_cheap_entry, PurchaseDecision


class ConfigurableCheapEntryTests(unittest.TestCase):
    def test_threshold_is_configurable_not_hardcoded_99(self):
        ev = evaluate_cheap_entry(0.90, cheap_entry_price=0.90, max_entry_price=0.90)
        self.assertEqual(ev.decision, PurchaseDecision.BUY)

    def test_polymarket_tick_drives_strategy(self):
        from pathlib import Path
        source = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("await self.strategy.on_market_tick(self)", source)


if __name__ == "__main__":
    unittest.main()
