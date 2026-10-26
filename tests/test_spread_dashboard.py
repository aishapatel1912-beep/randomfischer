import unittest

from config import WORKER_CONFIGS


class DashboardConfigTests(unittest.TestCase):
    def test_worker_config_exposes_inventory_accumulation_settings(self):
        wc = WORKER_CONFIGS[0]
        self.assertTrue(wc.inventory_accumulation_enabled)
        self.assertLessEqual(wc.cheap_side_min_price, wc.cheap_side_max_price)
        self.assertAlmostEqual(wc.target_combined_pair_cost, 0.98)
        self.assertAlmostEqual(wc.max_combined_pair_cost, 0.98)
        self.assertEqual(wc.order_type, "FOK")


if __name__ == "__main__":
    unittest.main()
