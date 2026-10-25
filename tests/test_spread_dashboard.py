import unittest

from config import WORKER_CONFIGS


class DashboardConfigTests(unittest.TestCase):
    def test_worker_config_exposes_inventory_accumulation_settings(self):
        wc = WORKER_CONFIGS[0]
        self.assertTrue(wc.inventory_accumulation_enabled)
        self.assertLessEqual(wc.cheap_entry_price, wc.max_entry_price)
        self.assertEqual(wc.order_type, "FOK")


if __name__ == "__main__":
    unittest.main()
