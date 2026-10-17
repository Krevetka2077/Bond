import unittest
from io import BytesIO

import numpy as np
import pandas as pd

from portfolio import optimize, portfolio_metrics, read_prices, validate_allocations


class PortfolioTests(unittest.TestCase):
    def test_example_prices(self):
        prices = read_prices("data/example_prices.csv")
        self.assertEqual(prices.shape, (504, 7))
        self.assertTrue((prices > 0).all().all())

    def test_rejects_missing_price(self):
        rows = ["Date,A,B"]
        rows += [f"2024-01-{i:02d},{100+i},{200+i}" for i in range(1, 31)]
        rows.append("2024-01-31,,231")
        with self.assertRaisesRegex(ValueError, "пропусков"):
            read_prices(BytesIO("\n".join(rows).encode()))

    def test_rejects_infeasible_bounds(self):
        table = pd.DataFrame({"Актив": ["A", "B"], "Текущая доля, %": [50, 50],
                              "Мин., %": [60, 60], "Макс., %": [100, 100]})
        with self.assertRaisesRegex(ValueError, "несовместимы"):
            validate_allocations(table, ["A", "B"])

    def test_tail_risk_uses_period_losses(self):
        returns = pd.DataFrame({"A": [-0.4] + [0.01] * 19,
                                "B": [0.0] * 20})
        metrics = portfolio_metrics(returns, np.array([1.0, 0.0]), 252)
        self.assertAlmostEqual(metrics["CVaR 95%, за период"], 0.4)
        self.assertAlmostEqual(metrics["Макс. просадка"], 0.4)

    def test_optimization_respects_asset_caps(self):
        prices = read_prices("data/example_prices.csv")
        returns = prices.pct_change().iloc[1:]
        lower = np.full(returns.shape[1], 0.05)
        upper = np.full(returns.shape[1], 0.35)
        for weights in optimize(returns, lower, upper, 0.0, 252).values():
            self.assertIsInstance(weights, np.ndarray)
            self.assertAlmostEqual(weights.sum(), 1, places=3)
            self.assertTrue((weights >= lower - 1e-3).all())
            self.assertTrue((weights <= upper + 1e-3).all())


if __name__ == "__main__":
    unittest.main()
