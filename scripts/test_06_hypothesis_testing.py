"""Focused checks of statistical units, exclusions and multiplicity."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ht", Path(__file__).with_name("06_hypothesis_testing.py"))
ht = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ht)


class DailyTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame(dict(datetime=pd.date_range("2011-01-01", periods=4, freq="h"),
                                 count=[1, 3, 5, 7], workingday=[0]*4,
                                 weather=[1, 1, 2, 2], temp=[10, 12, 14, 16]))

    def test_daily_unit_and_tied_weather(self):
        d = ht.aggregate(self.sample())
        self.assertEqual(len(d), 1)
        self.assertEqual(d.observed_hours.iloc[0], 4)
        self.assertEqual(d.demand.iloc[0], 4)
        self.assertEqual(d.observed_total.iloc[0], 16)
        self.assertTrue(pd.isna(d.weather.iloc[0]))
        self.assertEqual(d.temp.iloc[0], 13)

    def test_bad_timestamps_and_calendar_flags(self):
        for field, value in [("datetime", "2011-01-01"), ("workingday", 1), ("count", -1)]:
            d = self.sample()
            d.loc[1, field] = value
            with self.assertRaises(ValueError):
                ht.aggregate(d)

    def test_holm_and_delta_direction(self):
        np.testing.assert_allclose(ht.holm([.04, .001, .02]), [.04, .003, .04])
        self.assertEqual(ht.delta(np.array([4, 5]), np.array([1, 2])), 1)
        self.assertEqual(ht.delta(np.array([1, 2]), np.array([4, 5])), -1)


if __name__ == "__main__":
    unittest.main()
