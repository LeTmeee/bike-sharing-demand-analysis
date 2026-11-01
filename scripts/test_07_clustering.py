"""Focused checks for daily profiles, exclusions and reproducibility."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


def load_module():
    path = Path(__file__).with_name("07_clustering.py")
    assert path.is_file(), "Daily clustering module has not been implemented"
    spec = importlib.util.spec_from_file_location("clustering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        self.c = load_module()

    def sample(self, days=3):
        times = pd.date_range("2011-01-01", periods=24*days, freq="h")
        return pd.DataFrame(dict(datetime=times, count=np.tile(np.arange(1, 25), days),
                                 season=1, workingday=0, holiday=0, weather=1,
                                 temp=10, atemp=12, humidity=50, windspeed=0))

    def test_missing_hour_and_entire_date_are_not_zero_filled(self):
        data = self.sample()
        data = data.drop(index=[5, *range(24, 48)])
        profiles, audit = self.c.build_daily(data)
        self.assertEqual(profiles.shape, (3, 24))
        self.assertTrue(np.isnan(profiles.iloc[0, 5]))
        self.assertTrue(profiles.iloc[1].isna().all())
        self.assertEqual(audit.observed_hours.tolist(), [23, 0, 24])
        self.assertEqual(self.c.select_profiles(profiles, min_days=1).index.tolist(), [profiles.index[2]])

    def test_rederive_time_and_ignore_other_target_columns(self):
        data = self.sample()
        data["hour"] = 99
        data["date"] = "bad"
        data["casual"] = np.inf
        profiles, _ = self.c.build_daily(data)
        np.testing.assert_array_equal(profiles.iloc[0], np.arange(1, 25))

    def test_weather_tie_and_daily_metadata(self):
        data = self.sample(1)
        data.loc[:11, "weather"] = 2
        _, audit = self.c.build_daily(data)
        self.assertTrue(pd.isna(audit.weather.iloc[0]))
        self.assertEqual(audit.weather_modal_share.iloc[0], .5)
        self.assertEqual(audit.weekday.iloc[0], 5)

    def test_invalid_inputs_are_rejected(self):
        for field, value in [("datetime", "2011-01-01 00:30:00"),
                             ("datetime", "2011-01-01 00:00:00"),
                             ("count", -1), ("count", np.inf), ("count", .5),
                             ("workingday", 1), ("weather", 8), ("humidity", 101)]:
            data = self.sample()
            if field == "count":
                data[field] = data[field].astype(float)
            data.loc[1, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.c.build_daily(data)

    def test_zero_total_exclusion_and_coverage_guard(self):
        data = self.sample(2)
        data.loc[:23, "count"] = 0
        profiles, _ = self.c.build_daily(data)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            self.c.select_profiles(profiles)
        selected = self.c.select_profiles(profiles, min_days=1)
        self.assertEqual(len(selected), 1)

    def test_share_profiles_remove_scale(self):
        profiles, _ = self.c.build_daily(self.sample(2))
        profiles.iloc[1] *= 10
        shares = self.c.representations(profiles)["share"]
        np.testing.assert_allclose(shares.iloc[0], shares.iloc[1])
        np.testing.assert_allclose(shares.sum(axis=1), 1)

    def test_reproducible_selection_and_degenerate_candidates(self):
        rng = np.random.default_rng(4)
        x = np.vstack([rng.normal(0, .01, (8, 24)), rng.normal(1, .01, (8, 24))])
        metrics, models, best = self.c.evaluate(x, max_k=4, seed=18, n_init=20)
        again, _, best_again = self.c.evaluate(x, max_k=4, seed=18, n_init=20)
        pd.testing.assert_frame_equal(metrics, again)
        self.assertEqual(best, 2)
        self.assertEqual(best, best_again)
        self.assertEqual(models[best].n_init, 20)
        repeated = np.repeat([[0., 0.], [1., 1.]], 5, axis=0)
        metrics, _, best = self.c.evaluate(repeated, max_k=6, seed=18, n_init=20)
        self.assertEqual(metrics.k.tolist(), [1, 2])
        self.assertEqual(best, 2)
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.c.evaluate(np.ones((5, 24)), max_k=6, seed=18, n_init=20)

    def test_insufficient_run_saves_audit_without_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.sample(2).to_csv(root / "input.csv", index=False)
            args = SimpleNamespace(input=root / "input.csv", tables_dir=root / "tables",
                                   figures_dir=root / "figures", max_k=6, seed=18, n_init=20)
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                self.c.run(args)
            self.assertTrue((args.tables_dir / "07_day_audit.csv").is_file())
            self.assertFalse((args.tables_dir / "07_assignments.csv").exists())

    def test_full_workflow_outputs_match_profiles_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.sample(30)
            # Two separated, non-identical daily shapes without using metadata.
            data["count"] = [int(5 + h + (80 if (d % 2 == 0 and h == 8) or
                                                (d % 2 == 1 and h == 13) else 0) + d % 3)
                             for d in range(30) for h in range(24)]
            data.to_csv(root / "input.csv", index=False)
            args = SimpleNamespace(input=root / "input.csv", tables_dir=root / "tables",
                                   figures_dir=root / "figures", max_k=3, seed=18, n_init=20)
            self.c.run(args)
            labels = pd.read_csv(args.tables_dir / "07_assignments.csv")
            self.assertEqual(labels.groupby("representation").size().to_dict(), {"raw": 30, "share": 30})
            distributions = pd.read_csv(args.tables_dir / "07_metadata_distribution.csv")
            np.testing.assert_allclose(distributions.groupby(["representation", "cluster", "field"]).within_cluster_share.sum(), 1)
            centers = pd.read_csv(args.tables_dir / "07_centroids.csv")
            for name in ["raw", "share"]:
                features = pd.read_csv(args.tables_dir / f"07_{name}_features.csv", index_col="date")
                assigned = labels.loc[labels.representation.eq(name)].set_index("date")
                for _, center in centers.loc[centers.representation.eq(name)].iterrows():
                    mean = features.loc[assigned.cluster.eq(center.cluster)].mean()
                    np.testing.assert_allclose(mean.to_numpy(), center[self.c.HOURS].to_numpy(dtype=float))
            metadata = json.loads((args.tables_dir / "07_run_metadata.json").read_text())
            self.assertEqual(metadata["included_days"], 30)
            self.assertEqual(metadata["feature_columns"], self.c.HOURS)
            self.assertEqual(len(list(args.figures_dir.glob("07_*.png"))), 7)


if __name__ == "__main__":
    unittest.main()
