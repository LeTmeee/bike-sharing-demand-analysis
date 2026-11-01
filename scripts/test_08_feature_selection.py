"""Guard against leakage and incorrect chronological model evaluation."""
import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import pandas as pd


def load_module():
    path = Path(__file__).with_name("08_feature_selection.py")
    spec = importlib.util.spec_from_file_location("feature_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        self.m = load_module()
        t = pd.date_range("2011-01-01", periods=40*24, freq="h")
        self.data = pd.DataFrame(dict(datetime=t, count=np.arange(len(t)) % 100,
                                     season=1, holiday=0, workingday=0, weather=1,
                                     temp=np.arange(len(t)) / 100, atemp=10,
                                     humidity=50, windspeed=0))

    def test_allowlist_and_calendar_rederived(self):
        self.data["hour"] = 999
        for col in ["casual", "registered", "registered_share", "log1p_count", "flag_iqr_count"]:
            self.data[col] = self.data["count"]
        x, y, t = self.m.prepare(self.data.sample(frac=1, random_state=18))
        self.assertEqual(list(x), self.m.NUMERIC + self.m.CATEGORICAL)
        np.testing.assert_array_equal(x.hour, t.dt.hour)
        np.testing.assert_array_equal(y, self.data["count"])

    def test_date_partitions_are_ordered_and_disjoint(self):
        x, y, t = self.m.prepare(self.data)
        train, test, folds = self.m.partitions(t, .2, 5)
        self.assertLess(t.iloc[train].max(), t.iloc[test].min())
        self.assertEqual(len(train) + len(test), len(x))
        for a, b in folds:
            self.assertLess(t.iloc[train[a]].max(), t.iloc[train[b]].min())
            self.assertFalse(set(t.iloc[train[a]].dt.date) & set(t.iloc[train[b]].dt.date))

    def test_preprocessing_uses_training_only_and_handles_new_category(self):
        x, _, _ = self.m.prepare(self.data)
        prep = self.m.preprocessing()
        prep.fit(x.iloc[:24])
        self.assertAlmostEqual(prep.named_transformers_["numeric"].mean_[0], x.temp.iloc[:24].mean())
        future = x.iloc[24:25].copy()
        future["weather"] = 4
        with self.assertWarnsRegex(UserWarning, "Found unknown categories"):
            self.assertTrue(np.isfinite(prep.transform(future)).all())

    def test_bad_input_rejected(self):
        for field, value in [("datetime", None), ("datetime", "2011-01-01 00:00:00"),
                             ("count", -1), ("count", .5), ("humidity", 101),
                             ("weather", 8), ("temp", np.inf)]:
            data = self.data.copy()
            if field == "count":
                data[field] = data[field].astype(float)
            data.loc[1, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.m.prepare(data)
        with self.assertRaises(ValueError):
            self.m.partitions(pd.Series(pd.date_range("2011-01-01", periods=3)), .2, 5)

    def test_final_test_targets_do_not_change_selection(self):
        """End-to-end: a changed test outcome must not affect tuning/importance."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, t = self.m.prepare(self.data)
            _, heldout, _ = self.m.partitions(t, .2, 3)
            for name in ["original", "altered"]:
                data = self.data.copy()
                if name == "altered":
                    data.loc[heldout, "count"] += 10000
                source = root / f"{name}.csv"
                data.to_csv(source, index=False)
                args = SimpleNamespace(input=source, tables_dir=root/name/"tables",
                                       figures_dir=root/name/"figures", reports_dir=root/name/"reports",
                                       seed=18, test_fraction=.2, cv_splits=3, trees=4, repeats=2)
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
                    self.m.run(args)
            for table in ["comparison", "cv_results", "lasso_coefficients", "permutation_repeats"]:
                pd.testing.assert_frame_equal(pd.read_csv(root/"original"/"tables"/f"08_{table}.csv"),
                                              pd.read_csv(root/"altered"/"tables"/f"08_{table}.csv"))
            original = pd.read_csv(root/"original"/"tables"/"08_model_metrics.csv")
            altered = pd.read_csv(root/"altered"/"tables"/"08_model_metrics.csv")
            self.assertGreater(altered.rmse.iloc[0], original.rmse.iloc[0])


if __name__ == "__main__":
    unittest.main()
