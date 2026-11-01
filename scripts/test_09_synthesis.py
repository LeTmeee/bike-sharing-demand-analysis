"""Integration checks for final synthesis provenance and report coverage."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('synthesis', ROOT / 'scripts/09_synthesis.py')
synthesis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthesis)


class SynthesisTests(unittest.TestCase):
    def copy_sources(self, target):
        for prefix in ['06_', '07_', '08_']:
            for source in (ROOT / 'outputs/tables').glob(prefix + '*'):
                shutil.copy2(source, target / source.name)

    def test_missing_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, '06'):
                synthesis.load_upstream(ROOT / 'data/processed/train_eda.csv', Path(temp), pd)

    def test_changed_input_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            self.copy_sources(target)
            path = target / '07_run_metadata.json'
            meta = json.loads(path.read_text(encoding='utf-8'))
            meta['input_sha256'] = 'different-input'
            path.write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '07.*SHA256'):
                synthesis.load_upstream(ROOT / 'data/processed/train_eda.csv', target, pd)

    def test_missing_table_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            self.copy_sources(target)
            (target / '08_model_metrics.csv').unlink()
            with self.assertRaisesRegex(ValueError, '08_model_metrics'):
                synthesis.load_upstream(ROOT / 'data/processed/train_eda.csv', target, pd)

    def test_report_covers_all_eight_steps_and_links_resolve(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/09_synthesis.py'),
                '--tables-dir', str(target / 'tables'), '--figures-dir', str(target / 'figures'),
                '--reports-dir', str(target / 'reports'),
                '--source-tables-dir', str(ROOT / 'outputs/tables'),
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(result.returncode, 0, result.stderr)
            report = target / 'reports/synthesis_summary.md'
            text = report.read_text(encoding='utf-8')
            for step in range(1, 9):
                self.assertIn(f'第{step:02d}步', text)
            for term in ['Mann', 'Kruskal', 'Lasso', 'silhouette', 'SHA256', '不作因果']:
                self.assertIn(term, text)
            metrics = pd.read_csv(ROOT / 'outputs/tables/08_model_metrics.csv')
            rmse = metrics.loc[(metrics.model == 'random_forest') & (metrics.split == 'final_test'), 'rmse'].iloc[0]
            self.assertIn(f'{rmse:.3f}', text)
            for link in re.findall(r'\]\(([^)]+)\)', text):
                if not link.startswith('http'):
                    self.assertTrue((report.parent / link).resolve().is_file(), link)
            manifest = json.loads((target / 'tables/09_run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['upstream_runs']), 3)
            self.assertTrue(manifest['evidence_files'])


if __name__ == '__main__':
    unittest.main()
