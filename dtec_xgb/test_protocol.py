"""Small synthetic CPU tests; never train on clinical data or submit Slurm jobs."""
import ast
import concurrent.futures
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import controller
import data
import worker


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dtec-protocol-')
        self.root = Path(self.tmp.name)
        rng = np.random.RandomState(7)
        self.X = rng.normal(size=(20, 20))
        self.y = np.repeat([3., 8., 13., 18.], 5)
        self.groups = np.repeat(['p0', 'p1', 'p2', 'p3'], 5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reference_objective_top5_top10_top15(self):
        tree = ast.parse((data.ROOT / 'reference' / 'optuna_xgb_is09_top15.py').read_text())
        original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'objective')
        params = dict(learning_rate=.03, max_depth=4, n_estimators=100)
        for k in (5, 10, 15):
            # Parameterize only reference's literal head(15), as required for the three Top-K configs.
            node = ast.parse(ast.unparse(original)).body[0]
            for item in ast.walk(node):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == 'head':
                    item.args = [ast.Constant(k)]
            scope = dict(np=np, pd=pd, gc=worker.gc, mean_squared_error=mean_squared_error,
                         X_all=self.X, y_all=self.y, groups_all=self.groups,
                         unique_pids=np.unique(self.groups),
                         patient_info={p: {'depresyon_skoru': self.y[self.groups == p][0]} for p in np.unique(self.groups)})
            def cpu_model(**kwargs):
                kwargs.update(device='cpu', n_jobs=1)
                return XGBRegressor(**kwargs)
            scope['XGBRegressor'] = cpu_model
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<reference objective>', 'exec'), scope)
            reference_dir = self.root / f'reference{k}'
            reference_dir.mkdir()
            old = Path.cwd()
            try:
                os.chdir(reference_dir)
                with contextlib.redirect_stdout(io.StringIO()):
                    expected = scope['objective'](optuna.trial.FixedTrial(params))
            finally:
                os.chdir(old)
            result_dir = self.root / f'new{k}'
            result_dir.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                actual = worker.objective(optuna.trial.FixedTrial(params), self.X, self.y, self.groups,
                                          k, result_dir, device='cpu', threads=1)
            self.assertEqual(expected, actual)
            old_imp = pd.read_csv(reference_dir / 'OPTUNA_TMP_IMPORTANCE.csv')
            new_imp = pd.read_csv(result_dir / 'trial_000' / 'OPTUNA_TMP_IMPORTANCE.csv')
            np.testing.assert_array_equal(old_imp.filter(like='feat_').values, new_imp.filter(like='feat_').values)

    def test_final_all_and_topk_metrics_model_and_single_all_lopo(self):
        params = dict(learning_rate=.03, max_depth=4, n_estimators=100)
        keys = [(pid, i) for pid in np.unique(self.groups) for i in range(5)]
        manifest = {'input_sha256': {'synthetic': 'test'}}
        for name, family, k in data.CONFIGURATIONS:
            path = self.root / name
            path.mkdir()
            (path / 'selected_11_chunks.csv').write_text('synthetic smoke selection\n')
            with patch.object(worker, 'lopo', wraps=worker.lopo) as run_lopo:
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics = worker.final_evaluation(self.X, self.y, self.groups, k, params, path,
                                                     'cpu', 1, family, manifest, keys, smoke=True)
                self.assertEqual(run_lopo.call_count, 1 if k is None else 2)
            predictions = pd.read_csv(path / 'patient_predictions.csv')
            err = predictions.true_score - predictions.predicted_score
            self.assertAlmostEqual(metrics['patient_level_rmse'], np.sqrt(np.mean(err**2)), places=12)
            self.assertAlmostEqual(metrics['patient_level_mae'], np.mean(np.abs(err)), places=12)
            info = json.loads((path / 'best_model_info.json').read_text())
            self.assertAlmostEqual(info['AE'], predictions.AE.min(), places=12)
            self.assertEqual(len(list(path.glob('*.pkl'))), 1)
            model = joblib.load(path / 'best_model.pkl')
            mask = self.groups == info['patient_id']
            med = float(np.median(model.predict(self.X[mask][:, info['feature_indices']])))
            self.assertEqual(med, info['predicted_score'])
            self.assertEqual(model.n_features_in_, self.X.shape[1] if k is None else k)
        all_dir = self.root / 'objective_all'
        all_dir.mkdir()
        with patch.object(worker, 'lopo', wraps=worker.lopo) as run_lopo:
            with contextlib.redirect_stdout(io.StringIO()):
                worker.objective(optuna.trial.FixedTrial(params), self.X, self.y, self.groups,
                                 None, all_dir, device='cpu', threads=1, smoke=True)
            self.assertEqual(run_lopo.call_count, 1)

    def test_gpu_guards_and_scheduler(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'synthetic', 'CUDA_VISIBLE_DEVICES': '2'}):
            self.assertEqual(worker.gpu_assignment()['xgboost_device'], 'cuda:0')
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'synthetic', 'CUDA_VISIBLE_DEVICES': '0,1'}):
            with self.assertRaises(RuntimeError):
                worker.gpu_assignment()
        active = peak = 0
        completed = []
        lock = threading.Lock()
        def fake_launch(name, *_):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.02)
            with lock:
                active -= 1
                completed.append(name)
            return name, 0
        with contextlib.redirect_stdout(io.StringIO()):
            controller.schedule(self.root, self.root, self.root, 20, fake_launch)
        self.assertEqual(peak, 4)
        self.assertEqual(set(completed), {c[0] for c in data.CONFIGURATIONS})
        command = controller.launch_command('mfcc_all', self.root, self.root, self.root, 20)
        self.assertIn('--gpus-per-task=1', command)
        self.assertIn('--exclusive', command)
        self.assertIn('--gpu-bind=single:1', command)

    def test_reference_final_top15_predictions_and_indices(self):
        tree = ast.parse((data.ROOT / 'reference' / 'optuna_xgb_is09_top15.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_loo_model_with_importance')
        params = dict(learning_rate=.03, max_depth=4, n_estimators=100)
        def cpu_model(**kwargs):
            kwargs.update(device='cpu', n_jobs=1)
            return XGBRegressor(**kwargs)
        scope = dict(np=np, os=os, gc=worker.gc, joblib=joblib, model_lock=threading.Lock(),
                     best_params=params, CUDA_DEVICE='cpu', XGBRegressor=cpu_model,
                     mean_squared_error=mean_squared_error)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference final fold>', 'exec'), scope)
        def reference_stage(X):
            results = []
            for pid in np.unique(self.groups):
                mask = self.groups == pid
                with contextlib.redirect_stdout(io.StringIO()):
                    result = scope['train_loo_model_with_importance']((pid, X[~mask], self.y[~mask],
                                X[mask], self.y[mask], self.y[mask][0], 'synthetic', str(self.root)))
                self.assertEqual(result['status'], 'success')
                results.append(result)
            return results
        full = reference_stage(self.X)
        imp = pd.DataFrame([{f'feat_{i}': v for i, v in enumerate(r['feature_importance'])} for r in full])
        path = self.root / 'reference_final_importance.csv'
        imp.to_csv(path, index=False, float_format='%.6f')
        avg = pd.read_csv(path).mean().sort_values(ascending=False)
        expected_indices = sorted(int(f.split('_')[1]) for f in avg.head(15).index)
        expected = reference_stage(self.X[:, expected_indices])
        output = self.root / 'final_parity'
        output.mkdir()
        (output / 'selected_11_chunks.csv').write_text('synthetic\n')
        with contextlib.redirect_stdout(io.StringIO()):
            worker.final_evaluation(self.X, self.y, self.groups, 15, params, output, 'cpu', 1, 'is09',
                                    {'input_sha256': {}}, [(p, i) for p in np.unique(self.groups) for i in range(5)])
        actual = pd.read_csv(output / 'patient_predictions.csv')
        np.testing.assert_array_equal(actual.predicted_score.values, [float(r['predicted_score']) for r in expected])
        self.assertEqual(json.loads((output / 'best_model_info.json').read_text())['feature_indices'], expected_indices)

    def test_real_metadata_selection_and_reordering(self):
        # Validation only on real metadata; no real-data models are fitted.
        if not (data.DEFAULT_DATA / 'metadata.csv').exists():
            self.skipTest('Private DTEC metadata is not present; synthetic protocol tests still run')
        selection = self.root / 'selection'
        report = data.prepare_selection(data.DEFAULT_DATA, selection)
        self.assertEqual((report['patients'], report['chunks']), (62, 1643))
        tables, keys, manifest = data.verify_selection(data.DEFAULT_DATA, selection)
        self.assertEqual(len(keys), 682)
        reverse = self.root / 'reordered'
        reverse.mkdir()
        for name in ['metadata.csv'] + [v[0] for v in data.FAMILIES.values()]:
            with (data.DEFAULT_DATA / name).open() as f:
                reader = csv.DictReader(f)
                columns, rows = reader.fieldnames, list(reader)
            with (reverse / name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(reversed(rows))
        _, reordered_keys, _ = data.verify_selection(reverse, selection)
        self.assertEqual(keys, reordered_keys)
        for family in data.FAMILIES:
            with patch.object(data, 'expected_selection', side_effect=AssertionError('Workers must not sample')):
                _, _, _, family_keys, _ = data.load_family(reverse, selection, family)
            self.assertEqual(keys, family_keys)
        # A changed clinical target must fail before training.
        path = reverse / data.FAMILIES['mfcc'][0]
        with path.open() as f:
            reader = csv.DictReader(f)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]['depresyon_skoru'] = '99'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            data.verify_selection(reverse, selection)


if __name__ == '__main__':
    unittest.main(verbosity=2)
