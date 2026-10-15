"""Synthetic tests: no clinical training, XGBoost fitting, GPU or Slurm use.

python -B edaic_xgb/test_runner.py
"""
import ast
import contextlib
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import weakref

import joblib
import numpy as np
import optuna
import pandas as pd

SOURCE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('edaic_runner', SOURCE / 'run_edaic_xgb.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SyntheticModel:
    fits = []
    living = weakref.WeakSet()
    max_living = 0

    def __init__(self, **params):
        self.params = params
        assert params['objective'] == 'reg:squarederror'
        assert params['device'] == 'cuda' and params['random_state'] == 42
        type(self).living.add(self)
        type(self).max_living = max(type(self).max_living, len(type(self).living))

    def fit(self, X, y):
        assert len(X) == len(y) == 6
        self.train_subjects = set(X[:, 0].astype(int))
        assert len(self.train_subjects) == 2
        self.feature_importances_ = np.linspace(1, 0, X.shape[1])
        type(self).fits.append((X.shape[1], self.params.copy()))

    def predict(self, X):
        assert len(X) == 3
        subject = int(X[0, 0])
        assert subject not in self.train_subjects
        assert set(X[:, 0].astype(int)) == {subject}
        bias = {1: 0.1, 2: 1.0, 3: 2.0}[subject]
        # Median differs from mean, so this checks the required aggregation.
        return X[:, 0] + bias + np.array([-1., 0., 10.])

    def get_booster(self):
        return self

    def save_config(self):
        return json.dumps({'learner': {'generic_param': {'device': 'synthetic'}}})


def main():
    m.verify_reference()
    real_collect = gc.collect
    gc.collect = lambda: None
    m.XGBRegressor = SyntheticModel
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        with tempfile.TemporaryDirectory(prefix='edaic-optuna-synthetic-') as temp:
            m.ROOT = Path(temp)
            m.REFERENCE = SOURCE / 'reference/optuna_xgb_is09_top15.py'
            (m.ROOT / 'SOURCE_PARENT_COMMIT').write_text((SOURCE / 'SOURCE_PARENT_COMMIT').read_text())
            (m.ROOT / 'SOURCE_COMMIT').write_text('a' * 40)
            (m.ROOT / 'metadata').mkdir()
            m.CONFIGS = {f: (m.ROOT / 'metadata' / f'{f}.csv', c)
                         for f, (_, c) in m.CONFIGS.items()}
            m.EXPECTED_DIMENSIONS = {f: 20 for f in m.CONFIGS}
            for family, (path, column) in m.CONFIGS.items():
                rows = []
                for subject in (1, 2, 3):
                    for segment in range(3):
                        rows.append({'file_name': f'{subject}-{segment}.wav',
                                     m.TARGET: subject,
                                     column: ','.join(map(str, [subject]*20))})
                pd.DataFrame(rows).to_csv(path, index=False)
            data, common = m.load_and_validate()
            # Execute the original objective unmodified against the same synthetic models.
            reference_text = m.REFERENCE.read_text()
            tree = ast.parse(reference_text)
            original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'objective')
            X, y, groups, pids, scores = data['is09']
            namespace = dict(vars(m), X_all=X, y_all=y, groups_all=groups, unique_pids=pids,
                             patient_info={pid: {m.TARGET: score} for pid, score in scores.items()})
            exec(compile(ast.Module(body=[original], type_ignores=[]), str(m.REFERENCE), 'exec'), namespace)
            fixed = dict(learning_rate=0.02, max_depth=7, n_estimators=200)
            previous = Path.cwd()
            os.chdir(temp)
            try:
                expected = namespace['objective'](optuna.trial.FixedTrial(fixed))
                actual = m.reference_objective(optuna.trial.FixedTrial(fixed), X, y, groups, pids,
                                                namespace['patient_info'], 15, m.ROOT)
            finally:
                os.chdir(previous)
            assert expected == actual
            print('PASS: unmodified reference Top-15 objective equals adapted objective.')
            SyntheticModel.fits.clear()
            m.main()
            assert json.loads((m.ROOT / 'run_manifest.json').read_text())['source_commit'] == 'a' * 40
            assert len(SyntheticModel.fits) == 1323
            summary = pd.read_csv(m.ROOT / 'experiment_summary.csv')
            assert len(summary) == 12
            assert summary.experiment_name.tolist() == [f'{f}_{"all" if k is None else f"top{k}"}' for f, k in m.ORDER]
            assert np.allclose(summary.patient_level_rmse, np.sqrt(5.01/3))
            assert np.allclose(summary.patient_level_mae, 3.1/3)
            assert (summary.loc[summary.top_k.notna(), 'patient_level_rmse'] > summary.loc[summary.top_k.notna(), 'best_fold_rmse']).all()
            assert pd.read_csv(m.ROOT / 'patient_fold_index.csv').patient_id.tolist() == [1, 2, 3]
            for f, k in m.ORDER:
                out = m.ROOT / f'{f}_{"all" if k is None else f"top{k}"}'
                stage = 'all_feature' if k is None else f'top{k}'
                results = pd.read_csv(out / f'{stage}_patient_results.csv')
                assert len(results) == 3 and results.patient_id.tolist() == [1, 2, 3]
                assert results.features_count.eq(20 if k is None else k).all()
                assert len(list(out.glob('*.pkl'))) == (0 if k is None else 1)
                trials = pd.read_csv(out / 'optuna_trials.csv')
                assert len(trials) == 20 and trials.state.eq('COMPLETE').all()
                study = optuna.load_study(study_name=out.name, storage=f"sqlite:///{out / 'optuna_study.sqlite3'}")
                distributions = study.trials[0].distributions
                assert distributions['learning_rate'] == optuna.distributions.FloatDistribution(1e-3, 0.1, log=True)
                assert distributions['max_depth'] == optuna.distributions.IntDistribution(4, 14)
                assert distributions['n_estimators'] == optuna.distributions.IntDistribution(100, 500)
                assert all(np.isclose(t.value, 3.1/3) for t in study.trials)
                assert sum(len(v) for v in np.load(out / f'{stage}_segment_predictions.npz').values()) == 9
                if k is not None:
                    model = joblib.load(out / 'best_topk_model.pkl')
                    assert model.train_subjects == {2, 3}
                    assert 'tree_method' not in model.params
                    assert len(pd.read_csv(out / f'selected_top{k}_features.csv')) == k
                    del model
                else:
                    assert not list(out.glob('selected*')) and not list(out.glob('top*'))
            before = len(SyntheticModel.fits)
            m.main()
            assert len(SyntheticModel.fits) == before  # Resume completed configs: zero refits.
            assert SyntheticModel.max_living <= 7
            # Resume a partially completed experiment without spending another 20 trials.
            out = m.ROOT / 'mfcc_top5'
            (out / 'experiment_complete.json').unlink()
            m.experiment('mfcc', 5, data['mfcc'])
            assert len(SyntheticModel.fits) == before + 6
            # Population mismatch must fail before any fit.
            path, col = m.CONFIGS['is09']
            df = pd.read_csv(path)
            df = df[~df.file_name.str.startswith('3-')]
            df.to_csv(path, index=False)
            try:
                m.load_and_validate()
                raise AssertionError('Population mismatch was not rejected')
            except ValueError as error:
                assert 'populations differ' in str(error)
            print('PASS: 12 configs, 240 real Optuna trials with synthetic fits; masks, median, all-patient metrics, one model per Top-K, ALL bypass, resume, population rejection.')
    finally:
        gc.collect = real_collect


if __name__ == '__main__':
    main()
