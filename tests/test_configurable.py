"""Synthetic model doubles exercise configuration and unchanged LOPO data flow."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import gc

import numpy as np
import pandas as pd
import pytest
import optuna
import xgboost

from optuna_xgb_configurable import FEATURE_SCHEMAS, load_config, run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('kind', ['mfcc', 'egemaps', 'is09'])
@pytest.mark.parametrize('k', [5, 10, 15, None])
def test_complete_configured_flow(kind, k, tmp_path, monkeypatch):
    name = f'{kind}_' + ('all' if k is None else f'top{k}')
    config = load_config(ROOT / 'shap_works/configs' / f'{name}.json')
    column, width = {'mfcc': ('features', 608), 'egemaps': ('egemaps_features', 88),
                     'is09': ('is09_features', 384)}[kind]
    assert (config['feature_column'], config['expected_feature_count']) == (column, width)
    assert config['target_column'] == 'depresyon_skoru'
    csv = tmp_path / 'data.csv'
    rows = []
    for pid in range(3):
        for segment in range(3):
            rows.append(dict(file_name=f'ad{pid}-clip{segment}.wav', label=pid,
                             depresyon_skoru=pid, anksiyete_skoru=99,
                             **{column: ' '.join(str(pid * 1000 + f) for f in range(width))}))
    pd.DataFrame(rows).to_csv(csv, index=False)
    config.update(csv_path=str(csv), output_dir=str(tmp_path / 'output'))
    fits = []
    trials = []

    class Model:
        def __init__(self, **params):
            self.params = params
        def fit(self, X, y):
            assert len(X) == 6  # all segments of two training patients
            assert len(np.unique(y)) == 2
            self.training_patients = set(y)
            self.width = X.shape[1]
            fits.append((self.width, self.params))
            self.feature_importances_ = np.arange(1, self.width + 1, dtype=float)
            return self
        def predict(self, X):
            assert len(X) == 3  # held-out patient's ALL segments
            heldout = int(X[0, 0] // 1000)
            assert heldout not in self.training_patients
            assert np.all(X[:, 0] // 1000 == heldout)
            # Median=2, mean !=2; reveals any aggregation change.
            return np.array([1., 2., 100.])

    class Trial:
        def suggest_float(self, name, lo, hi, log):
            assert (name, lo, hi, log) == ('learning_rate', 1e-3, .1, True)
            return .01
        def suggest_int(self, name, lo, hi):
            assert (name, lo, hi) in [('max_depth', 4, 14), ('n_estimators', 100, 500)]
            return lo

    class Study:
        best_params = dict(learning_rate=.01, max_depth=4, n_estimators=100)
        best_value = 1.
        def optimize(self, objective, n_trials):
            assert n_trials == 20
            for _ in range(n_trials):
                trials.append(objective(Trial()))

    def create_study(*, direction):
        assert direction == 'minimize'
        return Study()

    monkeypatch.setattr(xgboost, 'XGBRegressor', Model)
    monkeypatch.setattr(optuna, 'create_study', create_study)
    monkeypatch.setattr(gc, 'collect', lambda: None)
    monkeypatch.setattr(__import__('joblib'), 'dump', lambda model, path: Path(path).write_text('synthetic'))
    run_experiment(config)
    expected_trial_widths = [width] * 3 + ([] if k is None else [k] * 3)
    assert [w for w, _ in fits[:len(expected_trial_widths)]] == expected_trial_widths
    assert [w for w, _ in fits[-3:]] == [width if k is None else k] * 3
    assert len(fits) == 20 * len(expected_trial_widths) + (3 if k is None else 6)
    for _, params in fits[:20 * len(expected_trial_widths)]:
        assert params['tree_method'] == 'hist'
        assert params['device'] == 'cuda'
        assert params['objective'] == 'reg:squarederror'
        assert params['random_state'] == 42
    assert trials == pytest.approx([1.] * 20) # baseline objective = mean per-patient RMSE
    output = Path(config['output_dir'])
    metrics = pd.read_csv(output / 'final_patient_level_rmse_mae.csv')
    assert metrics.setting.tolist() == ['ALL_FEATURES'] + ([] if k is None else [f'TOP_{k}_FEATURES'])
    assert metrics.num_patients.tolist() == [3] * len(metrics)
    assert metrics.patient_level_rmse.tolist() == pytest.approx([np.sqrt(5 / 3)] * len(metrics), abs=1e-6)
    assert metrics.patient_level_mae.tolist() == pytest.approx([1.] * len(metrics))
    result = pd.read_csv(output / 'enhanced_combined_standard_loo_no_flag_results.csv')
    assert result.predicted_score.iloc[:3].tolist() == [2.] * 3
    assert result.patient_id.iloc[:3].str.startswith('depr_ad').all()
    assert (output / 'OPTUNA_TMP_IMPORTANCE.csv').exists() == (k is not None)
    assert len(list(output.glob('patient_*/*.pkl'))) == (3 if k is None else 6)


def test_wrong_dimension_stops_before_optuna(tmp_path, monkeypatch):
    config = load_config(ROOT / 'shap_works/configs/mfcc_all.json')
    csv = tmp_path / 'wrong.csv'
    pd.DataFrame([dict(file_name='ad1-a.wav', depresyon_skoru=1, features='1 2')]).to_csv(csv, index=False)
    config.update(csv_path=str(csv), output_dir=str(tmp_path / 'out'))
    monkeypatch.setattr(optuna, 'create_study', lambda **kw: pytest.fail('must stop before tuning'))
    with pytest.raises(ValueError, match='expected 608 features, got 2'):
        run_experiment(config)


def test_numeric_parser_and_patient_extraction_match_baseline():
    baseline = ast.parse((ROOT / 'optuna_xgb_is09_top15.py').read_text())
    new = ast.parse((ROOT / 'optuna_xgb_configurable.py').read_text())
    for name in ['parse_token', 'parse_features', 'extract_patient_id_from_filename', 'train_loo_model_with_importance']:
        old_fn = next(n for n in ast.walk(baseline) if isinstance(n, ast.FunctionDef) and n.name == name)
        new_fn = next(n for n in ast.walk(new) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old_fn) == ast.dump(new_fn)


@pytest.mark.parametrize('update', [dict(target_column='anksiyete_skoru'), dict(top_k=608),
                                     dict(feature_column='is09_features'), dict(expected_feature_count=1)])
def test_reject_invalid_config(update, tmp_path):
    config = load_config(ROOT / 'shap_works/configs/mfcc_all.json')
    config.update(update)
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)
