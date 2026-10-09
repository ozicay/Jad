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

from shap_works.optuna_xgb_configurable_shap_fixed_params import FEATURE_SCHEMAS, load_config, run_experiment

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('target', ['depresyon_skoru', 'anksiyete_skoru'])
@pytest.mark.parametrize('kind', ['mfcc', 'egemaps', 'is09'])
@pytest.mark.parametrize('k', [5, 10, 15])
def test_complete_configured_flow(target, kind, k, tmp_path, monkeypatch):
    name = f'{kind}_' + ('all' if k is None else f'top{k}')
    if target == 'anksiyete_skoru':
        from shap_works.optuna_xgb_configurable_shap_fixed_params_anksiyete import load_config as config_loader, run_experiment as runner
        config = config_loader(ROOT / 'shap_works/shap_fixed_param_configs/anksiyete' / f'anksiyete_{name}.json')
    else:
        config = load_config(ROOT / 'shap_works/shap_fixed_param_configs' / f'{name}.json')
        runner = run_experiment
    column, width = {'mfcc': ('features', 608), 'egemaps': ('egemaps_features', 88),
                     'is09': ('is09_features', 384)}[kind]
    assert (config['feature_column'], config['expected_feature_count']) == (column, width)
    assert config['target_column'] == target
    csv = tmp_path / 'data.csv'
    rows = []
    for pid in range(3):
        for segment in range(3):
            rows.append(dict(file_name=f'ad{pid}-clip{segment}.wav', label=pid,
                             depresyon_skoru=pid if target == 'depresyon_skoru' else 99,
                             anksiyete_skoru=pid if target == 'anksiyete_skoru' else 99,
                             **{column: ' '.join(str(pid * 1000 + f) for f in range(width))}))
    pd.DataFrame(rows).to_csv(csv, index=False)
    config.update(csv_path=str(csv), output_dir=str(tmp_path / 'output'))
    shap_calls = []
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
            self.training_rows = X.copy()
            return self
        @property
        def feature_importances_(self):
            pytest.fail("Gain importance must never be used")
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

    import shap
    class Explainer:
        def __init__(self, model):
            self.model = model
        def __call__(self, X_train):
            np.testing.assert_array_equal(X_train, self.model.training_rows)
            assert len(X_train) == 6
            shap_calls.append(X_train.shape[1])
            values = np.tile(np.arange(1, X_train.shape[1] + 1, dtype=float), (len(X_train), 1))
            values[::2] *= -1
            return SimpleNamespace(values=values)
    monkeypatch.setattr(shap, 'TreeExplainer', Explainer)
    monkeypatch.setattr(xgboost, 'XGBRegressor', Model)
    monkeypatch.setattr(optuna, 'create_study', create_study)
    monkeypatch.setattr(gc, 'collect', lambda: None)
    monkeypatch.setattr(__import__('joblib'), 'dump', lambda model, path: Path(path).write_text('synthetic'))
    gain_dir = tmp_path / 'gain'
    gain_dir.mkdir()
    pd.DataFrame([dict(setting='ALL_FEATURES', num_patients=3, patient_level_rmse=1., patient_level_mae=1.), dict(setting=f'TOP_{k}_FEATURES', num_patients=3, patient_level_rmse=1., patient_level_mae=1.)]).to_csv(gain_dir / 'final_patient_level_rmse_mae.csv', index=False)
    pd.DataFrame([dict(patient_id=f'depr_ad{pid}', true_score=pid, predicted_score=2.) for pid in range(3)]).to_csv(gain_dir / f'enhanced_combined_standard_loo_TOP{k}_results.csv', index=False)
    config['gain_output_dir'] = str(gain_dir)
    monkeypatch.setattr(optuna, 'create_study', lambda **kw: pytest.fail('No Optuna permitted'))
    runner(config)
    expected_trial_widths = [width] * 3 + ([] if k is None else [k] * 3)
    assert [w for w, _ in fits[:len(expected_trial_widths)]] == expected_trial_widths
    assert [w for w, _ in fits[-3:]] == [width if k is None else k] * 3
    assert len(fits) == 6
    assert all(params == dict(**config['fixed_params'], objective='reg:squarederror', random_state=42, tree_method='hist', device='cuda') for _, params in fits)
    for _, params in fits[:20 * len(expected_trial_widths)]:
        assert params['tree_method'] == 'hist'
        assert params['device'] == 'cuda'
        assert params['objective'] == 'reg:squarederror'
        assert params['random_state'] == 42
    assert trials == [] # baseline objective = mean per-patient RMSE
    output = Path(config['output_dir'])
    metrics = pd.read_csv(output / 'final_patient_level_rmse_mae.csv')
    assert metrics.setting.tolist() == ['ALL_FEATURES'] + ([] if k is None else [f'TOP_{k}_FEATURES'])
    assert metrics.num_patients.tolist() == [3] * len(metrics)
    assert metrics.patient_level_rmse.tolist() == pytest.approx([np.sqrt(5 / 3)] * len(metrics), abs=1e-6)
    assert metrics.patient_level_mae.tolist() == pytest.approx([1.] * len(metrics))
    result = pd.read_csv(output / 'enhanced_combined_standard_loo_no_flag_results.csv')
    assert result.predicted_score.iloc[:3].tolist() == [2.] * 3
    assert result.patient_id.iloc[:3].str.startswith('depr_ad').all()
    assert not (output / 'SHAP_OPTUNA_TMP_IMPORTANCE.csv').exists()
    assert (output / 'gain_vs_shap_by_patient.csv').exists()
    assert len(list(output.glob('patient_*/*.pkl'))) == (3 if k is None else 6)


    assert shap_calls.count(width) == 3
    if k is not None:
        assert shap_calls.count(k) == 0
        selected = pd.read_csv(output / f'selected_top{k}_features.csv')
        assert selected.feature_index.tolist() == list(range(width - 1, width - k - 1, -1))
    ranking = pd.read_csv(output / 'shap_global_ranking.csv')
    assert len(ranking) == width
    assert ranking.columns.tolist() == ['rank', 'feature', 'feature_index', 'mean_abs_shap']
    assert (output / 'shap_importance_all_models.csv').exists()
    assert (output / 'shap_importance_summary.csv').exists()



def test_fixed_script_has_no_tuning_and_preserves_final_method():
    source=(ROOT / 'shap_works/optuna_xgb_configurable_shap_fixed_params.py').read_text()
    for forbidden in ['optuna', 'study.optimize', 'create_study', 'model.feature_importances_', 'head(15)', 'train_test_split', 'GroupKFold']:
        assert forbidden not in source
    assert 'best_params = config["fixed_params"].copy()' in source
    assert 'np.mean(np.abs(shap_array), axis=0)' in source
    base=ast.parse((ROOT / 'shap_works/optuna_xgb_configurable.py').read_text())
    new=ast.parse(source)
    for name in ['parse_token','parse_features','extract_patient_id_from_filename']:
        fn=lambda t: next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(fn(base))==ast.dump(fn(new))


def test_params_match_verified_gain_registry():
    records=json.loads((ROOT/'shap_works/gain_best_params_verified.json').read_text())
    assert len(records)==24
    for record in records:
        if record['target_column']=='depresyon_skoru' and record['top_k'] is not None:
            c=load_config(ROOT/'shap_works/shap_fixed_param_configs'/f"{record['experiment_name']}.json")
            assert c['fixed_params']==record['best_params']
            assert c['gain_parameter_source']==record['stdout']
            assert c['csv_path']==record['config']['csv_path']


def test_fixed_anxiety_only_changes_target_validation():
    depression=(ROOT/'shap_works/optuna_xgb_configurable_shap_fixed_params.py').read_text()
    anxiety=(ROOT/'shap_works/optuna_xgb_configurable_shap_fixed_params_anksiyete.py').read_text()
    assert anxiety.replace('anksiyete_skoru','depresyon_skoru')==depression


def test_fixed_anxiety_params_match_gain_registry():
    from shap_works.optuna_xgb_configurable_shap_fixed_params_anksiyete import load_config as anxiety_loader
    records=json.loads((ROOT/'shap_works/gain_best_params_verified.json').read_text())
    for x in records:
        if x['target_column']=='anksiyete_skoru' and x['top_k'] is not None:
            name='anksiyete_'+x['experiment_name'].removeprefix('anx_')
            c=anxiety_loader(ROOT/'shap_works/shap_fixed_param_configs/anksiyete'/f'{name}.json')
            assert c['fixed_params']==x['best_params']
            assert c['gain_parameter_source']==x['stdout']
            assert c['csv_path']==x['config']['csv_path']
