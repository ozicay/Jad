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

from shap_works.optuna_xgb_configurable_shap import FEATURE_SCHEMAS, load_config, run_experiment

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('target', ['depresyon_skoru', 'anksiyete_skoru'])
@pytest.mark.parametrize('kind', ['mfcc', 'egemaps', 'is09'])
@pytest.mark.parametrize('k', [5, 10, 15, None])
def test_complete_configured_flow(target, kind, k, tmp_path, monkeypatch):
    name = f'{kind}_' + ('all' if k is None else f'top{k}')
    if target == 'anksiyete_skoru':
        from shap_works.optuna_xgb_configurable_shap_anksiyete import load_config as config_loader, run_experiment as runner
        config = config_loader(ROOT / 'shap_works/shap_configs/anksiyete' / f'anksiyete_{name}.json')
    else:
        config = load_config(ROOT / 'shap_works/shap_configs' / f'{name}.json')
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
    runner(config)
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
    assert (output / 'SHAP_OPTUNA_TMP_IMPORTANCE.csv').exists() == (k is not None)
    assert len(list(output.glob('patient_*/*.pkl'))) == (3 if k is None else 6)


    assert shap_calls.count(width) == (3 if k is None else 20 * 3 + 3)
    if k is not None:
        assert shap_calls.count(k) == 3
        selected = pd.read_csv(output / f'selected_top{k}_features.csv')
        assert selected.feature_index.tolist() == list(range(width - 1, width - k - 1, -1))
    ranking = pd.read_csv(output / 'shap_global_ranking.csv')
    assert len(ranking) == width
    assert ranking.columns.tolist() == ['rank', 'feature', 'feature_index', 'mean_abs_shap']
    assert (output / 'shap_importance_all_models.csv').exists()
    assert (output / 'shap_importance_summary.csv').exists()


def test_real_tree_shap_training_smoke():
    import shap
    from shap_works.optuna_xgb_configurable_shap import mean_abs_training_shap
    X = np.arange(80, dtype=float).reshape(20, 4)
    y = X[:, 2] / 10
    model = xgboost.XGBRegressor(n_estimators=2, max_depth=2, n_jobs=1).fit(X, y)
    actual = mean_abs_training_shap(model, X)
    expected = np.mean(np.abs(shap.TreeExplainer(model)(X).values), axis=0)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (4,)


@pytest.mark.parametrize('kind', ['explanation', 'array', 'legacy'])
def test_training_shap_compatibility(kind, monkeypatch):
    import shap
    from shap_works.optuna_xgb_configurable_shap import mean_abs_training_shap
    X = np.zeros((2, 3))
    values = np.array([[1., -2., 3.], [-3., 4., -5.]])
    seen = []
    def explain(data):
        seen.append(data)
        return SimpleNamespace(values=values) if kind == 'explanation' else values
    fake = SimpleNamespace(shap_values=explain) if kind == 'legacy' else explain
    monkeypatch.setattr(shap, 'TreeExplainer', lambda model: fake)
    np.testing.assert_array_equal(mean_abs_training_shap(object(), X), [2., 3., 4.])
    assert seen[0] is X


def test_only_training_importance_method_and_artifacts_changed():
    original = (ROOT / 'shap_works/optuna_xgb_configurable.py').read_text()
    modified = (ROOT / 'shap_works/optuna_xgb_configurable_shap.py').read_text()
    assert 'model.feature_importances_' not in modified
    assert modified.count('mean_abs_training_shap(model, X_train)') == 3
    assert 'np.mean(np.abs(shap_array), axis=0)' in modified
    # Normalize authorized fold-importance substitutions and output names.
    modified = modified.replace('mean_abs_training_shap(model, X_train) if top_k', 'model.feature_importances_.copy() if top_k')
    modified = modified.replace('feature_importance = mean_abs_training_shap(model, X_train)', 'feature_importance = model.feature_importances_')
    modified = modified.replace('shap_importance_all_models.csv', 'feature_importance_all_models.csv').replace('shap_importance_summary.csv', 'feature_importance_summary.csv').replace('SHAP_OPTUNA_TMP_IMPORTANCE.csv', 'OPTUNA_TMP_IMPORTANCE.csv')
    tree = ast.parse(modified)
    base = ast.parse(original)
    run = lambda t: next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == 'run_experiment')
    class RemoveArtifactWrites(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'shap_ranking' for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value).startswith('shap_ranking.'):
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if ast.unparse(node.test) == 'top_k is not None' and node.orelse and ast.unparse(node.orelse[0]) == 'X_selected = X_all':
                node.orelse = node.orelse[:1]
            return self.generic_visit(node)
    assert ast.dump(run(base)) == ast.dump(RemoveArtifactWrites().visit(run(tree)))


def test_anxiety_shap_only_changes_config_target():
    depression = (ROOT / 'shap_works/optuna_xgb_configurable_shap.py').read_text()
    anxiety = (ROOT / 'shap_works/optuna_xgb_configurable_shap_anksiyete.py').read_text()
    assert anxiety.replace('anksiyete_skoru', 'depresyon_skoru') == depression


def test_anxiety_shap_rejects_depression_config():
    from shap_works.optuna_xgb_configurable_shap_anksiyete import load_config as anxiety_loader
    with pytest.raises(ValueError, match='Target must be anksiyete_skoru'):
        anxiety_loader(ROOT / 'shap_works/shap_configs/mfcc_top5.json')
