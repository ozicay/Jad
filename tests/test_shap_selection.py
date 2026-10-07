"""Exercise isolated source functions without starting the clinical experiment."""
import ast
from pathlib import Path
from types import SimpleNamespace
from threading import Lock

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'optuna_xgb_is09_top15_shap.py').read_text()
TREE = ast.parse(SOURCE)


def source_function(name, **globals_):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(np=np, pd=pd, shap=shap, **globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT), 'exec'), ns)
    return ns[name]


@pytest.mark.parametrize('kind', ['explanation', 'array', 'legacy'])
def test_mean_absolute_training_shap(kind):
    values = np.array([[1., -2., 3.], [-1., 4., -9.]])
    train = np.zeros_like(values)
    seen = []

    def explain(X):
        seen.append(X)
        return SimpleNamespace(values=values) if kind == 'explanation' else values

    explainer = SimpleNamespace(shap_values=explain) if kind == 'legacy' else explain
    fake = SimpleNamespace(TreeExplainer=lambda model: explainer)
    helper = source_function('mean_abs_training_shap')
    helper.__globals__['shap'] = fake
    result = helper(object(), train)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, [1., 3., 6.])
    assert seen[0] is train


def test_real_xgboost_shap_smoke():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 20))
    y = X[:, 3] - X[:, 7]
    model = XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, random_state=42).fit(X, y)
    actual = source_function('mean_abs_training_shap')(model, X)
    expected = np.abs(shap.TreeExplainer(model)(X).values).mean(axis=0)
    assert actual.shape == (20,)
    np.testing.assert_allclose(actual, expected)


def test_aggregation_ranking_and_original_columns(tmp_path):
    # Execute the actual final ranking/slicing source, including its CSV outputs.
    start = SOURCE.index('avg_importance = importance_df[feature_cols].mean().sort_values(ascending=False)')
    stop = SOURCE.index('# 3) LOO işler listesini yeniden hazırla', start)
    folds = np.array([np.arange(20), np.arange(20) * 3.], dtype=float)
    frame = pd.DataFrame(folds, columns=[f'feat_{i}' for i in range(20)])
    X = np.arange(140).reshape(7, 20)
    ns = dict(np=np, pd=pd, os=__import__('os'), importance_df=frame,
              feature_cols=list(frame.columns), X_all=X, base_model_dir=str(tmp_path))
    exec(SOURCE[start:stop], ns)
    np.testing.assert_array_equal(ns['avg_importance'].values, 2 * np.arange(19, -1, -1))
    assert ns['top_feature_indices'] == list(range(5, 20))
    assert ns['X_top15'].shape == (7, 15)
    np.testing.assert_array_equal(ns['X_top15'], X[:, 5:20])
    ranking = pd.read_csv(tmp_path / 'shap_global_ranking.csv')
    selected = pd.read_csv(tmp_path / 'selected_top15_features.csv')
    assert ranking['mean_abs_shap'].is_monotonic_decreasing
    assert len(selected) == 15
    assert selected['feature_index'].tolist() == list(range(19, 4, -1))


def test_final_fold_explains_training_only(tmp_path):
    train = np.zeros((4, 20))
    test = np.full((2, 20), 999.)
    seen = []

    class Model:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            assert X is train
        def predict(self, X):
            assert X is test
            return np.array([2., 4.])

    def importance(model, X):
        seen.append(X)
        return np.arange(20.)

    fn = source_function('train_loo_model_with_importance',
        XGBRegressor=Model, best_params={}, CUDA_DEVICE='cuda:0',
        mean_abs_training_shap=importance, mean_squared_error=mean_squared_error,
        os=__import__('os'), joblib=SimpleNamespace(dump=lambda *args: None),
        model_lock=Lock(), gc=SimpleNamespace(collect=lambda: None))
    result = fn(('heldout', train, np.zeros(4), test, np.zeros(2), 3., 'depr_only', str(tmp_path)))
    assert result['status'] == 'success'
    assert result['predicted_score'] == 3.
    assert len(seen) == 1 and seen[0] is train


def test_all_shap_calls_use_x_train_and_preserve_lopo():
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'mean_abs_training_shap']
    assert len(calls) == 2
    assert all(isinstance(c.args[1], ast.Name) and c.args[1].id == 'X_train' for c in calls)
    original = ast.parse((ROOT / 'optuna_xgb_is09_top15.py').read_text())
    # Compare all split assignments structurally, ignoring Top-15 naming corrections.
    def splits(tree):
        names = {'mask', 'target_mask', 'train_mask', 'X_train', 'y_train',
                 'X_test', 'X_target_test', 'y_target_test', 'groups_all', 'unique_pids',
                 'patient_info', 'true_score', 'true'}
        return [ast.unparse(n).replace('top40', 'top15').replace('top10_idx', 'top15_idx')
                for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    assert splits(original) == splits(TREE)
    for expression in ['n_trials=20', 'random_state=42', 'MAX_WORKERS = 5',
                       'direction="minimize"', 'np.median(target_predictions)',
                       'np.median(preds)']:
        assert expression in SOURCE


def test_optuna_each_fold_explains_training_only(tmp_path):
    X = np.arange(120.).reshape(6, 20)
    groups = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    seen = []
    fitted = []

    class Model:
        def __init__(self, **kwargs):
            pass
        def fit(self, X_train, y_train):
            fitted.append(X_train.copy())
        def predict(self, X_test):
            return np.ones(len(X_test))

    def importance(model, X_train):
        seen.append(X_train.copy())
        return np.arange(20.)

    trial = SimpleNamespace(suggest_float=lambda *a, **k: .01,
                            suggest_int=lambda name, *a: 4 if name == 'max_depth' else 100)
    fn = source_function('objective', XGBRegressor=Model,
        X_all=X, y_all=np.ones(6), groups_all=groups, unique_pids=np.unique(groups),
        patient_info={p: {'depresyon_skoru': 1.} for p in np.unique(groups)},
        mean_abs_training_shap=importance, mean_squared_error=mean_squared_error,
        base_model_dir=str(tmp_path), os=__import__('os'),
        gc=SimpleNamespace(collect=lambda: None))
    assert fn(trial) == 0.
    assert len(seen) == 3
    assert len(fitted) == 6
    for pid, explained in zip(np.unique(groups), seen):
        np.testing.assert_array_equal(explained, X[groups != pid])
        assert not any(np.array_equal(row, heldout) for row in explained for heldout in X[groups == pid])
    fn(trial)
    assert len(seen) == 6  # Every trial creates its own explanations.


def test_model_parameters_optuna_ranges_and_metrics_unchanged():
    original = ast.parse((ROOT / 'optuna_xgb_is09_top15.py').read_text())
    def relevant(tree):
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            text = ast.unparse(node.func)
            if text in {'XGBRegressor', 'mean_squared_error', 'np.median',
                        'trial.suggest_float', 'trial.suggest_int',
                        'optuna.create_study', 'study.optimize'}:
                out.append(ast.unparse(node))
        return out
    assert relevant(original) == relevant(TREE)
