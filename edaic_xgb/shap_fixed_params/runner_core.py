#!/usr/bin/env python3
"""EDAIC controlled SHAP: fixed matching gain parameters; nine Top-K configs."""
import argparse
import fcntl
import ast
import concurrent.futures
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from threading import Lock, local
from queue import Queue
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent
CUDA_DEVICE = 'cuda:0'
MAX_WORKERS = 1
GPU_DEVICES = ("cuda:0",)
CURRENT_TOP_K = None
from shap_ranking import mean_abs_training_shap
_GPU_LOCAL = local()
RESUME_FROM_SHA = None
TARGET = 'depresyon_skoru'
CONFIGS = {
    'mfcc': (ROOT.parent / 'metadata/metadata_with_mfcc.csv', 'features'),
    'egemaps': (ROOT.parent / 'metadata/metadata_with_egemapsv0b1.csv', 'egemaps_features'),
    'is09': (ROOT.parent / 'metadata/metadata_with_is09.csv', 'is09_features'),
}
ORDER = [(feature, k) for feature in CONFIGS for k in (5, 10, 15)]
TRIALS = 0
EXPECTED_DIMENSIONS = {'mfcc': 608, 'egemaps': 88, 'is09': 384}
REFERENCE = ROOT / 'reference/optuna_xgb_is09_top15.py'
PARAMS = dict(objective='reg:squarederror', random_state=42,
              device='cuda' if 'cuda' in CUDA_DEVICE else 'cpu')
num_re = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+(?:\.\d+)?)?')


def parse_token(tok: str) -> float:
    if 'e' in tok or 'E' in tok:
        mant, exp = re.split('[eE]', tok)
        return float(mant) * (10 ** float(exp))
    return float(tok)


def parse_features(s: str) -> np.ndarray:
    toks = re.split(r'[,\s]+', s)
    return np.array([parse_token(t) for t in toks if t and num_re.fullmatch(t)], dtype=float)


def extract_patient_id_from_filename(filename):
    base_name = os.path.basename(str(filename))
    patient_id = base_name.split('-')[0].lower()
    return patient_id


def now():
    return datetime.now(ZoneInfo('Europe/Istanbul')).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def journal(message):
    local = datetime.now(ZoneInfo('Europe/Istanbul'))
    directory = ROOT / 'docs/agent-journal'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{local.date()}.md').open('a') as f:
        f.write(f'\n- {local.isoformat()} — {message}\n')


def verify_reference():
    original = ast.parse(REFERENCE.read_text())
    current = ast.parse(Path(__file__).read_text())
    for name in ('parse_token', 'parse_features', 'extract_patient_id_from_filename'):
        old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
        for fn in (old, new):
            fn.body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        assert ast.dump(old) == ast.dump(new), name
    assert len(ORDER) == 9 and all(k in (5,10,15) for _,k in ORDER)
    print('PASS fixed-parameter protocol: unchanged parsing/IDs; nine Top-K configs; no tuning.')


def load_fixed_params(name):
    registry = json.loads((ROOT / 'gain_best_params_verified.json').read_text())
    record = registry[name]
    baseline = ROOT.parent / name
    if sha256(baseline / 'best_params.json') != record['best_params_sha256']:
        raise ValueError(f'{name}: original gain parameters changed')
    params = json.loads((baseline / 'best_params.json').read_text())
    assert params == record['params']
    csv_path, column = CONFIGS[name.split('_')[0]]
    assert sha256(csv_path) == record['dataset_sha256']
    return params





def load_and_validate():
    verify_reference()
    datasets = {}
    reference_segments = None
    common = None
    for feature, (path, column) in CONFIGS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, index_col=0, encoding='utf-8')
        if df.index.name != 'file_name' or column not in df or TARGET not in df:
            raise ValueError(f'{path}: expected file_name index, {column}, {TARGET}; got {df.columns.tolist()}')
        df['original_path'] = df.index
        df['patient_id'] = df['original_path'].map(extract_patient_id_from_filename)
        df[column] = df[column].apply(parse_features)
        df = df[df[column].notna()].reset_index(drop=True)
        dimensions = df[column].map(len)
        if dimensions.nunique() != 1 or dimensions.iloc[0] != EXPECTED_DIMENSIONS[feature]:
            raise ValueError(f'{feature}: inconsistent or insufficient feature dimensions')
        X_all = np.vstack(df[column].values)
        y_all = df[TARGET].values
        groups_all = df['patient_id'].values
        unique_pids = np.unique(groups_all)
        if len(unique_pids) < 2 or not np.isfinite(X_all).all() or not np.isfinite(y_all.astype(float)).all():
            raise ValueError(f'{feature}: invalid patient population/features/scores')
        if df.groupby('patient_id')[TARGET].nunique().max() != 1:
            raise ValueError(f'{feature}: conflicting depression scores within a patient')
        segment_scores = df.set_index('original_path')[TARGET].sort_index()
        if not segment_scores.index.is_unique:
            raise ValueError(f'{feature}: duplicate segments')
        print(f'{feature}: CSV={path}; feature_column={column}; dimension={X_all.shape[1]}; target={TARGET}; segments={len(df)}; patients={len(unique_pids)}')
        print(f'{feature} sorted patient IDs: {unique_pids.tolist()}')
        if common is None:
            common = unique_pids
            reference_segments = segment_scores
        elif not np.array_equal(common, unique_pids):
            raise ValueError(f'{feature}: patient populations differ; missing={sorted(set(common)-set(unique_pids))}; extra={sorted(set(unique_pids)-set(common))}')
        elif not reference_segments.equals(segment_scores):
            raise ValueError(f'{feature}: segment population or depression scores differ')
        for target_patient in unique_pids:
            target_mask = groups_all == target_patient
            train_mask = ~target_mask
            assert target_mask.any() and train_mask.any()
            assert not (target_mask & train_mask).any()
            assert target_patient not in groups_all[train_mask]
            assert np.all(groups_all[target_mask] == target_patient)
        datasets[feature] = (X_all, y_all, groups_all, unique_pids,
                             df.groupby('patient_id')[TARGET].first().to_dict())
    print('PASS: all three feature sets have identical patient folds, segments and depression scores.')
    return datasets, common


def metrics(results, setting):
    y_true = results['true_score'].values.astype(float)
    y_pred = results['predicted_score'].values.astype(float)
    return dict(setting=setting, num_patients=len(results),
                patient_level_rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
                patient_level_mae=np.mean(np.abs(y_true - y_pred)))


def gpu_executor():
    available = Queue()
    for device in GPU_DEVICES:
        available.put(device)
    def initialize():
        _GPU_LOCAL.device = available.get_nowait()
    return concurrent.futures.ThreadPoolExecutor(max_workers=len(GPU_DEVICES), initializer=initialize)


def run_lopo(X_all, y_all, groups_all, unique_pids, scores, out, stage, params, save_best=False):
    lock = Lock()
    best = {'rmse': float('inf'), 'model': None, 'result': None}
    reported_device = set()

    def fold(target_patient):
        target_mask = groups_all == target_patient
        train_mask = ~target_mask
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_target_test = X_all[target_mask]
        y_target_test = y_all[target_mask]
        true_score = scores[target_patient]
        device = _GPU_LOCAL.device
        model = XGBRegressor(**dict(params, device=device))
        model.fit(X_train, y_train)
        target_predictions = model.predict(X_target_test)
        predicted_median = np.median(target_predictions)
        rmse = np.sqrt(mean_squared_error([true_score], [predicted_median]))
        if stage == 'all_feature' or (stage == 'full_feature' and CURRENT_TOP_K is not None):
            feature_importance = mean_abs_training_shap(model, X_train)
        else:
            # Diagnostic gain only; never used for Top-K feature selection.
            feature_importance = model.feature_importances_
        result = dict(patient_id=target_patient, true_score=true_score,
                      predicted_score=predicted_median, rmse=rmse,
                      training_samples=len(X_train), test_samples=len(X_target_test),
                      features_count=X_train.shape[1], status='success')
        with lock:
            if device not in reported_device:
                actual_device = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                if actual_device != device:
                    raise RuntimeError(f'GPU assignment failed: requested {device}, model used {actual_device}')
                reported_device.add(device)
                print(f'{out.name} {stage}: worker={device}; actual XGBoost device={actual_device}', flush=True)
                print(f'GPU_USAGE verified_training_devices={len(reported_device)}/{len(GPU_DEVICES)} devices={sorted(reported_device)}', flush=True)
            if save_best and (rmse < best['rmse'] or
                              (rmse == best['rmse'] and target_patient < best['result']['patient_id'])):
                best.update(rmse=rmse, model=model, result=result.copy())
            print(f'{out.name} {stage} patient={target_patient} RMSE={rmse:.6f}', flush=True)
        del model
        return result, feature_importance.copy(), target_predictions.copy()

    results, importances, predictions = {}, {}, {}
    # One active fold on this config process GPU; retain only the best model.
    # Tasks construct their masks on demand rather than storing every training matrix.
    with gpu_executor() as executor:
        futures = {executor.submit(fold, pid): pid for pid in unique_pids}
        for future in concurrent.futures.as_completed(futures):
            result, importance, segment_predictions = future.result()
            pid = result['patient_id']
            results[pid], importances[pid], predictions[pid] = result, importance, segment_predictions
    assert set(results) == set(unique_pids)
    print(f'{out.name} {stage}: GPUs actually used={len(reported_device)}; devices={sorted(reported_device)}', flush=True)
    result_df = pd.DataFrame([results[pid] for pid in unique_pids])
    result_df.to_csv(out / f'{stage}_patient_results.csv', index=False)
    # Retain every segment prediction, in original per-patient segment order.
    np.savez_compressed(out / f'{stage}_segment_predictions.npz', **predictions)
    importance_df = pd.DataFrame([
        dict(results[pid], **{f'feat_{i}': value for i, value in enumerate(importances[pid])})
        for pid in unique_pids])
    if save_best:
        model_path = out / 'best_topk_model.pkl'
        joblib.dump(best['model'], model_path)
        assert len(list(out.glob('*.pkl'))) == 1
        best['model'] = None
    gc.collect()
    return result_df, importance_df, best









def experiment(feature, top_k, data):
    setting = 'all' if top_k is None else f'top{top_k}'
    name = f'{feature}_{setting}'
    out = ROOT / name
    out.mkdir(exist_ok=True)
    assert out.resolve().is_relative_to(ROOT)
    X_all, y_all, groups_all, unique_pids, scores = data
    csv_path, column = CONFIGS[feature]
    config = dict(experiment_name=name, feature_type=feature, top_k=top_k,
                  csv_path=str(csv_path), feature_column=column, target_column=TARGET,
                  original_feature_count=X_all.shape[1], num_patients=len(unique_pids),
                  optuna_trials=0, fixed_registry_sha256=sha256(ROOT / 'gain_best_params_verified.json'), reference_sha256=sha256(REFERENCE),
                  dataset_sha256=sha256(csv_path), runner_sha256=sha256(__file__))
    config_path = out / 'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        old = json.loads(config_path.read_text())
        expected = dict(old, runner_sha256=config['runner_sha256'])
        if not RESUME_FROM_SHA or old['runner_sha256'] != RESUME_FROM_SHA or expected != config:
            raise ValueError(f'{name}: configuration or source differs from existing attempt')
        (out / f'config_before_gpu_migration_{RESUME_FROM_SHA[:12]}.json').write_text(config_path.read_text())
    config_path.write_text(json.dumps(config, indent=2))
    complete = out / 'experiment_complete.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        for filename, digest in record['artifacts_sha256'].items():
            if sha256(out / filename) != digest:
                raise ValueError(f'{name}: completed artifact changed: {filename}')
        print(f'{name}: already completed; source, config and artifact hashes match; reusing summary.', flush=True)
        return record['summary']
    print(f'\nSTART {name}: time={now()}; CSV={csv_path}; feature_column={column}; original_features={X_all.shape[1]}; setting={setting}; patients={len(unique_pids)}', flush=True)
    best_params = load_fixed_params(out.name)
    (out / "fixed_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
    params = dict(best_params, **PARAMS)  # Reference final models omit trial-only tree_method.
    full, importance, _ = run_lopo(X_all, y_all, groups_all, unique_pids, scores,
                                   out, 'all_feature' if top_k is None else 'full_feature', params)
    # xgb.py writes six-decimal importances and reads them back for Top-K selection.
    importance.to_csv(out / 'shap_importance_all_models.csv', index=False, float_format='%.6f')
    importance = pd.read_csv(out / 'shap_importance_all_models.csv')
    feature_cols = [c for c in importance.columns if c.startswith('feat_')]
    avg_importance = importance[feature_cols].mean().sort_values(ascending=False)
    pd.DataFrame({'feature_name': avg_importance.index,
                  'feature_index': [int(f.split('_')[1]) for f in avg_importance.index],
                  'average_importance': avg_importance.values,
                  'rank': np.arange(1, len(avg_importance) + 1)}).to_csv(out / 'shap_importance_summary.csv', index=False)
    final_metrics = [metrics(full, 'ALL_FEATURES')]
    best_fields = dict(best_fold_patient_id=None, best_fold_rmse=None, best_model_path=None)
    if top_k is None:
        X_selected = X_all
        final = full
    else:
        top_features = avg_importance.head(top_k).index.tolist()
        top_feature_indices = sorted([int(f.split('_')[1]) for f in top_features])
        X_selected = X_all[:, top_feature_indices]
        assert X_selected.shape[1] == top_k
        pd.DataFrame({'rank': range(1, top_k + 1), 'feature_name': top_features,
                      'feature_index': [int(f.split('_')[1]) for f in top_features],
                      'average_importance': avg_importance.head(top_k).values}).to_csv(out / f'selected_top{top_k}_features.csv', index=False)
        final, top_importance, best = run_lopo(X_selected, y_all, groups_all, unique_pids, scores,
                                               out, f'top{top_k}', params, save_best=True)
        top_importance.to_csv(out / f'top{top_k}_feature_importance_all_models.csv', index=False)
        comparison = full[['patient_id', 'rmse']].rename(columns={'rmse': 'rmse_all'}).merge(
            final[['patient_id', 'rmse']].rename(columns={'rmse': f'rmse_top{top_k}'}),
            on='patient_id', how='inner', validate='one_to_one')
        assert len(comparison) == len(unique_pids)
        comparison['delta_rmse'] = comparison[f'rmse_top{top_k}'] - comparison['rmse_all']
        comparison['improved'] = comparison['delta_rmse'] < 0
        comparison.to_csv(out / f'rmse_comparison_all_vs_top{top_k}_by_patient.csv', index=False)
        model_path = str(out / 'best_topk_model.pkl')
        info = dict(experiment_name=name, **best['result'], top_k=top_k,
                    selected_feature_indices=json.dumps(top_feature_indices),
                    selected_feature_names=json.dumps([f'feat_{i}' for i in top_feature_indices]),
                    model_path=model_path)
        pd.DataFrame([info]).to_csv(out / 'best_topk_model_info.csv', index=False)
        best_fields = dict(best_fold_patient_id=best['result']['patient_id'],
                           best_fold_rmse=best['rmse'], best_model_path=model_path)
        final_metrics.append(metrics(final, f'TOP_{top_k}_FEATURES'))
        print(f'{name}: selected_count={top_k}; selected_indices={top_feature_indices}; best_fold={best_fields}', flush=True)
    pd.DataFrame(final_metrics).to_csv(out / 'final_patient_level_rmse_mae.csv', index=False)
    row = dict(experiment_name=name, feature_type=feature, top_k=top_k,
               num_features_used=X_selected.shape[1], **metrics(final, setting), **best_fields,
               best_learning_rate=best_params["learning_rate"], best_max_depth=best_params["max_depth"],
               best_n_estimators=best_params["n_estimators"])
    print(f'END {name}: time={now()}; patient_level_RMSE={row["patient_level_rmse"]:.8f}; patient_level_MAE={row["patient_level_mae"]:.8f}', flush=True)
    record = dict(summary=row, artifacts_sha256={p.name: sha256(p) for p in out.iterdir()
                  if p.suffix in ('.csv', '.npz', '.pkl')})
    complete.write_text(json.dumps(record, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    return row


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()





