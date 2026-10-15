#!/usr/bin/env python3
"""EDAIC Optuna: reference optuna_xgb_is09_top15.py, twelve sequential experiments.

Run: python run_edaic_xgb.py
Validate without training: python run_edaic_xgb.py --validate-only
"""
import argparse
import fcntl
import optuna
import ast
import concurrent.futures
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from threading import Lock
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
MAX_WORKERS = 5
TARGET = 'depresyon_skoru'
CONFIGS = {
    'mfcc': (ROOT / 'metadata/metadata_with_mfcc.csv', 'features'),
    'egemaps': (ROOT / 'metadata/metadata_with_egemapsv0b1.csv', 'egemaps_features'),
    'is09': (ROOT / 'metadata/metadata_with_is09.csv', 'is09_features'),
}
ORDER = [(feature, k) for feature in CONFIGS for k in (5, 10, 15, None)]
TRIALS = 20
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
    """Verify the unchanged parser, patient-ID rule and extracted Optuna objective."""
    source = ast.parse(REFERENCE.read_text())
    current = ast.parse(Path(__file__).read_text())
    for name in ('parse_token', 'parse_features', 'extract_patient_id_from_filename'):
        old = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
        for function in (old, new):
            function.body = [n for n in function.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False), name
    original = ast.get_source_segment(REFERENCE.read_text(), next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'objective'))
    adapted = adapt_objective(original)
    expected = ast.parse(adapted).body[0]
    actual = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'reference_objective')
    assert ast.dump(expected, include_attributes=False) == ast.dump(actual, include_attributes=False), 'Optuna objective differs from documented adaptation'
    assert len(ORDER) == 12 and [k for _, k in ORDER[:4]] == [5, 10, 15, None]
    reference_trials = next(n for n in ast.walk(source) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'optimize')
    assert TRIALS == ast.literal_eval(next(k.value for k in reference_trials.keywords if k.arg == 'n_trials'))
    print('Reference verification passed: parser, ID extraction, complete adapted Optuna objective, search ranges and 20 trials.')


def adapt_objective(original):
    # Only interface, isolated paths, logging and requested Top-K/ALL changes.
    adapted = original.replace('def objective(trial):',
        'def reference_objective(trial, X_all, y_all, groups_all, unique_pids, patient_info, top_k, output):')
    adapted = adapted.replace('"OPTUNA_TMP_IMPORTANCE.csv"', 'output / "OPTUNA_TMP_IMPORTANCE.csv"')
    adapted = adapted.replace('avg_imp.head(15)', 'avg_imp.head(top_k)')
    adapted = adapted.replace('    # ------------------------------\n    # 2) SAVE TO TEMP CSV',
        '    if top_k is None:\n        return np.mean([r["rmse"] for r in LOO_results])\n\n    # ------------------------------\n    # 2) SAVE TO TEMP CSV')
    adapted = adapted.replace('        model = XGBRegressor(**params)',
        '        print(f"Optuna trial {trial.number}: held-out={pid}; features={X_train.shape[1]}", flush=True)\n        model = XGBRegressor(**params)')
    return adapted


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


def run_lopo(X_all, y_all, groups_all, unique_pids, scores, out, stage, params, save_best=False):
    lock = Lock()
    best = {'rmse': float('inf'), 'model': None, 'result': None}
    reported_device = []

    def fold(target_patient):
        target_mask = groups_all == target_patient
        train_mask = ~target_mask
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_target_test = X_all[target_mask]
        y_target_test = y_all[target_mask]
        true_score = scores[target_patient]
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        target_predictions = model.predict(X_target_test)
        predicted_median = np.median(target_predictions)
        rmse = np.sqrt(mean_squared_error([true_score], [predicted_median]))
        feature_importance = model.feature_importances_
        result = dict(patient_id=target_patient, true_score=true_score,
                      predicted_score=predicted_median, rmse=rmse,
                      training_samples=len(X_train), test_samples=len(X_target_test),
                      features_count=X_train.shape[1], status='success')
        with lock:
            if not reported_device:
                actual_device = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                reported_device.append(actual_device)
                print(f'{out.name} {stage}: actual XGBoost device={actual_device}', flush=True)
            if save_best and (rmse < best['rmse'] or
                              (rmse == best['rmse'] and target_patient < best['result']['patient_id'])):
                best.update(rmse=rmse, model=model, result=result.copy())
            print(f'{out.name} {stage} patient={target_patient} RMSE={rmse:.6f}', flush=True)
        del model
        return result, feature_importance.copy(), target_predictions.copy()

    results, importances, predictions = {}, {}, {}
    # Five active folds as in xgb.py; only the current best candidate is retained.
    # Tasks construct their masks on demand rather than storing every training matrix.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fold, pid): pid for pid in unique_pids}
        for future in concurrent.futures.as_completed(futures):
            result, importance, segment_predictions = future.result()
            pid = result['patient_id']
            results[pid], importances[pid], predictions[pid] = result, importance, segment_predictions
    assert set(results) == set(unique_pids)
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


def reference_objective(trial, X_all, y_all, groups_all, unique_pids, patient_info, top_k, output):

    params = {
        "objective": "reg:squarederror",

        # learning rate 1e-4 ile 0.3 arası (log-scale önerilir)
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.1, log=True
        ),

        # max_depth 4 ile 14 arası integer
        "max_depth": trial.suggest_int(
            "max_depth", 4, 14
        ),

        # n_estimators 100 ile 500 arası integer
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 500
        ),

        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # ------------------------------------
    # 1) FULL-FEATURE LOO (importance + RMSE)
    # ------------------------------------
    LOO_results = []


    for pid in unique_pids:

        mask = (groups_all == pid)

        X_train = X_all[~mask]
        y_train = y_all[~mask]
        X_test  = X_all[mask]

        print(f"Optuna trial {trial.number}: held-out={pid}; features={X_train.shape[1]}", flush=True)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        pred_med = np.median(preds)
        true = patient_info[pid]['depresyon_skoru']

        rmse = np.sqrt(mean_squared_error([true],[pred_med]))


        LOO_results.append({
            "pid":pid,
            "rmse":rmse,
            "importance":model.feature_importances_.copy()
        })

        del model
        gc.collect()


    if top_k is None:
        return np.mean([r["rmse"] for r in LOO_results])

    # ------------------------------
    # 2) SAVE TO TEMP CSV (exact same as main!)
    # ------------------------------
    imp_rows=[]
    for r in LOO_results:
        row={"pid":r["pid"],"rmse":r["rmse"]}
        for i,v in enumerate(r["importance"]):
            row[f"feat_{i}"]=v
        imp_rows.append(row)

    tmp_imp_df=pd.DataFrame(imp_rows)
    tmp_imp_df.to_csv(output / "OPTUNA_TMP_IMPORTANCE.csv", index=False)


    # ------------------------------
    # 3) RELOAD + compute avg imp (exact same as main!)
    # ------------------------------
    imp_df = pd.read_csv(output / "OPTUNA_TMP_IMPORTANCE.csv")

    feat_cols = [c for c in imp_df.columns if c.startswith("feat_")]

    avg_imp = imp_df[feat_cols].mean().sort_values(ascending=False)

    top_features = avg_imp.head(top_k).index.tolist()  # top feature değiştirme yeri
    top10_idx = sorted([int(f.split("_")[1]) for f in top_features])


    # ------------------------------6.6027
    # 4) TOP-10 FEATURE LOO
    # ------------------------------
    rmses=[]
    for pid in unique_pids:

        mask = (groups_all == pid)

        X_train = X_all[~mask][:,top10_idx]
        y_train = y_all[~mask]
        X_test  = X_all[mask][:,top10_idx]

        print(f"Optuna trial {trial.number}: held-out={pid}; features={X_train.shape[1]}", flush=True)
        model = XGBRegressor(**params)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        med = np.median(preds)

        true = patient_info[pid]['depresyon_skoru']
        rmse = np.sqrt(mean_squared_error([true],[med]))

        rmses.append(rmse)

        del model
        gc.collect()

    mean_rmse = np.mean(rmses)

    print(f"OPTUNA trial → mean = {mean_rmse:.4f}")

    return mean_rmse

def tune(data, top_k, out):
    X_all, y_all, groups_all, unique_pids, scores = data
    patient_info = {pid: {TARGET: score} for pid, score in scores.items()}
    study = optuna.create_study(direction="minimize", study_name=out.name,
                                storage=f"sqlite:///{out / 'optuna_study.sqlite3'}", load_if_exists=True)
    # Exclusive directory lock means remaining RUNNING trials belong to a stopped attempt.
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    completed = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    if completed > TRIALS:
        raise ValueError('Study contains more complete trials than the reference budget')
    def objective(trial):
        trial_dir = out / 'optuna_trials' / f'trial_{trial.number:03d}'
        trial_dir.mkdir(exist_ok=True, parents=True)
        print(f'{out.name}: Optuna trial {trial.number}; top_k={top_k}', flush=True)
        return reference_objective(trial, X_all, y_all, groups_all, unique_pids, patient_info, top_k, trial_dir)
    def record(study, trial):
        study.trials_dataframe().to_csv(out / 'optuna_trials.csv', index=False)
    if completed < TRIALS:
        study.optimize(objective, n_trials=TRIALS-completed, callbacks=[record])
    record(study, None)
    (out / 'best_params.json').write_text(json.dumps(study.best_params, indent=2))
    print(f'{out.name}: Optuna COMPLETE best_trial={study.best_trial.number}; mean_fold_RMSE={study.best_value}; params={study.best_params}', flush=True)
    return study.best_params


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
                  optuna_trials=TRIALS, reference_sha256=sha256(REFERENCE),
                  dataset_sha256=sha256(csv_path), runner_sha256=sha256(__file__))
    config_path = out / 'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f'{name}: configuration or source differs from existing attempt')
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
    best_params = tune(data, top_k, out)
    params = dict(best_params, **PARAMS)  # Reference final models omit trial-only tree_method.
    full, importance, _ = run_lopo(X_all, y_all, groups_all, unique_pids, scores,
                                   out, 'all_feature' if top_k is None else 'full_feature', params)
    # xgb.py writes six-decimal importances and reads them back for Top-K selection.
    importance.to_csv(out / 'feature_importance_all_models.csv', index=False, float_format='%.6f')
    importance = pd.read_csv(out / 'feature_importance_all_models.csv')
    feature_cols = [c for c in importance.columns if c.startswith('feat_')]
    avg_importance = importance[feature_cols].mean().sort_values(ascending=False)
    pd.DataFrame({'feature_name': avg_importance.index,
                  'feature_index': [int(f.split('_')[1]) for f in avg_importance.index],
                  'average_importance': avg_importance.values,
                  'rank': np.arange(1, len(avg_importance) + 1)}).to_csv(out / 'feature_importance_summary.csv', index=False)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    datasets, common = load_and_validate()
    if args.validate_only:
        print('Validation complete. No training or experiment outputs created.')
        return
    names = [f'{f}_{"all" if k is None else f"top{k}"}' for f, k in ORDER]
    lock_handle = (ROOT / '.run.lock').open('a')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another EDAIC runner is active in this directory')
    manifest_path = ROOT / 'run_manifest.json'
    identity = dict(script_sha256=sha256(__file__), reference_sha256=sha256(REFERENCE),
                    datasets={f: sha256(p) for f, (p, c) in CONFIGS.items()})
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('identity') != identity:
            raise ValueError('Existing run source/data identity differs; refusing to mix attempts')
    with (ROOT / 'sequential_run.log').open('a') as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            print(f'Run start={now()}; output_root={ROOT}; parameters={PARAMS}; fold_workers={MAX_WORKERS}')
            for feature, data in datasets.items():
                print(f'{feature} sorted patient IDs: {data[3].tolist()}')
            pd.DataFrame({'fold_index': np.arange(len(common)), 'patient_id': common}).to_csv(ROOT / 'patient_fold_index.csv', index=False)
            source_commit_path = ROOT / 'SOURCE_COMMIT'
            manifest = dict(start_time=now(), source_commit=source_commit_path.read_text().strip() if source_commit_path.exists() else None,
                            parent_commit=(ROOT / 'SOURCE_PARENT_COMMIT').read_text().strip(),
                            source_status='SOURCE_COMMIT when deployed from published source; SHA256 identifies exact files',
                            script_sha256=sha256(__file__), reference_sha256=sha256(REFERENCE), identity=identity,
                            parameters=PARAMS, python=sys.version, xgboost=xgboost.__version__,
                            numpy=np.__version__, pandas=pd.__version__, optuna=optuna.__version__,
                            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                            optuna_trials=TRIALS, optuna_objective='mean per-patient fold RMSE',
                            target_column=TARGET, patient_id_source="basename(file_name).split('-')[0].lower()",
                            feature_names='No semantic feature names supplied; feat_N identifies original vector index.',
                            experiment_order=names,
                            datasets={f: dict(csv_path=str(p), feature_column=c, sha256=sha256(p),
                                              dimension=datasets[f][0].shape[1], patients=len(common))
                                      for f, (p, c) in CONFIGS.items()})
            (ROOT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
            journal(f'EDAIC Optuna sıralı koşusu başladı. 12 config; hedef={TARGET}; '
                    f'ortak hasta sayısı={len(common)}. LOPO, feature_importances_, '
                    f'segment tahminlerinin medyanı ve hasta düzeyinde RMSE/MAE. '
                    f'Kaynak commit yok; SHA256 ve veri/config kimlikleri: {ROOT / "run_manifest.json"}. '
                    f'Çıktı kökü={ROOT}; log=sequential_run.log; Slurm job={os.environ.get("SLURM_JOB_ID", "yok")}.')
            summary = []
            for feature, top_k in ORDER:
                summary.append(experiment(feature, top_k, datasets[feature]))
                pd.DataFrame(summary).to_csv(ROOT / 'experiment_summary.csv', index=False)
                row = summary[-1]
                journal(f'{row["experiment_name"]} tüm foldlar tamamlandı; '
                        f'hasta düzeyi RMSE={row["patient_level_rmse"]:.8f}, '
                        f'MAE={row["patient_level_mae"]:.8f}. '
                        f'Kanıt={ROOT / row["experiment_name"] / "final_patient_level_rmse_mae.csv"}; '
                        f'tamamlanan config={len(summary)}/12. Top-K model kaydı yalnızca artifact.')
            assert len(summary) == 12
            print(f'COMPLETE all 12 experiments: {now()}')
            journal('12/12 deney tamamlandı; experiment_summary.csv ve patient_fold_index.csv yazıldı. '
                    'Koşu çıktıları üretildi; bağımsız gerçek-sonuç denetimi ayrıca yapılabilir.')
        except Exception:
            import traceback
            traceback.print_exc()
            journal('Sıralı koşu hata ile durdu; ayrıntı sequential_run.log içinde. '
                    'Tamamlanmamış configler tamamlandı olarak değerlendirilmemeli.')
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == '__main__':
    main()
