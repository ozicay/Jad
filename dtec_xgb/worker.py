"""D3TEC adaptation of the root optuna_xgb_is09_top15.py scientific flow.

LOPO folds run serially: parallelism is exclusively between configurations.
No pooled-anxiety scientific code is imported or copied.
"""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost

from data import (CONFIGURATIONS, DEFAULT_DATA, ROOT, REFERENCE_SHA256, FAMILIES,
                  load_family, sha256, write_json)

TRIALS = 40  # User explicitly changed the reference's 20 trials to 40.
LOG = logging.getLogger(__name__)


def parameters(trial, device, threads):
    # Same suggestions, bounds, log scale, model seed and hist as reference objective.
    return dict(objective='reg:squarederror',
                learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                max_depth=trial.suggest_int('max_depth', 4, 14),
                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                tree_method='hist', device=device, random_state=42, n_jobs=threads)


def verify_booster_device(model, expected):
    config = json.loads(model.get_booster().save_config())
    actual = config['learner']['generic_param']['device']
    if expected.startswith('cuda') and actual != 'cuda:0':
        raise RuntimeError(f'GPU fallback prohibited: expected cuda:0, got {actual}')
    if expected == 'cpu' and actual != 'cpu':
        raise RuntimeError(f'Expected CPU smoke model, got {actual}')
    return actual


def lopo(X, y, groups, params, directory, stage, save_best=False, feature_indices=None):
    """Original patient split, fit/predict, median, singleton RMSE and importance."""
    rows, importance_rows, segments = [], [], []
    best_model, best_row = None, None
    unique_pids = np.unique(groups)
    for pid in unique_pids:
        mask = groups == pid
        X_train, y_train, X_test = X[~mask], y[~mask], X[mask]
        true = y[mask][0]
        print(f'    🎯 LOO Training {pid} (DTEC SM-27) [{stage}]', flush=True)
        print(f'      📊 Training: {len(X_train)} samples with {X_train.shape[1]} features; Testing: {len(X_test)} samples', flush=True)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        actual_device = verify_booster_device(model, params['device'])
        preds = model.predict(X_test)
        med = np.median(preds)
        rmse = np.sqrt(mean_squared_error([true], [med]))
        # Scalar conversion is only for output. Original median is retained for error calculation.
        row = dict(patient_id=pid, true_score=float(true), predicted_score=float(med),
                   rmse=float(rmse), AE=float(abs(float(true) - float(med))),
                   training_samples=len(X_train), test_samples=len(X_test),
                   features_count=X.shape[1], target_dataset='d3tec_sm27')
        rows.append(row)
        importance_rows.append(dict(patient_id=pid, rmse=rmse,
                                    **{f'feat_{i}': v for i, v in enumerate(model.feature_importances_.copy())}))
        for local_index, prediction in zip(np.flatnonzero(mask), preds):
            segments.append(dict(selected_row=int(local_index), patient_id=pid, prediction=float(prediction)))
        if save_best and (best_row is None or row['AE'] < best_row['AE']):
            # Keep one current candidate in memory, persist only the final minimum-AE fold.
            best_model, best_row = model, row.copy()
        print(f'      ✅ {pid}: True={true:.3f}, Pred={med:.3f}, RMSE={rmse:.3f}', flush=True)
        LOG.info('%s patient=%s true=%s predicted=%s AE=%s train=%d test=%d device=%s',
                 stage, pid, true, med, row['AE'], len(X_train), len(X_test), actual_device)
        del model
        gc.collect()
    predictions = pd.DataFrame(rows)
    predictions.to_csv(directory / f'{stage}_patient_predictions.csv', index=False)
    pd.DataFrame(segments).to_csv(directory / f'{stage}_segment_predictions.csv', index=False)
    return predictions, pd.DataFrame(importance_rows), best_model, best_row


def select_features(importance, k, path, final=False):
    # Reference trial: default CSV precision. Reference final: float_format='%.6f'.
    importance.to_csv(path, index=False, float_format='%.6f' if final else None)
    imp_df = pd.read_csv(path)
    feat_cols = [c for c in imp_df.columns if c.startswith('feat_')]
    avg_imp = imp_df[feat_cols].mean().sort_values(ascending=False)
    top_features = avg_imp.head(k).index.tolist()
    indices = sorted([int(f.split('_')[1]) for f in top_features])
    return indices, avg_imp


def objective(trial, X, y, groups, k, directory, device='cuda:0', threads=20, smoke=False):
    print(f'\n🚀 OPTUNA trial {trial.number}: FULL-FEATURE LOO', flush=True)
    params = parameters(trial, device, threads)
    if smoke:
        # Only a synthetic, separately labelled smoke attempt may override model cost.
        params.update(n_estimators=3, max_depth=2)
    trial_dir = directory / f'trial_{trial.number:03d}'
    trial_dir.mkdir()
    full, importance, _, _ = lopo(X, y, groups, params, trial_dir, 'full')
    if k is None:
        # User-requested All extension: one full-feature LOPO, no second LOPO/retraining.
        value = np.mean(full['rmse'].values)
        indices = list(range(X.shape[1]))
    else:
        indices, _ = select_features(importance.rename(columns={'patient_id': 'pid'}), k,
                                     trial_dir / 'OPTUNA_TMP_IMPORTANCE.csv')
        print(f'  🔝 TOP-{k} FEATURE LOO; indices={indices}', flush=True)
        reduced, _, _, _ = lopo(X[:, indices], y, groups, params, trial_dir, 'topk')
        value = np.mean(reduced['rmse'].values)
    write_json(trial_dir / 'trial_info.json', dict(number=trial.number, objective=float(value),
               params=params, selected_feature_indices=indices))
    print(f'OPTUNA trial → mean = {value:.4f}', flush=True)
    return value


def final_evaluation(X, y, groups, k, best_params, directory, device, threads,
                     family, manifest, keys, smoke=False):
    # Reference final model has best_params + objective/random_state/device;
    # do not add Optuna's tree_method here (reference relies on the XGBoost default).
    params = dict(best_params, objective='reg:squarederror', random_state=42,
                  device=device, n_jobs=threads)
    if smoke:
        params.update(n_estimators=3, max_depth=2)
    print('\n🚀 STARTING STANDARD LOO TRAINING WITH SIMPLE FEATURE IMPORTANCE...', flush=True)
    full, importance, model, best = lopo(X, y, groups, params, directory, 'full', save_best=k is None)
    if k is None:
        importance.to_csv(directory / 'feature_importance_all_models.csv', index=False, float_format='%.6f')
        indices = list(range(X.shape[1]))
        predictions = full
        avg_imp = None
    else:
        indices, avg_imp = select_features(importance, k, directory / 'feature_importance_all_models.csv', final=True)
        print(f'\n🚀 RETRAIN: TOP-{k} feature ile LOO başlıyor; indices={indices}', flush=True)
        predictions, _, model, best = lopo(X[:, indices], y, groups, params, directory, 'topk', save_best=True)
    full_true = full['true_score'].values.astype(float)
    full_pred = full['predicted_score'].values.astype(float)
    pd.DataFrame([dict(setting='ALL_FEATURES', patients=len(full),
                       patient_level_rmse=float(np.sqrt(mean_squared_error(full_true, full_pred))),
                       patient_level_mae=float(np.mean(np.abs(full_true-full_pred))))]).to_csv(
                           directory / 'full_feature_metrics.csv', index=False)
    if len(predictions) != len(np.unique(groups)) or model is None:
        raise RuntimeError('Incomplete final LOPO')
    # Exact reference final metric formulas, using unrounded patient predictions.
    y_true = predictions['true_score'].values.astype(float)
    y_pred = predictions['predicted_score'].values.astype(float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    predictions.to_csv(directory / 'patient_predictions.csv', index=False)
    pd.DataFrame([dict(model_input_position=position, feature_index=i, feature_name=f'feat_{i}', family=family,
                       importance_rank=(list(avg_imp.index).index(f'feat_{i}') + 1) if avg_imp is not None else None)
                  for position, i in enumerate(indices)]).to_csv(directory / 'selected_features.csv', index=False)
    if avg_imp is not None:
        pd.DataFrame([dict(rank=rank, feature_name=name, mean_importance=float(value))
                      for rank, (name, value) in enumerate(avg_imp.items(), 1)]).to_csv(
                          directory / 'feature_gain_ranking.csv', index=False)
    joblib.dump(model, directory / 'best_model.pkl')
    selected_csv = directory / 'selected_11_chunks.csv'
    best_info = dict(best, selection_rule='minimum held-out AE during final LOPO; ties use sorted patient ID',
                     limitation='Not a model selected for best generalization across patients',
                     feature_indices=indices, feature_names=[f'feat_{i}' for i in indices],
                     feature_name_source='positional feat_i identifiers from reference; acoustic names not in vector CSV',
                     family=family, feature_column=FAMILIES[family][1], best_params=best_params,
                     effective_model_params=params, source_metadata=manifest['input_sha256'],
                     selection_sha256=sha256(selected_csv), held_out_chunk_ids=[f'{pid}_chunk_{i:02d}' for pid, i in keys if pid == best['patient_id']],
                     training_patients=[pid for pid in np.unique(groups) if pid != best['patient_id']],
                     reference_sha256=REFERENCE_SHA256, source_commit=os.environ.get('DTEC_SOURCE_COMMIT'))
    write_json(directory / 'best_model_info.json', best_info)
    metrics = dict(feature_family=family, top_k='All' if k is None else k,
                   patients=len(predictions), segments=len(y), patient_level_rmse=float(rmse),
                   patient_level_mae=float(mae), best_model_patient=best['patient_id'], best_model_AE=best['AE'],
                   selection_sha256=sha256(selected_csv), smoke=smoke)
    pd.DataFrame([metrics]).to_csv(directory / 'final_metrics.csv', index=False)
    print(f'\n📊 FINAL PATIENT-LEVEL METRICS: Patients={len(predictions)} RMSE={rmse:.4f} MAE={mae:.4f}', flush=True)
    print(f'💾 Minimum-AE fold model: {best["patient_id"]}, AE={best["AE"]:.6f}', flush=True)
    return metrics


def gpu_assignment():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not os.environ.get('SLURM_JOB_ID') or len([v for v in visible.split(',') if v.strip()]) != 1:
        raise RuntimeError(f'Worker requires one Slurm-assigned visible GPU, got {visible!r}')
    return dict(cuda_visible_devices=visible, xgboost_device='cuda:0',
                slurm_job_id=os.environ['SLURM_JOB_ID'], slurm_step_id=os.environ.get('SLURM_STEP_ID'))


def run(name, data_dir, selection_dir, run_dir, threads):
    assignments = gpu_assignment()
    _, family, k = next(c for c in CONFIGURATIONS if c[0] == name)
    X, y, groups, keys, manifest = load_family(data_dir, selection_dir, family)
    directory = Path(run_dir) / name
    directory.mkdir(exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(directory / 'execution.log'), logging.StreamHandler()])
    write_json(directory / 'gpu_assignment.json', assignments)
    print(f'🚀 DTEC {name}: Using cuda:0; CUDA_VISIBLE_DEVICES={assignments["cuda_visible_devices"]}', flush=True)
    print(f'📊 FINAL DATASET: {len(np.unique(groups))} patients, {len(y)} samples, {X.shape[1]} features', flush=True)
    print('🚀 OPTUNA hyperparameter tuning starting...', flush=True)
    shutil.copyfile(Path(selection_dir) / 'selected_11_chunks.csv', directory / 'selected_11_chunks.csv')
    if sha256(directory / 'selected_11_chunks.csv') != manifest['selection_sha256']:
        raise RuntimeError('Worker selection copy differs')
    write_json(directory / 'experiment_config.json', dict(name=name, family=family, top_k=k,
               trials=TRIALS, model_seed=42, optuna_sampler='default, unseeded (reference)',
               selection_seed=42, threads=threads, target='depresyon_skoru', aggregation='median',
               data_directory=str(data_dir), selection=manifest, gpu=assignments,
               reference_sha256=REFERENCE_SHA256,
               source_sha256={p.name: sha256(p) for p in ROOT.glob('*.py')},
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             xgboost=xgboost.__version__, optuna=optuna.__version__)))
    study = optuna.create_study(direction='minimize', storage=f'sqlite:///{directory / "optuna_study.sqlite3"}',
                                study_name=name)
    # Default sampler is intentionally unseeded, as in reference; only sampling chunks uses a new seed.
    try:
        study.optimize(lambda trial: objective(trial, X, y, groups, k, directory,
                                               device='cuda:0', threads=threads), n_trials=TRIALS)
    finally:
        study.trials_dataframe().to_csv(directory / 'optuna_trials.csv', index=False)
    write_json(directory / 'best_params.json', dict(params=study.best_params, value=study.best_value,
                                                  trial_number=study.best_trial.number))
    print('Best params:', study.best_params, flush=True)
    print(f'🏆 BEST OPTUNA RMSE: {study.best_value:.4f}', flush=True)
    metrics = final_evaluation(X, y, groups, k, study.best_params, directory, 'cuda:0', threads,
                               family, manifest, keys)
    write_json(directory / 'COMPLETE.json', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=[c[0] for c in CONFIGURATIONS], required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--selection-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=20)
    args = parser.parse_args()
    run(args.config, args.data_dir.resolve(), args.selection_dir.resolve(), args.run_dir.resolve(), args.threads)
