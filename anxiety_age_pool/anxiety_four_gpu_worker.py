"""Table-4 pooled anxiety age_group runner. --validate-only performs no model fitting."""
from pathlib import Path
import argparse
import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import re
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs'
REFERENCE = ROOT / 'geriatri_job/geriatri_nongeri_pool_egemaps_top15.py'
SELECTED = ROOT / 'geriatri_job/xgb_codes/depr_egemaps_pool_top15_60trial_204/selected_102_nongeriatric_patients.csv'
TRIALS = 40
MAX_WORKERS = 5
DEVICE = 'cuda'
TARGET = 'anksiyete_skoru'
FAMILIES = {
    'mfcc': ('metadata_with_mfcc_anxiety.csv', 'metadata_w2v2_scores_mfcc_depression.csv', 'metadata_w2v2_scores_mfcc_comorbid.csv', 'mfcc_features', 'features', 608),
    'egemaps': ('metadata_with_egemapsv01bb.csv', 'metadata_with_egemaps.csv', 'metadata_with_egemaps.csv', 'egemaps_features', 'egemaps_features', 88),
    'is09': ('metadata_with_is09.csv', 'metadata_with_is09.csv', 'metadata_with_is09.csv', 'is09_features', 'is09_features', 384),
}
CONFIGURATIONS = [(f'{family}_{"all" if k is None else f"top{k}"}', family, k)
                  for family in FAMILIES for k in (5, 10, 15, None)]
num_re = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+(?:\.\d+)?)?')


def progress(context, message, *args):
    logging.info('[%s] ' + message, context, *args)


def report_selection(context, avg_imp, selected_indices, age_index, top_k):
    progress(context, 'AVERAGED GAIN IMPORTANCE (%d full candidate features)', len(avg_imp))
    for rank, (feature, importance) in enumerate(avg_imp.items(), 1):
        progress(context, '  Gain rank %d: %s mean_gain=%.10f', rank, feature, importance)
    progress(context, '%s: %d selected features; numeric model input indices=%s',
             'ALL FEATURES' if top_k is None else f'TOP-{top_k}', len(selected_indices), selected_indices)
    age_feature = f'feat_{age_index}'
    progress(context, 'age_group index=%d mean_gain=%.10f rank=%d/%d selected=%s',
             age_index, avg_imp[age_feature], avg_imp.index.get_loc(age_feature)+1,
             len(avg_imp), age_index in selected_indices)


def report_metadata(context, config, groups, info):
    for key in ('geriatric_metadata', 'nongeriatric_depression_metadata', 'nongeriatric_comorbid_metadata'):
        progress(context, 'Metadata loaded during preflight: %s=%s', key, config[key])
    progress(context, 'Exact non-geriatric patient selection: %s', config['selected_102_patient_csv'])
    for dataset in ('nongeriatric_depr_only', 'nongeriatric_depr_anx', 'geriatric'):
        ids = {pid for pid, value in info.items() if value['dataset'] == dataset}
        progress(context, 'Dataset %s: patients=%d segments=%d', dataset, len(ids), int(np.isin(groups, list(ids)).sum()))
    progress(context, 'POOLED: geriatric=%d nongeriatric=%d total_patients=%d total_segments=%d',
             config['num_geriatric_patients'], config['num_nongeriatric_patients'], config['total_patients'], config['total_segments'])
    progress(context, 'Feature dimensions: acoustic=%d augmented=%d age_group_index=%d; geriatric=1 nongeriatric=0',
             config['original_acoustic_feature_count'], config['augmented_full_feature_count'], config['age_group_feature_index'])


def parse_token(tok):
    if 'e' in tok or 'E' in tok:
        mant, exp = re.split('[eE]', tok)
        return float(mant) * (10 ** float(exp))
    return float(tok)


def parse_features(s):
    toks = re.split(r'[,\s]+', s)
    return np.array([parse_token(t) for t in toks if t and num_re.fullmatch(t)], dtype=float)


def extract_patient_id_from_filename(filename):
    return os.path.basename(str(filename)).split('-')[0].lower()


def load_family(family):
    geri, depr, anx, geri_col, nong_col, dimension = FAMILIES[family]
    paths = [ROOT / 'non_geriatri_metadata/depression' / depr,
             ROOT / 'non_geriatri_metadata/comorbid' / anx,
             ROOT / 'geriatri_new/geriatri_metadata' / geri]
    started = time.perf_counter()
    progress(family, 'Loading metadata; selected patient CSV=%s', SELECTED)
    selection = pd.read_csv(SELECTED)
    if len(selection) != 102 or selection.patient_id.nunique() != 102:
        raise ValueError('Selection CSV must contain exactly 102 unique patients')
    selected = set(selection.patient_id)
    frames = []
    for path, col, prefix, dataset, age in zip(
        paths, [nong_col, nong_col, geri_col], ['nong_depr_', 'nong_anx_', 'geri_'],
        ['nongeriatric_depr_only', 'nongeriatric_depr_anx', 'geriatric'], [0, 0, 1]
    ):
        progress(family, 'Loading %s: %s; feature_column=%s target_column=%s', dataset, path, col, TARGET)
        d = pd.read_csv(path, **({'index_col': 0, 'encoding': 'utf-8'} if age == 0 else {'encoding': 'utf-8-sig'}))
        if col not in d or TARGET not in d or (age == 1 and 'file_name' not in d):
            raise ValueError(f'Required columns missing: {path}')
        names = d.index if age == 0 else d.file_name
        d['patient_id'] = [prefix + extract_patient_id_from_filename(s) for s in names]
        d['dataset'], d['age_group'] = dataset, age
        d['cohort'] = 'geriatric' if age else 'nongeriatric'
        progress(family, 'Loaded %s: segments=%d patients=%d', dataset, len(d), d.patient_id.nunique())
        if age == 0:
            d = d[d.patient_id.isin(selected)].copy()
            progress(family, 'After exact patient filtering: segments=%d patients=%d', len(d), d.patient_id.nunique())
        d['parsed_features'] = d[col].apply(parse_features)
        if not d.parsed_features.map(len).eq(dimension).all():
            raise ValueError(f'{path}: expected {dimension} acoustic features; found {d.parsed_features.map(len).value_counts().to_dict()}')
        progress(family, 'Verified %s: acoustic dimension=%d age_group=%d', dataset, dimension, age)
        frames.append(d)
    df = pd.concat(frames, axis=0, ignore_index=True)
    actual = set(df.loc[df.age_group.eq(0), 'patient_id'])
    if actual != selected:
        raise ValueError(f'{family}: selected patient mismatch; missing={sorted(selected-actual)}, extra={sorted(actual-selected)}')
    # Validate selected CSV identity and targets without altering segment inclusion.
    first = df[df.age_group.eq(0)].groupby('patient_id').first()
    for row in selection.itertuples():
        if first.loc[row.patient_id, 'dataset'] != row.dataset or float(first.loc[row.patient_id, 'depresyon_skoru']) != float(getattr(row, 'depresyon_skoru')):
            raise ValueError(f'{family}: selection CSV dataset/target mismatch')
    if not df.groupby('patient_id').age_group.nunique().eq(1).all():
        raise ValueError('Patient has more than one age_group')
    if not df.loc[df.cohort.eq('geriatric'), 'age_group'].eq(1).all() or not df.loc[df.cohort.eq('nongeriatric'), 'age_group'].eq(0).all():
        raise ValueError('Incorrect age_group coding')
    X_all = np.column_stack([np.vstack(df.parsed_features.values), df.age_group.to_numpy()])
    y_all = df[TARGET].astype(float).to_numpy()
    if not np.isfinite(X_all).all() or not np.isfinite(y_all).all():
        raise ValueError('Non-finite features/targets')
    groups = df.patient_id.to_numpy()
    info = df.groupby('patient_id').agg({TARGET: 'first', 'dataset': 'first', 'cohort': 'first', 'age_group': 'first'}).to_dict('index')
    counts = dict(num_geriatric_patients=frames[2].patient_id.nunique(), num_nongeriatric_patients=len(actual), total_patients=len(info),
                  num_geriatric_segments=len(frames[2]), num_nongeriatric_segments=len(frames[0])+len(frames[1]), total_segments=len(df))
    config = dict(geriatric_metadata=str(paths[2]), nongeriatric_depression_metadata=str(paths[0]), nongeriatric_comorbid_metadata=str(paths[1]),
                  selected_102_patient_csv=str(SELECTED), feature_column={'geriatric': geri_col, 'nongeriatric': nong_col}, target_column=TARGET,
                  original_acoustic_feature_count=dimension, augmented_full_feature_count=X_all.shape[1], age_group_feature_index=dimension,
                  age_group_definition={'geriatric': 1, 'nongeriatric': 0}, random_state=42, device=DEVICE, max_workers=MAX_WORKERS,
                  reference_code=str(REFERENCE), reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_commit=os.environ.get('ANXIETY_SOURCE_COMMIT'),
                  metadata_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  selected_patient_csv_sha256=hashlib.sha256(SELECTED.read_bytes()).hexdigest(), **counts)
    report_metadata(family, config, groups, info)
    progress(family, 'Metadata validation finished in %.2f seconds', time.perf_counter()-started)
    return X_all, y_all, groups, info, config


def select_indices(avg_imp, top_k):
    if top_k is None:
        return list(range(len(avg_imp)))
    top_features = avg_imp.head(top_k).index.tolist()
    return sorted([int(f.split('_')[1]) for f in top_features])


def lopo(X, y, groups, info, params, *, parallel=False, retain_best=False, context='LOPO'):
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    started = time.perf_counter()
    total = len(np.unique(groups))
    progress(context, 'LOPO starting: folds=%d features=%d workers=%d', total, X.shape[1], MAX_WORKERS if parallel else 1)
    def fold(pid):
        fold_started = time.perf_counter()
        mask = groups == pid
        progress(context, 'Training held-out patient %s (%s): train_segments=%d test_segments=%d features=%d',
                 pid, info[pid]['dataset'], int((~mask).sum()), int(mask.sum()), X.shape[1])
        model = XGBRegressor(**params)
        model.fit(X[~mask], y[~mask])
        prediction = float(np.median(model.predict(X[mask])))
        true = float(info[pid][TARGET])
        row = dict(patient_id=pid, cohort=info[pid]['cohort'], age_group=int(info[pid]['age_group']), target_dataset=info[pid]['dataset'],
                   true_score=true, predicted_score=prediction, absolute_error=abs(true-prediction), squared_error=(true-prediction)**2,
                   rmse=float(np.sqrt(mean_squared_error([true], [prediction]))), training_samples=int((~mask).sum()),
                   test_samples=int(mask.sum()), selected_feature_count=X.shape[1])
        progress(context, 'Prediction %s: true=%.6f median_prediction=%.6f AE=%.6f squared_error=%.6f fold_RMSE=%.6f elapsed=%.2fs',
                 pid, true, prediction, row['absolute_error'], row['squared_error'], row['rmse'], time.perf_counter()-fold_started)
        importance = model.feature_importances_.copy()
        return row, importance, model if retain_best else None
    rows, importance_rows = [], []
    best_model, best_row = None, None
    def consume(result):
        nonlocal best_model, best_row
        row, importance, model = result
        rows.append(row)
        importance_rows.append(dict(patient_id=row['patient_id'], rmse=row['rmse'], **{f'feat_{i}': v for i, v in enumerate(importance)}))
        if retain_best and (best_row is None or (row['absolute_error'], row['patient_id']) < (best_row['absolute_error'], best_row['patient_id'])):
            best_model, best_row = model, row
        progress(context, 'LOPO progress %d/%d: completed %s; elapsed=%.2fs', len(rows), total, row['patient_id'], time.perf_counter()-started)
    if parallel:
        # Bounded pending futures avoid retaining every fitted model in memory.
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            pids = iter(np.unique(groups))
            pending = {executor.submit(fold, pid) for pid in [next(pids, None) for _ in range(MAX_WORKERS)] if pid is not None}
            while pending:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
                    pid = next(pids, None)
                    if pid is not None:
                        pending.add(executor.submit(fold, pid))
                del done, future
    else:
        for pid in np.unique(groups):
            consume(fold(pid))
            gc.collect()
    if len(rows) != len(np.unique(groups)):
        raise RuntimeError('Incomplete LOPO: refusing partial metrics')
    progress(context, 'LOPO finished: %d/%d patients; mean fold RMSE=%.8f elapsed=%.2fs', len(rows), total, np.mean([row['rmse'] for row in rows]), time.perf_counter()-started)
    return pd.DataFrame(rows), pd.DataFrame(importance_rows), best_model, best_row


def averaged_importance(path):
    d = pd.read_csv(path)
    return d[[c for c in d if c.startswith('feat_')]].mean().sort_values(ascending=False)


def run_experiment(name, family, top_k, data):
    import optuna
    import joblib
    from sklearn.metrics import mean_squared_error
    X, y, groups, info, base_config = data
    directory = OUTPUT / name
    if (directory / 'final_metrics.csv').exists():
        raise FileExistsError(f'Completed configuration must not be overwritten: {directory}')
    started = time.perf_counter()
    handler = logging.FileHandler(directory / 'run.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger('optuna').addHandler(handler)
    try:
        progress(name, 'EXPERIMENT START: top_k=%s trials=%d device=%s max_workers=%d output=%s', top_k, TRIALS, DEVICE, MAX_WORKERS, directory)
        report_metadata(name, base_config, groups, info)
        def objective(trial):
            trial_started = time.perf_counter()
            trial_context = f'{name} trial {trial.number+1}/{TRIALS}'
            params = {'objective': 'reg:squarederror',
                      'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                      'max_depth': trial.suggest_int('max_depth', 4, 14),
                      'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                      'tree_method': 'hist', 'device': DEVICE, 'random_state': 42}
            progress(trial_context, 'OPTUNA TRIAL START params=%s', params)
            _, importance, _, _ = lopo(X, y, groups, info, params, context=f'{trial_context} full-feature importance')
            temporary = directory / 'OPTUNA_TMP_IMPORTANCE.csv'
            importance.to_csv(temporary, index=False)
            avg_imp = averaged_importance(temporary)
            selected_indices = select_indices(avg_imp, top_k)
            report_selection(trial_context, avg_imp, selected_indices, X.shape[1]-1, top_k)
            progress(trial_context, 'RETRAINING selected feature matrix shape=(%d, %d)', len(X), len(selected_indices))
            predictions, _, _, _ = lopo(X[:, selected_indices], y, groups, info, params, context=f'{trial_context} selected-feature evaluation')
            value = float(predictions.rmse.mean())
            progress(trial_context, 'OPTUNA TRIAL FINISHED objective mean fold RMSE=%.8f elapsed=%.2fs', value, time.perf_counter()-trial_started)
            return value
        # Reference uses the default sampler, with no explicit sampler seed/pruning.
        from anxiety_gpu_controller import prepare_study
        study = prepare_study(directory, name)
        remaining_trials = TRIALS - len(study.trials)
        progress(name, 'Preserved completed trials=%d; remaining=%d; visible GPU=%s',
                 len(study.trials), remaining_trials, os.environ.get('CUDA_VISIBLE_DEVICES'))
        study.optimize(objective, n_trials=remaining_trials,
                       callbacks=[lambda study, trial: study.trials_dataframe().to_csv(directory / 'optuna_trials.csv', index=False)])
        best_params = study.best_params
        progress(name, 'OPTUNA COMPLETE: best_trial=%d best_objective=%.8f best_params=%s elapsed=%.2fs',
                 study.best_trial.number+1, study.best_value, best_params, time.perf_counter()-started)
        (directory / 'best_params.json').write_text(json.dumps(best_params, indent=2)+'\n')
        # Reference final models omit tree_method: preserve that default.
        final_params = dict(best_params, objective='reg:squarederror', random_state=42, device=DEVICE)
        full_predictions, importance, _, _ = lopo(X, y, groups, info, final_params, parallel=True, context=f'{name} best-parameter full-feature LOPO')
        full_predictions.to_csv(directory / 'full_feature_patient_predictions.csv', index=False)
        importance_path = directory / 'feature_importance_all_models.csv'
        # Preserve reference final-stage CSV roundtrip at six decimal places.
        importance.to_csv(importance_path, index=False, float_format='%.6f')
        avg_imp = averaged_importance(importance_path)
        selected_indices = select_indices(avg_imp, top_k)
        report_selection(name, avg_imp, selected_indices, X.shape[1]-1, top_k)
        ranking = pd.DataFrame([dict(rank_by_gain=rank, feature_name=feature, feature_index=int(feature.split('_')[1]),
                                    mean_gain_importance=float(value), is_age_group=int(feature.split('_')[1]) == X.shape[1]-1)
                                for rank, (feature, value) in enumerate(avg_imp.items(), 1)])
        ranking.to_csv(directory / 'feature_gain_ranking.csv', index=False)
        selected = ranking[ranking.feature_index.isin(selected_indices)].copy()
        selected.to_csv(directory / 'selected_feature_indices.csv', index=False)
        inputs = selected.sort_values('feature_index').rename(columns={'rank_by_gain': 'gain_rank'}).reset_index(drop=True)
        inputs.insert(0, 'model_input_position', range(len(inputs)))
        inputs.to_csv(directory / 'selected_features.csv', index=False)
        age = ranking[ranking.is_age_group].iloc[0]
        age_selected = X.shape[1]-1 in selected_indices
        pd.DataFrame([dict(age_group_feature_index=X.shape[1]-1, mean_gain_importance=age.mean_gain_importance,
                           age_group_rank=int(age.rank_by_gain), age_group_selected=age_selected)]).to_csv(directory / 'age_group_feature_report.csv', index=False)
        progress(name, 'FINAL RETRAINING START: fixed global selected indices=%s matrix_shape=(%d, %d)', selected_indices, len(X), len(selected_indices))
        predictions, final_importance, model, best_row = lopo(X[:, selected_indices], y, groups, info, final_params, parallel=True, retain_best=True, context=f'{name} final retraining')
        predictions.sort_values('patient_id').to_csv(directory / 'patient_predictions.csv', index=False)
        final_importance.to_csv(directory / 'final_selected_feature_importance_all_models.csv', index=False)
        model_path = directory / 'best_ae_model.pkl'
        joblib.dump(model, model_path)
        progress(name, 'Retained minimum-AE model: patient=%s true=%.6f prediction=%.6f AE=%.8f model=%s',
                 best_row['patient_id'], best_row['true_score'], best_row['predicted_score'], best_row['absolute_error'], model_path)
        model_info = dict(best_row, test_segment_count=best_row['test_samples'], feature_type=family, top_k=top_k,
                          selected_feature_indices=json.dumps(selected_indices), age_group_selected=age_selected, model_path=str(model_path))
        pd.DataFrame([model_info]).to_csv(directory / 'best_ae_model_info.csv', index=False)
        metrics = dict(experiment=name, feature_type=family, top_k=top_k, num_patients=len(predictions),
                       patient_level_rmse=float(np.sqrt(mean_squared_error(predictions.true_score, predictions.predicted_score))),
                       patient_level_mae=float(predictions.absolute_error.mean()), age_group_feature_index=X.shape[1]-1,
                       age_group_rank=int(age.rank_by_gain), age_group_selected=age_selected,
                       best_learning_rate=best_params['learning_rate'], best_max_depth=best_params['max_depth'], best_n_estimators=best_params['n_estimators'])
        pd.DataFrame([metrics]).to_csv(directory / 'final_metrics.csv', index=False)
        progress(name, 'FINAL METRICS: all %d patients; patient-level RMSE=%.8f MAE=%.8f', len(predictions), metrics['patient_level_rmse'], metrics['patient_level_mae'])
        progress(name, 'EXPERIMENT COMPLETE: outputs=%s elapsed=%.2fs (%.2f minutes)', directory, time.perf_counter()-started, (time.perf_counter()-started)/60)
        return metrics
    except Exception:
        logging.exception('[%s] Experiment failed after %.2fs', name, time.perf_counter()-started)
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        logging.getLogger('optuna').removeHandler(handler)
        handler.close()

