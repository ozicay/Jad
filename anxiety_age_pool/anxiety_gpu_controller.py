"""Four independent config workers; each Slurm step is bound to one GPU."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs'
REQUIRED = ['experiment_config.json','optuna_trials.csv','best_params.json','patient_predictions.csv',
            'final_metrics.csv','feature_importance_all_models.csv','selected_features.csv',
            'selected_feature_indices.csv','best_ae_model.pkl','best_ae_model_info.csv','run.log']


def history_trials(path):
    import pandas as pd
    import optuna
    d = pd.read_csv(path, float_precision='round_trip')
    if len(d) > 40 or not d.state.eq('COMPLETE').all() or list(d.number) != list(range(len(d))):
        raise ValueError(f'Cannot import non-contiguous completed trial history: {path}')
    distributions = {'learning_rate':optuna.distributions.FloatDistribution(1e-3,.1,log=True),
                     'max_depth':optuna.distributions.IntDistribution(4,14),
                     'n_estimators':optuna.distributions.IntDistribution(100,500)}
    result=[]
    for row in d.itertuples():
        params={'learning_rate':float(row.params_learning_rate),'max_depth':int(row.params_max_depth),
                'n_estimators':int(row.params_n_estimators)}
        result.append(optuna.trial.create_trial(params=params, distributions=distributions, value=float(row.value)))
    return result


def prepare_study(directory, name):
    import optuna
    database = directory / 'optuna_study.sqlite3'
    history = directory / 'optuna_trials.csv'
    # A prepared DB is always the authoritative history on subsequent restarts.
    new_database = not database.exists()
    imported = history_trials(history) if new_database and history.exists() else []
    study = optuna.create_study(direction='minimize',storage=f'sqlite:///{database}',study_name=name,load_if_exists=True)
    if imported:
        study.add_trials(imported)
        (directory / 'imported_trial_history.json').write_text(json.dumps({
            'source_csv_sha256':hashlib.sha256(history.read_bytes()).hexdigest(),
            'completed_trials_imported':len(imported),'incomplete_trial_restarted':True,
            'sampler_state':'original unseeded in-memory sampler state unavailable; new default sampler uses preserved history',
            'original_trial_timestamps':'preserved in migration snapshot; imported trials receive new storage timestamps'},indent=2)+'\n')
    if len(study.trials)>40 or any(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError('Study has incomplete trials; inspect attempt before restarting')
    return study


def completed(name):
    import pandas as pd
    d=OUTPUT/name
    if not (d/'final_metrics.csv').exists():
        return False
    missing=[f for f in REQUIRED if not (d/f).is_file()]
    if missing:
        raise ValueError(f'{name}: final metrics exist but required artifacts missing: {missing}')
    predictions=pd.read_csv(d/'patient_predictions.csv')
    trials=pd.read_csv(d/'optuna_trials.csv')
    if len(predictions)!=204 or predictions.patient_id.nunique()!=204 or len(trials)!=40 or not trials.state.eq('COMPLETE').all():
        raise ValueError(f'{name}: completed cohort/trial coverage mismatch')
    return True


def validate_data():
    import pandas as pd
    import anxiety_four_gpu_worker as w
    data={family:w.load_family(family) for family in w.FAMILIES}
    base=data['mfcc']
    expected=pd.read_csv(ROOT/'expected_pooled_patient_index.csv')
    actual=pd.DataFrame([dict(patient_id=pid,cohort=v['cohort'],age_group=v['age_group']) for pid,v in sorted(base[3].items())])
    if not actual.equals(expected.sort_values('patient_id').reset_index(drop=True)):
        raise ValueError('Pooled cohort differs from Table-4')
    for family,loaded in data.items():
        if set(loaded[2])!=set(base[2]) or loaded[3]!=base[3]:
            raise ValueError(f'{family}: cohort/target mismatch')
    return data


def worker(name):
    import anxiety_four_gpu_worker as w
    configs={n:(family,k) for n,family,k in w.CONFIGURATIONS}
    family,k=configs[name]
    if completed(name):
        raise ValueError(f'Refusing to run completed config {name}')
    data=w.load_family(family)
    directory=OUTPUT/name;directory.mkdir(exist_ok=True)
    old_config=json.loads((directory/'experiment_config.json').read_text()) if (directory/'experiment_config.json').exists() else None
    if old_config and (old_config['target_column']!=w.TARGET or old_config['top_k']!=k or old_config['selected_patient_csv_sha256']!=data[4]['selected_patient_csv_sha256']):
        raise ValueError('Config identity changed across migration')
    config=dict(data[4],experiment_name=name,feature_type=family,top_k=k,optuna_trials=w.TRIALS,
                execution='four concurrent configurations; one GPU per config',
                migration_from_job='47287479',execution_job=os.environ.get('SLURM_JOB_ID'),
                visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                previous_source_commit=old_config.get('source_commit') if old_config else None)
    (directory/'experiment_config.json').write_text(json.dumps(config,indent=2)+'\n')
    w.run_experiment(name,family,k,data)


def collect_summary():
    import pandas as pd
    import anxiety_four_gpu_worker as w
    rows=[pd.read_csv(OUTPUT/name/'final_metrics.csv') for name,_,_ in w.CONFIGURATIONS if completed(name)]
    if rows:
        temporary=OUTPUT/'anxiety_age_pool_summary.csv.tmp'
        pd.concat(rows,ignore_index=True).to_csv(temporary,index=False)
        temporary.replace(OUTPUT/'anxiety_age_pool_summary.csv')


def launch(name):
    command=['srun','--exclusive','--exact','--nodes=1','--ntasks=1','--cpus-per-task=20',
             '--gpus-per-task=1','--gpu-bind=single:1',sys.executable,'-u',str(Path(__file__)), '--worker',name]
    with (ROOT/'logs'/f'gpu4_{os.environ["SLURM_JOB_ID"]}_{name}.out').open('a') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    return name,result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--worker')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.worker:
        worker(args.worker);return
    import anxiety_four_gpu_worker as w
    if args.validate_only:
        data=validate_data()
        for name,family,k in w.CONFIGURATIONS:
            print(name,'top_k=',k,'target=',w.TARGET,'trials=',w.TRIALS,'patients=',data[family][4]['total_patients'],
                  'features=',data[family][0].shape[1],'completed=',completed(name))
        print('VALIDATION PASSED; no fitting or Optuna trials executed.');return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training controller must run inside Slurm')
    pending=[name for name,_,_ in w.CONFIGURATIONS if not completed(name)]
    collect_summary()
    failures=[]
    # Four bounded config workers; each step receives a different GPU from Slurm.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures={executor.submit(launch,name):name for name in pending}
        for future in concurrent.futures.as_completed(futures):
            name,code=future.result()
            print('CONFIG EXIT',name,code,flush=True)
            if code:failures.append((name,code))
            else:collect_summary()
    if failures:
        raise RuntimeError(f'Configuration failures: {failures}')
    collect_summary()

if __name__=='__main__':
    main()
