"""EDAIC fixed-gain-parameter SHAP: four isolated config processes, one per GPU."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def validate(core):
    registry=json.loads((ROOT/'gain_best_params_verified.json').read_text())
    for name, record in registry.items():
        core.load_fixed_params(name)
        assert core.sha256(ROOT.parent/name/'final_patient_level_rmse_mae.csv') == record['final_metrics_sha256']
    data, pids = core.load_and_validate()
    assert len(pids) == 233
    for X, y, groups, ids, scores in data.values():
        assert len(X) == 4829
    return data, pids


def worker(name):
    import runner_core as core
    feature, setting = name.split('_')
    top_k = None if setting == 'all' else int(setting[3:])
    assert (feature, top_k) in core.ORDER
    core.CURRENT_TOP_K = top_k
    data, pids = validate(core)
    out = ROOT/name
    out.mkdir(exist_ok=True)
    lock = (out/'.config.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(f'CONFIG_GPU experiment={name} CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} logical_device=cuda:0', flush=True)
    row = core.experiment(feature, top_k, data[feature])
    if (out/'shap_importance_summary.csv').exists():
        (out/'shap_global_ranking.csv').write_bytes((out/'shap_importance_summary.csv').read_bytes())
    print(f'CONFIG_COMPLETE {name}: {json.dumps(row)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Internal isolated config worker')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.config:
        worker(args.config)
        return
    import runner_core as core
    import shap
    data, pids = validate(core)
    print(f'SHAP={shap.__version__}; four config workers; one GPU per config; no Optuna; fixed gain parameters', flush=True)
    if args.validate_only:
        return
    lock = (ROOT/'.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = [f'{f}_{"all" if k is None else f"top{k}"}' for f,k in core.ORDER]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    assert len(visible) == len(set(visible)) == 4, visible
    core.pd.DataFrame({'fold_index':range(len(pids)), 'patient_id':pids}).to_csv(ROOT/'patient_fold_index.csv',index=False)
    identity = dict(ranking='mean absolute TreeSHAP on each full-feature fold X_train',
        optuna_trials=0, parameter_strategy='matching gain best parameters; no tuning',
        fixed_registry_sha256=core.sha256(ROOT/'gain_best_params_verified.json'),
        population='233 patients; 4829 segments; original EDAIC all segments',
        source_commit=(ROOT/'SOURCE_COMMIT').read_text().strip(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), visible_devices=visible,
        script_sha256={p.name:core.sha256(p) for p in ROOT.glob('*.py')},
        datasets={f:core.sha256(p) for f,(p,c) in core.CONFIGS.items()},
        experiment_order=names, start_time=core.now())
    manifest = ROOT/'run_manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        for key in ('source_commit','script_sha256','datasets','fixed_registry_sha256'):
            assert previous[key]==identity[key], f'Resume identity mismatch: {key}'
    manifest.write_text(json.dumps(identity,indent=2)+'\n')
    core.journal(f'EDAIC fixed-parameter SHAP başladı; 233 hasta/4829 segment; 9 Top-K config; no new Optuna; GPU başına ayrı config; job={identity["slurm_job_id"]}; source={identity["source_commit"]}; kanıt=run_manifest.json.')
    logs=ROOT/'logs'
    logs.mkdir(exist_ok=True)
    devices=queue.Queue()
    for gpu in visible:
        devices.put(gpu)
    # The controller imports metadata only for validation, then frees it before training.
    del data
    def launch(name):
        gpu=devices.get()
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1')
            print(f'START_CONFIG {name}: physical_device={gpu}',flush=True)
            with (logs/f'{name}.log').open('a') as log:
                result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--config',name],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} failed: exit={result.returncode}; log={logs/name}.log')
            return json.loads((ROOT/name/'experiment_complete.json').read_text())['summary']
        finally:
            devices.put(gpu)
    results={}
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(launch,name):name for name in names}
        for future in concurrent.futures.as_completed(futures):
            name=futures[future]
            try:
                results[name]=future.result()
                core.pd.DataFrame([dict(results[n], result_origin='new_fixed_shap') for n in names if n in results] + [dict(v['gain_summary'], result_origin='reused_gain_all') for n,v in json.loads((ROOT/'gain_best_params_verified.json').read_text()).items() if n.endswith('_all')]).to_csv(ROOT/'experiment_summary.csv',index=False)
                print(f'COMPLETE_CONFIG {name}; finished={len(results)}/9',flush=True)
                core.journal(f'{name} tamamlandı; RMSE={results[name]["patient_level_rmse"]}; MAE={results[name]["patient_level_mae"]}; kanıt={ROOT/name/"final_patient_level_rmse_mae.csv"}.')
            except Exception as exc:
                errors.append(str(exc))
                print(f'FAILED_CONFIG {name}: {exc}',flush=True)
    if errors:
        raise RuntimeError('\n'.join(errors))
    assert len(results)==9
    print('COMPLETE all 9 EDAIC fixed SHAP experiments; 3 ALL gain baselines reused',flush=True)


if __name__=='__main__':
    main()
