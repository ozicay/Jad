"""One shared random-11 selection, then four dynamically scheduled Slurm workers."""
import argparse
import concurrent.futures
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from data import (CONFIGURATIONS, DEFAULT_DATA, ROOT, REFERENCE_SHA256,
                  inspect_metadata, prepare_selection, sha256, verify_selection, write_json)


def launch_command(name, data_dir, selection_dir, run_dir, threads):
    # Same config-level srun isolation used by anxiety_gpu_controller.launch().
    return ['srun', '--exclusive', '--exact', '--nodes=1', '--ntasks=1',
            f'--cpus-per-task={threads}', '--mem=40G', '--gpus-per-task=1', '--gpu-bind=single:1',
            sys.executable, '-u', str(ROOT / 'worker.py'), '--config', name,
            '--data-dir', str(data_dir), '--selection-dir', str(selection_dir),
            '--run-dir', str(run_dir), '--threads', str(threads)]


def launch(name, data_dir, selection_dir, run_dir, threads):
    command = launch_command(name, data_dir, selection_dir, run_dir, threads)
    # stdout includes original-style prints, logging and Optuna's stderr in one config log.
    with (run_dir / 'logs' / f'{name}.log').open('x') as log:
        print('LAUNCH', name, json.dumps(command), flush=True)
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return name, result.returncode


def summary(run_dir):
    import pandas as pd
    rows = []
    for name, _, _ in CONFIGURATIONS:
        marker = run_dir / name / 'COMPLETE.json'
        if marker.exists():
            info = json.loads(marker.read_text())
            rows.append(dict(configuration=name, **info))
    if rows:
        path = run_dir / 'experiment_summary.csv'
        tmp = path.with_suffix('.csv.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(path)


def schedule(data_dir, selection_dir, run_dir, threads, launcher=launch):
    failures = []
    # Four queue consumers; a free slot immediately takes the next independent config.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(launcher, name, data_dir, selection_dir, run_dir, threads): name
                   for name, _, _ in CONFIGURATIONS}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                _, code = future.result()
                if code:
                    failures.append(dict(configuration=name, returncode=code))
            except Exception as error:
                failures.append(dict(configuration=name, error=str(error)))
            print('CONFIG EXIT', name, 'FAILED' if any(f['configuration'] == name for f in failures) else 'COMPLETE', flush=True)
            summary(run_dir)
    if failures:
        write_json(run_dir / 'failures.json', failures)
        raise RuntimeError(f'Configuration failures: {failures}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--selection-dir', type=Path, default=ROOT / 'selection')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    data_dir, selection_dir = args.data_dir.resolve(), args.selection_dir.resolve()
    if sha256(ROOT / 'reference' / 'optuna_xgb_is09_top15.py') != REFERENCE_SHA256:
        raise ValueError('Scientific reference snapshot changed')
    if args.threads < 1 or args.threads > 20:
        raise ValueError('Use 1..20 threads per configuration')
    if args.prepare_only:
        print(json.dumps(prepare_selection(data_dir, selection_dir), indent=2))
        print('SELECTION PREPARED; no models fitted or trials run.')
        return
    tables, keys, manifest = verify_selection(data_dir, selection_dir)
    print(f'COMMON SELECTION VERIFIED: {len(keys)} chunks; sha256={manifest["selection_sha256"]}', flush=True)
    for name, family, k in CONFIGURATIONS:
        print(name, 'top_k=', k, 'trials=40 patients=', manifest['patients'], 'segments=', len(keys), flush=True)
    if args.validate_only:
        plan = [launch_command(name, data_dir, selection_dir, ROOT / 'runs' / 'PLAN_ONLY', args.threads)
                for name, _, _ in CONFIGURATIONS]
        write_json(selection_dir / 'gpu_launch_plan.json', plan)
        print('VALIDATION PASSED; srun plan written; actual four-GPU binding requires a Slurm allocation.')
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Real experiments may only run inside the user-approved Slurm job')
    if int(os.environ.get('SLURM_NTASKS', '0')) != 4:
        raise RuntimeError('Controller requires a four-task/four-GPU job allocation')
    run_dir = (args.run_dir or ROOT / 'runs' / f'{datetime.now():%Y%m%dT%H%M%S}_job{os.environ["SLURM_JOB_ID"]}').resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'logs').mkdir()
    write_json(run_dir / 'run_identity.json', dict(selection_sha256=manifest['selection_sha256'],
               reference_sha256=REFERENCE_SHA256, source_commit=os.environ.get('DTEC_SOURCE_COMMIT'),
               slurm_job=os.environ['SLURM_JOB_ID'], configurations=CONFIGURATIONS))
    schedule(data_dir, selection_dir, run_dir, args.threads)
    if not all((run_dir / name / 'COMPLETE.json').exists() for name, _, _ in CONFIGURATIONS):
        raise RuntimeError('Not all twelve configs produced a completion marker')
    summary(run_dir)
    print('COMPLETE all 12 DTEC configurations', flush=True)


if __name__ == '__main__':
    main()
