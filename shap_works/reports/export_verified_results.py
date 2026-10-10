"""Read-only MN5 verification of six experiment groups; emits aggregate JSON only."""
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path('/gpfs/projects/etur92/ozu150751/jad/shap_works')
GROUPS = [
 ('gain_depresyon', 'depresyon_skoru', 'gain', 'optuna', 'logs/submissions.tsv', 'configs', '0d53a434116f0670b8415cd08eb617f53736988e'),
 ('gain_anksiyete', 'anksiyete_skoru', 'gain', 'optuna', 'logs/anxiety/submissions.tsv', 'configs/anxiety', 'aa874c44fbda40d23d814aace74d2bf16ee0d222'),
 ('shap_optuna_depresyon', 'depresyon_skoru', 'shap', 'optuna', 'shap_logs/submissions.tsv', 'shap_configs', '9b520cc3849acc3175e08ee1740e9b9fc9b9c9f0'),
 ('shap_optuna_anksiyete', 'anksiyete_skoru', 'shap', 'optuna', 'shap_logs/anksiyete/submissions.tsv', 'shap_configs/anksiyete', '3accadba6b47d37742948323717ea049144e24ae'),
 ('shap_fixed_depresyon', 'depresyon_skoru', 'shap', 'fixed_gain', 'shap_fixed_param_logs/submissions.tsv', 'shap_fixed_param_configs', '113643fb27a6199efca4adf8db90f56eed470f72'),
 ('shap_fixed_anksiyete', 'anksiyete_skoru', 'shap', 'fixed_gain', 'shap_fixed_param_logs/anksiyete/submissions.tsv', 'shap_fixed_param_configs/anksiyete', '7ad18f3db13535f6c4fbcd0ceab66537c39787b4'),
]

def read_csv(path):
 with path.open(newline='') as f: return list(csv.DictReader(f))

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

jobs=[]
for group,target,method,params,ledger,configs,commit in GROUPS:
 with (ROOT/ledger).open() as f:
  for row in csv.DictReader(f, delimiter='\t'):
   row.update(group=group,target=target,method=method,param_strategy=params,config_path=str(ROOT/configs/(row['experiment_name']+'.json')),source_commit=commit)
   jobs.append(row)
assert len(jobs)==66
raw=subprocess.check_output(['sacct','-j',','.join(x['job_id'] for x in jobs),'-X','-n','-P','--format=JobIDRaw,State,ExitCode'],text=True)
states={}
for line in raw.splitlines():
 f=line.split('|')
 if len(f)>=3: states[f[0]]=(f[1],f[2])
summary=[];detailed=[];patient_checks={};param_checks={};issues=[]
for job in jobs:
 name=job['experiment_name']; group=job['group']
 try:
  assert states.get(job['job_id'])==('COMPLETED','0:0'), states.get(job['job_id'])
  cp=Path(job['config_path']); config=json.loads(cp.read_text()); out=Path(config['output_dir'])
  assert config['experiment_name']==name and config['target_column']==job['target']
  kind=config['feature_type'];k=config['top_k'];expected=120
  metrics_path=out/'final_patient_level_rmse_mae.csv'; metrics=read_csv(metrics_path)
  assert [r['setting'] for r in metrics]==['ALL_FEATURES']+([] if k is None else [f'TOP_{k}_FEATURES'])
  stdout=Path(job['stdout']);stderr=Path(job['stderr']); log=stdout.read_text(errors='replace')+'\n'+stderr.read_text(errors='replace')
  errors=[l for l in log.splitlines() if re.search(r'Traceback|Error training|Job failed|Trial .* failed|No successful results|Exception|❌',l)]
  assert not errors, errors[:2]
  if job['param_strategy']=='optuna':
   assert len(re.findall(r'Trial \d+ finished',log))==20
   matches=re.findall(r'^Best params: (\{[^\n]+\})',log,re.M)
   import ast
   assert len(matches)==1
   hyperparameters=ast.literal_eval(matches[0])
  else:
   hyperparameters=config['fixed_params']
   assert json.loads((out/'fixed_params.json').read_text())==hyperparameters
   import ast
   matches=re.findall(r'^Best params: (\{[^\n]+\})',Path(config['gain_parameter_source']).read_text(),re.M)
   assert len(matches)==1 and ast.literal_eval(matches[0])==hyperparameters
  param_checks[(group,kind,k)]=hyperparameters
  assert len(list(out.glob('patient_*/*.pkl')))==(120 if k is None else 240)
  for metric in metrics:
   setting=metric['setting']; is_all=setting=='ALL_FEATURES'
   rp=out/('enhanced_combined_standard_loo_no_flag_results.csv' if is_all else f'enhanced_combined_standard_loo_TOP{k}_results.csv')
   patients=[p for p in read_csv(rp) if not p['patient_id'].startswith('OVERALL')]
   assert len(patients)==expected==int(metric['num_patients'])
   assert len({p['patient_id'] for p in patients})==expected
   assert all(p['status']=='success' for p in patients)
   assert all(int(float(p['features_count']))==(config['expected_feature_count'] if is_all else k) for p in patients)
   truth={p['patient_id']:float(p['true_score']) for p in patients}
   diffs=[float(p['true_score'])-float(p['predicted_score']) for p in patients]
   recomputed_rmse=math.sqrt(sum(d*d for d in diffs)/expected);recomputed_mae=sum(abs(d) for d in diffs)/expected
   rmse=float(metric['patient_level_rmse']);mae=float(metric['patient_level_mae'])
   assert math.isfinite(rmse) and math.isfinite(mae)
   # Baseline stores predictions AND labels at three decimals. Triangle inequality
   # bounds RMSE/MAE discrepancy by .001; reported metrics round to six decimals.
   rmse_diff=abs(recomputed_rmse-rmse);mae_diff=abs(recomputed_mae-mae)
   assert rmse_diff<=.001001 and mae_diff<=.001001,(rmse_diff,mae_diff)
   old=patient_checks.setdefault(job['target'],truth);assert old==truth,'patient/target set mismatch'
   row=dict(group=group,target_column=job['target'],importance_method=job['method'],parameter_strategy=job['param_strategy'],experiment_name=name,feature_type=kind,top_k=k,setting=setting,num_patients=expected,patient_level_rmse=rmse,patient_level_mae=mae,job_id=job['job_id'],slurm_state='COMPLETED',exit_code='0:0',source_commit=job['source_commit'],config_path=str(cp),config_sha256=digest(cp),metrics_path=str(metrics_path),metrics_sha256=digest(metrics_path),patient_results_path=str(rp),patient_results_sha256=digest(rp),stdout=str(stdout),stderr=str(stderr),learning_rate=hyperparameters['learning_rate'],max_depth=hyperparameters['max_depth'],n_estimators=hyperparameters['n_estimators'],reused_gain_all=False,recomputed_rmse_from_rounded_csv=recomputed_rmse,recomputed_mae_from_rounded_csv=recomputed_mae,rmse_rounding_difference=rmse_diff,mae_rounding_difference=mae_diff)
   detailed.append(row)
   if (k is None and is_all) or (k is not None and not is_all):summary.append(row.copy())
 except Exception as exc: issues.append(dict(group=group,experiment=name,job_id=job['job_id'],error=str(exc)))
if not issues:
 assert len(summary)==66 and len(detailed)==120
 for target in ['depresyon_skoru','anksiyete_skoru']:
  suffix='depresyon' if target=='depresyon_skoru' else 'anksiyete'
  for kind in ['mfcc','egemaps','is09']:
   base=next(x for x in summary if x['group']==f'gain_{suffix}' and x['feature_type']==kind and x['top_k'] is None)
   reuse=base.copy();reuse.update(group=f'shap_fixed_{suffix}',importance_method='shap',parameter_strategy='reused_gain_all',reused_gain_all=True)
   summary.append(reuse)
   for k in [5,10,15]:
    assert param_checks[(f'gain_{suffix}',kind,k)]==param_checks[(f'shap_fixed_{suffix}',kind,k)]
    gain=next(x for x in detailed if x['group']==f'gain_{suffix}' and x['feature_type']==kind and x['top_k']==k and x['setting']=='ALL_FEATURES')
    fixed=next(x for x in detailed if x['group']==f'shap_fixed_{suffix}' and x['feature_type']==kind and x['top_k']==k and x['setting']=='ALL_FEATURES')
    assert abs(gain['patient_level_rmse']-fixed['patient_level_rmse'])<=.000001
    assert abs(gain['patient_level_mae']-fixed['patient_level_mae'])<=.000001
print(json.dumps(dict(issues=issues,summary=summary,detailed=detailed,checked_jobs=len(jobs),rounding_tolerance=.001001)))
