"""Create shareable aggregate CSVs from the verified MN5 export; no patient data."""
import argparse
import csv
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

parser=argparse.ArgumentParser()
parser.add_argument('--input',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
lines=Path(args.input).read_text().splitlines()
data=json.loads(next(line for line in lines if line.startswith('{')))
assert not data['issues'], data['issues']
assert data['checked_jobs']==66 and len(data['summary'])==72 and len(data['detailed'])==120
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
kind_order={'mfcc':0,'egemaps':1,'is09':2}
rows=sorted(data['summary'],key=lambda r:(r['group'],kind_order[r['feature_type']],r['top_k'] if r['top_k'] is not None else 99))
for row in rows:row['feature_setting']='ALL' if row['top_k'] is None else f"TOP_{row['top_k']}"

def write(name,records):
 with (out/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
write('six_groups_results.csv',rows)
write('all_run_metrics_with_full_feature_controls.csv',data['detailed'])
for group in sorted({r['group'] for r in rows}):
 group_rows=[r for r in rows if r['group']==group];assert len(group_rows)==12
 write(group+'.csv',group_rows)
comparison=[]
for target,suffix in [('depresyon_skoru','depresyon'),('anksiyete_skoru','anksiyete')]:
 for kind in ['mfcc','egemaps','is09']:
  for k in [5,10,15,None]:
   c=dict(target_column=target,feature_type=kind,feature_setting='ALL' if k is None else f'TOP_{k}',num_patients=120)
   for strategy in ['gain','shap_optuna','shap_fixed']:
    r=next(r for r in rows if r['group']==strategy+'_'+suffix and r['feature_type']==kind and r['top_k']==k)
    c[strategy+'_rmse']=r['patient_level_rmse'];c[strategy+'_mae']=r['patient_level_mae']
    c[strategy+'_job_id']=r['job_id']
   c['shap_fixed_reuses_gain_all']=k is None
   comparison.append(c)
write('gain_shap_comparison.csv',comparison)
max_rmse=max(r['rmse_rounding_difference'] for r in data['detailed']);max_mae=max(r['mae_rounding_difference'] for r in data['detailed'])
verified=dict(verified_at=datetime.datetime.now(ZoneInfo('Europe/Istanbul')).isoformat(),jobs=66,all_completed=True,all_exit_codes='0:0',summary_rows=72,direct_metric_rows=120,reused_gain_all_rows=6,patients_per_setting=120,errors=data['issues'],max_rmse_recomputation_difference=max_rmse,max_mae_recomputation_difference=max_mae,rounding_tolerance=data['rounding_tolerance'],fixed_gain_params_match=True,fixed_full_feature_controls_match_gain=True,source='MN5 production final_patient_level_rmse_mae.csv and stored patient predictions; sacct and log checks',rounding_note='Patient predictions/labels stored at three decimals; final metrics stored at six. CSV values are the authoritative final metrics, recomputation uses a 0.001001 bound. Models were not reloaded or predictions rerun.')
(out/'verification.json').write_text(json.dumps(verified,indent=2)+'\n')
print(json.dumps(verified,indent=2))
for c in comparison:print(c)
