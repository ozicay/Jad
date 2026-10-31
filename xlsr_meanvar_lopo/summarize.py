"""Verify each completed configuration from saved patient predictions and checkpoint SHA."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from train_lopo import config_name,patient_metrics
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--target',required=True,choices=['depresyon_skoru','anksiyete_skoru']);a=p.parse_args()
root=Path(a.output)/a.target;rows=[]
for dataset,selection in itertools.product(['geriatric','non_geriatric'],[0,5,10,15]):
    dest=root/config_name((dataset,0,'xlsr300m_meanvar',selection))
    metric=json.loads((dest/'metrics.json').read_text());assert metric['status']=='complete'
    verified=patient_metrics(pd.read_csv(dest/'patient_predictions.csv'))
    for key in ('rmse','mae','pearson_r','pearson_p'):np.testing.assert_allclose(metric[key],verified[key],rtol=1e-10,atol=1e-12,equal_nan=True)
    checkpoint=json.loads((dest/'best_fold_model.json').read_text());assert checkpoint['status']=='verified_complete'
    assert hashlib.sha256((dest/checkpoint['file']).read_bytes()).hexdigest()==checkpoint['sha256']
    rows.append({key:metric[key] for key in ('target','dataset','selection','num_patients','rmse','mae','pearson_r','pearson_p')})
pd.DataFrame(rows).to_csv(root/'experiment_summary.csv',index=False)
print(root/'experiment_summary.csv')
