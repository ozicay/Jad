"""Read-only config/header/dimension audit; no model training."""
import csv
import sys
from pathlib import Path
import re
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optuna_xgb_configurable_shap_fixed_params_anksiyete import load_config

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'shap_fixed_param_configs/anksiyete'
names = [f'anksiyete_{kind}_{setting}' for kind in ('mfcc', 'egemaps', 'is09')
         for setting in ('top5', 'top10', 'top15')]
assert sorted(p.stem for p in root.glob('*.json')) == sorted(names)
print("experiment\tlearning_rate\tmax_depth\tn_estimators\tgain_parameter_source\tcsv_path\tfeature_column\ttop_k\toutput_dir")
configs=[]
import ast
for name in names:
    config=load_config(root/f"{name}.json")
    assert config["experiment_name"]==name
    assert config["top_k"]==int(name.split("top")[1])
    assert config["output_dir"]==f"/gpfs/projects/etur92/ozu150751/jad/shap_works/shap_fixed_param_exp_models/{name}"
    log=Path(config["gain_parameter_source"]).read_text()
    import re
    matches=re.findall(r"^Best params: (\{[^\n]+\})",log,re.M)
    assert len(matches)==1 and ast.literal_eval(matches[0])==config["fixed_params"]
    assert config["gain_parameter_source"].startswith("/gpfs/projects/etur92/ozu150751/jad/shap_works/logs/anxiety/anx_"+name.removeprefix("anksiyete_")+"_")
    params=config["fixed_params"]
    print("\t".join(str(x) for x in [name,params["learning_rate"],params["max_depth"],params["n_estimators"],config["gain_parameter_source"],config["csv_path"],config["feature_column"],config["top_k"],config["output_dir"]] ))
    configs.append(config)
# Parse every row with baseline token rules and validate dimension, keeping all rows.
num_re = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+(?:\.\d+)?)?')
for config in configs[::3]:
    count = 0
    patients = set()
    with open(config['csv_path'], newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[0] == 'file_name', reader.fieldnames
        assert {'file_name', 'label', 'depresyon_skoru', 'anksiyete_skoru', config['feature_column']} <= set(reader.fieldnames)
        for row in reader:
            tokens = re.split(r'[,\s]+', row[config['feature_column']])
            width = sum(bool(t and num_re.fullmatch(t)) for t in tokens)
            assert width == config['expected_feature_count'], (config['feature_type'], count, width)
            float(row[config['target_column']])
            patients.add(Path(row['file_name']).name.split('-')[0].lower())
            count += 1
    assert count and len(patients) > 1
    print(f"VERIFIED {config['feature_type']}: {count} rows, {len(patients)} patients, {config['expected_feature_count']} features")
