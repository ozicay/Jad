"""Read-only config/header/dimension audit; no model training."""
import csv
import sys
from pathlib import Path
import re
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optuna_xgb_anxiety_configurable import load_config

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'configs/anxiety'
names = [f'anx_{kind}_{setting}' for kind in ('mfcc', 'egemaps', 'is09')
         for setting in ('top5', 'top10', 'top15', 'all')]
assert sorted(p.stem for p in root.glob('*.json')) == sorted(names)
fields = ['experiment_name', 'csv_path', 'feature_column', 'expected_feature_count',
          'target_column', 'top_k', 'output_dir']
print('\t'.join(fields))
configs = []
for name in names:
    config = load_config(root / f'{name}.json')
    assert config['experiment_name'] == name
    assert config['output_dir'] == f'/gpfs/projects/etur92/ozu150751/jad/shap_works/exp_models/{name}'
    assert config['top_k'] == (None if name.endswith('_all') else int(name.split('top')[1]))
    configs.append(config)
    print('\t'.join(str(config[k]) for k in fields))
# Parse every row with baseline token rules and validate dimension, keeping all rows.
num_re = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+(?:\.\d+)?)?')
for config in configs[::4]:
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
            float(row['depresyon_skoru'])
            patients.add(Path(row['file_name']).name.split('-')[0].lower())
            count += 1
    assert count and len(patients) > 1
    print(f"VERIFIED {config['feature_type']}: {count} rows, {len(patients)} patients, {config['expected_feature_count']} features")
