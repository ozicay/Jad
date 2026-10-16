"""Select five shared segments per eligible EDAIC patient, without replacement."""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
FILES = ('metadata_with_mfcc.csv', 'metadata_with_egemapsv0b1.csv', 'metadata_with_is09.csv')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source):
    datasets = []
    sources = {}
    common = None
    for filename in FILES:
        path = source / filename
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
        index = {r['file_name']: r for r in rows}
        if len(index) != len(rows):
            raise ValueError(f'Duplicate segments: {path}')
        identity = {key: float(r['depresyon_skoru']) for key, r in index.items()}
        if common is None:
            common = identity
        elif common != identity:
            raise ValueError('Cross-feature segment/target mismatch')
        datasets.append((filename, headers, index))
        sources[filename] = digest(path)
    patients = defaultdict(list)
    scores = {}
    for filename, score in common.items():
        pid = Path(filename).name.split('-')[0].lower()
        if pid in scores and scores[pid] != score:
            raise ValueError(f'Conflicting scores for patient {pid}')
        scores[pid] = score
        patients[pid].append(filename)
    rng = random.Random(SEED)
    chosen, excluded = [], []
    for pid in sorted(patients):
        files = sorted(patients[pid])
        if len(files) < 5:
            excluded.append(dict(patient_id=pid, num_segments=len(files)))
            continue
        chosen.extend((pid, name) for name in sorted(rng.sample(files, 5)))
    assert len(chosen) == 1145 and len({pid for pid, _ in chosen}) == 229
    # Metadata outputs contain only the selected rows, in a shared deterministic order.
    output = ROOT / 'metadata'
    output.mkdir(exist_ok=True)
    outputs = {}
    for filename, headers, index in datasets:
        import io
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator='\n')
        writer.writeheader()
        writer.writerows(index[name] for _, name in chosen)
        content = buffer.getvalue().encode()
        path = output / filename
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'Refusing to change existing selected metadata: {path}')
        path.write_bytes(content)
        outputs[filename] = digest(path)
    selection = ROOT / 'selected_segments.csv'
    with selection.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'file_name'])
        writer.writerows(chosen)
    with (ROOT / 'excluded_patients.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'num_segments'])
        writer.writeheader()
        writer.writerows(excluded)
    manifest = dict(seed=SEED, segments_per_patient=5, sampling='without replacement',
        selection_order='sorted patient IDs; random.Random(42).sample(sorted filenames, 5)',
        eligible_patients=229, selected_segments=1145, excluded_patients=len(excluded),
        source_metadata_sha256=sources, selected_metadata_sha256=outputs,
        selected_segments_sha256=digest(selection),
        target='depresyon_skoru', fixed_selection_across_trials_features_and_configs=True)
    path = ROOT / 'selection_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing selection manifest differs')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print('PASS: 229 patients, 1145 segments, five per patient, identical selection across all three features; seed=42.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-metadata', type=Path, default=ROOT.parent/'metadata')
    prepare(parser.parse_args().source_metadata)
