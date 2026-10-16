"""Strict D3TEC metadata alignment and one shared, persisted random-11 selection."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"
FAMILIES = {
    "mfcc": ("metadata_with_mfcc_dtec", "features", 608),
    "egemaps": ("metadata_with_egemaps_dtec", "egemaps_features", 88),
    "is09": ("metadata_with_is09_dtec", "is09_features", 384),
}
CONFIGURATIONS = [(f"{family}_{'all' if k is None else f'top{k}'}", family, k)
                  for family in FAMILIES for k in (5, 10, 15, None)]
SELECTION_SEED = 42
CHUNKS = 11
REFERENCE_SHA256 = "e78e97f99278321b3d12035b638e874f2088f8cb5b828a08af7313b188e4fb51"
num_re = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+(?:\.\d+)?)?')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


# Verbatim numeric parsing from the original scientific source.
def parse_token(tok: str) -> float:
    if 'e' in tok or 'E' in tok:
        mant, exp = re.split('[eE]', tok)
        return float(mant) * (10 ** float(exp))
    return float(tok)


def parse_features(s: str) -> np.ndarray:
    toks = re.split(r'[,\s]+', s)
    return np.array([parse_token(t) for t in toks if t and num_re.fullmatch(t)], dtype=float)


def identity(row):
    return f"d3tec_{int(row['Participant_ID']):03d}", int(row['chunk_index'])


def inspect_metadata(directory):
    """Fail rather than silently dropping malformed, missing or mismatched rows."""
    directory = Path(directory)
    inputs = {"base": (directory / "metadata.csv", None, None)}
    inputs.update({family: (directory / filename, col, dim)
                   for family, (filename, col, dim) in FAMILIES.items()})
    tables, hashes, semantic_hashes = {}, {}, {}
    for name, (path, col, dim) in inputs.items():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"Participant_ID", "chunk_index", "depresyon_skoru", "PHQ-9 Score",
                        "file_name", "source_device", "duration_seconds", "source_intervals_json"}
            if col:
                required.add(col)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or [])}")
            table = {}
            for row in reader:
                key = identity(row)
                if key in table:
                    raise ValueError(f"{name}: duplicate patient/chunk ID {key}")
                score = float(row['depresyon_skoru'])
                if not np.isfinite(score) or score != float(row['PHQ-9 Score']) or not 0 <= score <= 27:
                    raise ValueError(f"{name}: invalid or inconsistent target for {key}")
                if row['source_device'] != 'SM-27' or float(row['duration_seconds']) != 20:
                    raise ValueError(f"{name}: expected SM-27 / 20 seconds for {key}")
                if col:
                    # Original parser is retained; reject invalid tokens instead of dropping them silently.
                    tokens = [t for t in re.split(r'[,\s]+', row[col]) if t]
                    if any(not num_re.fullmatch(t) for t in tokens):
                        raise ValueError(f"{name}: invalid feature token for {key}")
                    vector = parse_features(row[col])
                    if vector.shape != (dim,) or not np.isfinite(vector).all():
                        raise ValueError(f"{name}: invalid feature dimensions/values for {key}")
                    row['_vector'] = vector
                table[key] = row
            if not table:
                raise ValueError(f"Empty metadata: {path}")
            tables[name], hashes[name] = table, sha256(path)
            canonical = []
            for key in sorted(table):
                row = {k: v for k, v in table[key].items() if k != '_vector'}
                row['Participant_ID'] = str(int(row['Participant_ID']))
                canonical.append(row)
            semantic_hashes[name] = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    base = tables['base']
    for name, table in tables.items():
        if set(table) != set(base):
            raise ValueError(f"{name}: patient/chunk set differs: missing={len(set(base)-set(table))}, extra={len(set(table)-set(base))}")
        for key, row in table.items():
            for col, value in base[key].items():
                equal = (int(row[col]) == int(value)) if col == 'Participant_ID' else row[col] == value
                if not equal:
                    raise ValueError(f"{name}: metadata mismatch {key}, column {col}")
    grouped = {}
    for key in sorted(base):
        grouped.setdefault(key[0], []).append(key)
    shortages = {pid: len(keys) for pid, keys in grouped.items() if len(keys) < CHUNKS}
    if shortages:
        print('INSUFFICIENT_CHUNKS', json.dumps(shortages), flush=True)
        raise ValueError("Patients have fewer than 11 chunks; no replacement sampling or exclusion allowed")
    for pid, keys in grouped.items():
        if len({float(base[k]['depresyon_skoru']) for k in keys}) != 1:
            raise ValueError(f"Inconsistent patient target: {pid}")
    report = dict(patients=len(grouped), chunks=len(base),
                  min_chunks=min(map(len, grouped.values())), max_chunks=max(map(len, grouped.values())),
                  insufficient_patients=shortages, input_sha256=hashes, semantic_input_sha256=semantic_hashes)
    return tables, grouped, report


def expected_selection(grouped):
    # Sort IDs first so selection is independent of row order, file order and feature family.
    rng = np.random.RandomState(SELECTION_SEED)
    result = []
    for pid, keys in sorted(grouped.items()):
        keys = sorted(keys)
        chosen = sorted(rng.choice(len(keys), CHUNKS, replace=False).tolist())
        result.extend(keys[i] for i in chosen)
    return result


def selection_keys(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [(r['patient_id'], int(r['chunk_index'])) for r in rows]


def prepare_selection(directory, selection_dir):
    tables, grouped, report = inspect_metadata(directory)
    selection_dir = Path(selection_dir)
    selection_dir.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = selection_dir / 'selected_11_chunks.csv', selection_dir / 'selection_manifest.json'
    if csv_path.exists() or manifest_path.exists():
        verify_selection(directory, selection_dir, (tables, grouped, report))
        return report
    keys = expected_selection(grouped)
    # Single controller prepares the complete shared CSV before starting any worker.
    with csv_path.open('x', newline="") as stream:
        fields = ['patient_id', 'Participant_ID', 'chunk_index', 'chunk_id', 'selection_seed']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pid, index in keys:
            writer.writerow(dict(patient_id=pid, Participant_ID=pid.split('_')[-1], chunk_index=index,
                                 chunk_id=f'{pid}_chunk_{index:02d}', selection_seed=SELECTION_SEED))
    write_json(manifest_path, dict(report, selection_seed=SELECTION_SEED, chunks_per_patient=CHUNKS,
                                  selected_rows=len(keys), selection_sha256=sha256(csv_path),
                                  method='RandomState(42).choice without replacement; sorted patient/chunk IDs'))
    return report


def verify_selection(directory, selection_dir, inspected=None):
    tables, grouped, report = inspected or inspect_metadata(directory)
    selection_dir = Path(selection_dir)
    manifest = json.loads((selection_dir / 'selection_manifest.json').read_text())
    if manifest['selection_seed'] != SELECTION_SEED or manifest['chunks_per_patient'] != CHUNKS:
        raise ValueError('Shared selection seed/count differs from the experiment protocol')
    path = selection_dir / 'selected_11_chunks.csv'
    if manifest['semantic_input_sha256'] != report['semantic_input_sha256']:
        raise ValueError("Metadata content changed after common selection (row reordering is allowed)")
    if manifest['selection_sha256'] != sha256(path):
        raise ValueError("Shared selection CSV hash mismatch")
    keys = selection_keys(path)
    # Workers only read/verify the saved list; no RNG or resampling inside any configuration.
    counts = {}
    for key in keys:
        if key not in tables['base']:
            raise ValueError(f'Selected chunk not in base metadata: {key}')
        counts[key[0]] = counts.get(key[0], 0) + 1
    if (len(set(keys)) != len(keys) or keys != sorted(keys) or set(counts) != set(grouped)
            or any(n != CHUNKS for n in counts.values())):
        raise ValueError('Shared selection must contain exactly 11 distinct chunks for every patient')
    for family in FAMILIES:
        if any(key not in tables[family] for key in keys):
            raise ValueError(f"{family}: selected chunk not found")
    return tables, keys, manifest


def load_family(directory, selection_dir, family):
    tables, keys, manifest = verify_selection(directory, selection_dir)
    rows = [tables[family][key] for key in keys]
    X = np.vstack([row['_vector'] for row in rows])
    y = np.array([float(row['depresyon_skoru']) for row in rows])
    groups = np.array([key[0] for key in keys])
    return X, y, groups, keys, manifest
