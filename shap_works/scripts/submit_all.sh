#!/bin/bash
set -euo pipefail
ROOT=/gpfs/projects/etur92/ozu150751/jad/shap_works
NAMES=(mfcc_top5 mfcc_top10 mfcc_top15 mfcc_all egemaps_top5 egemaps_top10 egemaps_top15 egemaps_all is09_top5 is09_top10 is09_top15 is09_all)
# Refuse duplicate launches. Ledger persists every accepted job immediately.
LEDGER="$ROOT/logs/submissions.tsv"
if [[ -e "$LEDGER" ]]; then
    echo "Submission ledger already exists: $LEDGER; inspect it before any resubmission." >&2
    exit 1
fi
for name in "${NAMES[@]}"; do
    test -f "$ROOT/configs/$name.json"
    test -d "$ROOT/exp_models/$name"
    if [[ -n "$(find "$ROOT/exp_models/$name" -mindepth 1 -print -quit)" ]]; then
        echo "Output directory is not empty: $name" >&2
        exit 1
    fi
done
set -o noclobber
printf 'experiment_name\tjob_id\tstdout\tstderr\n' > "$LEDGER"
for name in "${NAMES[@]}"; do
    job=$(sbatch --parsable --job-name="$name" "$ROOT/scripts/run_experiment.slurm" "$ROOT/configs/$name.json")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s\n' "$name" "$job" "$ROOT/logs/${name}_${job}.out" "$ROOT/logs/${name}_${job}.err" | tee -a "$LEDGER"
done
