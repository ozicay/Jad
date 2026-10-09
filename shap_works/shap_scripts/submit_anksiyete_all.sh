#!/bin/bash
set -euo pipefail
ROOT=/gpfs/projects/etur92/ozu150751/jad/shap_works
NAMES=(anksiyete_mfcc_top5 anksiyete_mfcc_top10 anksiyete_mfcc_top15 anksiyete_mfcc_all anksiyete_egemaps_top5 anksiyete_egemaps_top10 anksiyete_egemaps_top15 anksiyete_egemaps_all anksiyete_is09_top5 anksiyete_is09_top10 anksiyete_is09_top15 anksiyete_is09_all)
# Refuse duplicate launches. Ledger persists every accepted job immediately.
LEDGER="$ROOT/shap_logs/anksiyete/submissions.tsv"
if [[ -e "$LEDGER" ]]; then
    echo "Submission ledger already exists: $LEDGER; inspect it before any resubmission." >&2
    exit 1
fi
for name in "${NAMES[@]}"; do
    test -f "$ROOT/shap_configs/anksiyete/$name.json"
    test -d "$ROOT/shap_exp_models/$name"
    if [[ -n "$(find "$ROOT/shap_exp_models/$name" -mindepth 1 -print -quit)" ]]; then
        echo "Output directory is not empty: $name" >&2
        exit 1
    fi
done
set -o noclobber
printf 'experiment_name\tjob_id\tstdout\tstderr\n' > "$LEDGER"
for name in "${NAMES[@]}"; do
    job=$(sbatch --parsable --job-name="shap_$name" "$ROOT/shap_scripts/run_anksiyete_experiment.slurm" "$ROOT/shap_configs/anksiyete/$name.json")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s\n' "$name" "$job" "$ROOT/shap_logs/anksiyete/shap_${name}_${job}.out" "$ROOT/shap_logs/anksiyete/shap_${name}_${job}.err" | tee -a "$LEDGER"
done
