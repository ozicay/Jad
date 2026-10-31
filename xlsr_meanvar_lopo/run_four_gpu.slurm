#!/bin/bash
#SBATCH --job-name=xlsr_meanvar_lopo
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${TARGET:?Set TARGET=depresyon_skoru or anksiyete_skoru}"
: "${INPUT_DIR:?Directory containing geriatric.csv and non_geriatric.csv}"
: "${MODEL_DIR:?Local XLS-R300M model directory}"
case "$TARGET" in depresyon_skoru|anksiyete_skoru) ;; *) exit 2 ;; esac
FEATURE_PYTHON=${FEATURE_PYTHON:-python}
TRAIN_PYTHON=${TRAIN_PYTHON:-python}
RUN_DIR=${RUN_DIR:-"$PWD/outputs/slurm_$TARGET"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR/features" "$RUN_DIR/logs"
if [[ -n "${FEATURE_PYTHONPATH:-}" ]]; then export PYTHONPATH="$FEATURE_PYTHONPATH"; fi
for group in geriatric non_geriatric; do
    srun --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
        "$FEATURE_PYTHON" extract_features.py --csv "$INPUT_DIR/$group.csv" \
        --output "$RUN_DIR/features/$group.csv" --model "$MODEL_DIR" --local-files-only --world-size 4
    "$FEATURE_PYTHON" extract_features.py --csv "$INPUT_DIR/$group.csv" \
        --output "$RUN_DIR/features/$group.csv" --model "$MODEL_DIR" --local-files-only --world-size 4 --merge-only
 done
if [[ -n "${TRAIN_PYTHONPATH:-}" ]]; then export PYTHONPATH="$TRAIN_PYTHONPATH"; fi
srun --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=0 \
    "$TRAIN_PYTHON" train_worker.py --data-dir "$RUN_DIR/features" --output "$RUN_DIR" --target "$TARGET"
"$TRAIN_PYTHON" summarize.py --output "$RUN_DIR" --target "$TARGET"
