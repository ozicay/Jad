#!/bin/bash
#SBATCH --job-name=depr-is09-top15-shap
#SBATCH --account=etur92
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/projects/etur92/ozu150751/jad/xgb_codes/depr_is09_top15_shap/logs/%x-%j.out
#SBATCH --error=/gpfs/projects/etur92/ozu150751/jad/xgb_codes/depr_is09_top15_shap/logs/%x-%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu150751/jad

# Resource/environment template: completed Jad job 45887169 (geriatri_gdö_xgb.slurm).
set -e
set -o pipefail
module purge
module load EB/apps
module load Python/3.12.3-GCCcore-13.3.0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONNOUSERSITE=0
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

PROJECT_ROOT=/gpfs/projects/etur92/ozu150751/jad
SOURCE_DIR="$PROJECT_ROOT/source-shap"
SCRIPT_PATH="$SOURCE_DIR/optuna_xgb_is09_top15_shap.py"
METADATA="$PROJECT_ROOT/geriatri_new/geriatri_metadata/metadata_with_is09.csv"

test -f "$METADATA"
test -f "$SCRIPT_PATH"
hostname
pwd
git -C "$SOURCE_DIR" status --short
git -C "$SOURCE_DIR" log -1 --oneline
git -C "$SOURCE_DIR" rev-parse HEAD
python --version
python -c "import shap, xgboost, optuna; print('SHAP', shap.__version__, 'XGBoost', xgboost.__version__, 'Optuna', optuna.__version__)"
python -u "$SCRIPT_PATH"
