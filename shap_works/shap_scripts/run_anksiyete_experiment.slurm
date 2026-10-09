#!/bin/bash
#SBATCH --account=etur92
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/projects/etur92/ozu150751/jad/shap_works/shap_logs/anksiyete/%x_%j.out
#SBATCH --error=/gpfs/projects/etur92/ozu150751/jad/shap_works/shap_logs/anksiyete/%x_%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu150751/jad

# Same environment/resources as completed Jad job 45887169; memory uses partition default.
set -euo pipefail
CONFIG_PATH=${1:?Usage: sbatch run_experiment.slurm CONFIG_PATH}
PROJECT_ROOT=/gpfs/projects/etur92/ozu150751/jad
SOURCE_DIR="$PROJECT_ROOT/shap_works/source-shap-anksiyete"
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

test -f "$CONFIG_PATH"
cd "$SOURCE_DIR"
git rev-parse HEAD
python --version
python -c "import numpy, pandas, sklearn, joblib, xgboost, optuna, shap; print('SHAP', shap.__version__, 'XGBoost', xgboost.__version__, 'Optuna', optuna.__version__)"
python -u "$PROJECT_ROOT/shap_works/optuna_xgb_configurable_shap_anksiyete.py" --config "$CONFIG_PATH"
