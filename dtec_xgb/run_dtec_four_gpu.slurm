#!/bin/bash
#SBATCH --job-name=dtec_random11_4gpu
#SBATCH --account=etur92
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Submit from the deployed dtec_xgb directory; create logs before sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the dtec_xgb directory}"
test -f controller.py
test -f SOURCE_COMMIT
module purge
module load EB/apps
module load Python/3.12.3-GCCcore-13.3.0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONPATH="/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool/env_site:/gpfs/projects/etur92/ozu150751/jad/py312_job_site"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DTEC_SOURCE_COMMIT="$(cat SOURCE_COMMIT)"
sha256sum -c SHA256SUMS
nvidia-smi
python -c 'import sys,numpy,pandas,xgboost,optuna,sklearn,joblib; print(sys.executable); print({m.__name__:(m.__version__,m.__file__) for m in (numpy,pandas,xgboost,optuna,sklearn,joblib)})'
# Complete common selection exists before any worker/configuration starts.
python -u controller.py --prepare-only
python -u controller.py --validate-only
python -u controller.py --threads 20
