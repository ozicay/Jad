#!/bin/bash
#SBATCH --job-name=anxiety_age_pool
#SBATCH --account=etur92
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool/logs/slurm_%j.out
#SBATCH --error=/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool/logs/slurm_%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool
set -euo pipefail
module purge
module load EB/apps
module load Python/3.12.3-GCCcore-13.3.0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONPATH="$PWD/env_site:/gpfs/projects/etur92/ozu150751/jad/py312_job_site:$HOME/.local/lib/python3.12/site-packages"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export ANXIETY_SOURCE_COMMIT="$(cat SOURCE_COMMIT)"
sha256sum -c SHA256SUMS
nvidia-smi
python record_environment.py "logs/environment_${SLURM_JOB_ID}.json"
# One srun, one Python process: all twelve configurations sequentially.
srun python -u run_anxiety_age_pool.py
