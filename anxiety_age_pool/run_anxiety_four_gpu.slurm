#!/bin/bash
#SBATCH --job-name=anxiety_age_pool_4gpu
#SBATCH --account=etur92
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool/logs/slurm_4gpu_%j.out
#SBATCH --error=/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool/logs/slurm_4gpu_%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool
set -euo pipefail
module purge
module load EB/apps
module load Python/3.12.3-GCCcore-13.3.0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONPATH="$PWD/env_site:/gpfs/projects/etur92/ozu150751/jad/py312_job_site:$HOME/.local/lib/python3.12/site-packages"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export ANXIETY_SOURCE_COMMIT="$(cat SOURCE_COMMIT_4GPU)"
sha256sum -c SHA256SUMS_4GPU
nvidia-smi
python record_environment.py "logs/environment_4gpu_${SLURM_JOB_ID}.json"
python -u anxiety_gpu_controller.py
