#!/bin/bash
#SBATCH --job-name=edaic_random5
#SBATCH --account=etur92
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/projects/etur92/ozu150751/jad/edaic_xgb/random_5_segment/logs/slurm_%j.out
#SBATCH --error=/gpfs/projects/etur92/ozu150751/jad/edaic_xgb/random_5_segment/logs/slurm_%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu150751/jad/edaic_xgb/random_5_segment
set -euo pipefail
module purge
module load EB/apps
module load Python/3.12.3-GCCcore-13.3.0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONPATH="/gpfs/projects/etur92/ozu150751/jad/anxiety_age_pool/env_site:/gpfs/projects/etur92/ozu150751/jad/py312_job_site:$HOME/.local/lib/python3.12/site-packages"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
sha256sum -c SHA256SUMS
hostname
nvidia-smi
python -c "import subprocess; names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines(); assert len(names)==4 and all('H100' in n for n in names), names; print('Verified four H100 GPUs:', names)"
python record_environment.py "logs/environment_${SLURM_JOB_ID}.json"
# One process, twelve ordered experiments; never a job array or twelve parallel jobs.
srun python -u run_random5.py
