#!/bin/bash
#SBATCH --job-name=edaic_xgb
#SBATCH --account=etur92
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/projects/etur92/ozu150751/jad/edaic_xgb/logs/slurm_%j.out
#SBATCH --error=/gpfs/projects/etur92/ozu150751/jad/edaic_xgb/logs/slurm_%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu150751/jad/edaic_xgb
set -euo pipefail
module purge
module load EB/apps
module load Python/3.12.3-GCCcore-13.3.0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONPATH="$PWD/env_site:/gpfs/projects/etur92/ozu150751/jad/py312_job_site:$HOME/.local/lib/python3.12/site-packages"
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
srun python -u run_edaic_xgb.py --resume-from-sha256 a8ac8030f8029f7dc8e2be353aded81c49e922f2a34b33a5984917dc22c191b8
