#!/bin/bash
#SBATCH --job-name=lgbm_carc_apr30
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --account=pollok_1603

set -e

TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Array:     $SLURM_ARRAY_TASK_ID -> task $TASK_ID"
echo "Host:      $(hostname)"
echo "Manifest:  ${HPC_MANIFEST:-_hpc_dispatch.json}"
echo "=========================================="

source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate project-cucuringu

cd /scratch1/jc_905/harxhar
export PYTHONPATH="/scratch1/jc_905/harxhar:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TASK_ID
export HPC_MANIFEST="${HPC_MANIFEST:-_hpc_dispatch_lgbm_carc_apr30.json}"

time python3 _hpc_dispatch.py
