#!/bin/bash
# SLURM CPU Array Job Template with TASK_OFFSET support (CARC Discovery)

#SBATCH --job-name=tune_tree
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=main
#SBATCH --account=pollok_1603

set -e

RESULT_DIR="${RESULT_DIR:-.}"
REPO_DIR="${REPO_DIR:-.}"

TASK_ID=$((SLURM_ARRAY_TASK_ID - 1 + ${TASK_OFFSET:-0}))

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Hostname:     $(hostname)"
echo "Task:         $TASK_ID (offset=${TASK_OFFSET:-0})"
echo "============================================"

if [ -n "$MODULES" ]; then
    for mod in $MODULES; do
        module load "$mod"
    done
fi

if [ -n "$CONDA_SOURCE" ]; then
    source "$CONDA_SOURCE"
fi
if [ -n "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
mkdir -p "$RESULT_DIR"

# Pin OpenMP / BLAS thread counts to the cgroup CPU allocation. XGBoost,
# LightGBM, NumPy, etc. otherwise call multiprocessing.cpu_count() which
# returns the host's full core count (not the cgroup limit) and oversubscribe.
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export NUMEXPR_NUM_THREADS="$NTHREADS"
export VECLIB_MAXIMUM_THREADS="$NTHREADS"

echo "Result dir:   $RESULT_DIR"
echo "Executor:     $EXECUTOR"
echo "============================================"

export TASK_ID RESULT_DIR
time $EXECUTOR ${EXTRA_ARGS:-}

echo "Job finished."
