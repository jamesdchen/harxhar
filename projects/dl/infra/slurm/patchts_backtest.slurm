#!/bin/bash

# ==============================================================
# PatchTST GPU backtest SLURM script.
#
# Submit from harxhar/ with:
#   sbatch projects/dl/infra/slurm/patchts_backtest.slurm
#   sbatch --export=GPU_COUNT=4 --gres=gpu:4 projects/dl/infra/slurm/patchts_backtest.slurm
# ==============================================================

#SBATCH --job-name=patchts
#SBATCH --output=logs/patchts_%j.out
#SBATCH --error=logs/patchts_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|a40|v100|l40s"
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --chdir=/home1/jc_905/harxhar

# --- CONFIGURATION (override via --export or edit here) ---
INPUT_PATH="${INPUT_PATH:-all30min}"
OUTPUT="${OUTPUT:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
TRAIN_WINDOW="${TRAIN_WINDOW:-}"

GPU_COUNT="${GPU_COUNT:-2}"

# --- ENVIRONMENT SETUP ---
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Hostname:   $(hostname)"
echo "Experiment: patchts"
echo "GPUs:       $GPU_COUNT"
echo "============================================"

module purge

# Activate conda environment with all dependencies
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate project-cucuringu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

mkdir -p logs

# --- BUILD COMMAND ---
CMD="python3 -m projects.dl.cli.gpu_executor --experiment patchts --input-path $INPUT_PATH --gpu-count $GPU_COUNT"

[ -n "$OUTPUT" ]        && CMD="$CMD --output $OUTPUT"
[ -n "$BATCH_SIZE" ]    && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$EPOCHS" ]        && CMD="$CMD --epochs $EPOCHS"
[ -n "$LEARNING_RATE" ] && CMD="$CMD --learning-rate $LEARNING_RATE"
[ -n "$TRAIN_WINDOW" ]  && CMD="$CMD --train-window $TRAIN_WINDOW"

echo "Running: $CMD"
echo "============================================"

time $CMD

echo "Job finished."
