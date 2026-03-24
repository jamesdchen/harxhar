#!/bin/bash

# ==============================================================
# AE+Ridge GPU backtest SLURM array script (chunked).
#
# Submit with:
#   sbatch --array=1-10 slurm/ae_ridge_backtest.slurm
#   sbatch --array=1-20 --export=GPU_COUNT=4 --gres=gpu:4 slurm/ae_ridge_backtest.slurm
# ==============================================================

#SBATCH --job-name=ae_ridge
#SBATCH --output=logs/ae_ridge_%A_%a.out
#SBATCH --error=logs/ae_ridge_%A_%a.err
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
RESULT_DIR="${RESULT_DIR:-}"

GPU_COUNT="${GPU_COUNT:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
TRAIN_WINDOW="${TRAIN_WINDOW:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"

# --- CHUNKING ---
if [ -n "$SLURM_ARRAY_TASK_COUNT" ]; then
    TOTAL_CHUNKS="$SLURM_ARRAY_TASK_COUNT"
else
    TOTAL_CHUNKS="${TOTAL_CHUNKS:-1}"
fi
CHUNK_ID=$((SLURM_ARRAY_TASK_ID - 1))

if [ -n "$RESULT_DIR" ]; then
    mkdir -p "$RESULT_DIR"
    OUTPUT_FILE="$RESULT_DIR/results_chunk_${SLURM_ARRAY_TASK_ID}.csv"
else
    OUTPUT_FILE=""
fi

# --- ENVIRONMENT SETUP ---
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array ID:   $SLURM_ARRAY_TASK_ID"
echo "Hostname:   $(hostname)"
echo "Experiment: ae_ridge"
echo "GPUs:       $GPU_COUNT"
echo "Chunk:      $CHUNK_ID / $TOTAL_CHUNKS"
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
CMD="python3 -m projects.dl.cli.gpu_executor --experiment ae_ridge --input-path $INPUT_PATH --gpu-count $GPU_COUNT"
CMD="$CMD --chunk-id $CHUNK_ID --total-chunks $TOTAL_CHUNKS"

[ -n "$OUTPUT_FILE" ]   && CMD="$CMD --output $OUTPUT_FILE"
[ -n "$BATCH_SIZE" ]    && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$EPOCHS" ]        && CMD="$CMD --epochs $EPOCHS"
[ -n "$LEARNING_RATE" ] && CMD="$CMD --learning-rate $LEARNING_RATE"
[ -n "$TRAIN_WINDOW" ]  && CMD="$CMD --train-window $TRAIN_WINDOW"
[ -n "$WEIGHTS_DIR" ]   && CMD="$CMD --weights-dir $WEIGHTS_DIR"

echo "Running: $CMD"
echo "============================================"

time $CMD

echo "Job finished."
