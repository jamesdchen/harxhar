#!/bin/bash

# ==============================================================
# GPU experiment SLURM script.
#
# Submit with:
#   sbatch slurm/submit_gpu.slurm                         # PatchTSMixer (default)
#   sbatch --export=EXPERIMENT=ae_ridge slurm/submit_gpu.slurm
#   sbatch --export=EXPERIMENT=patchts,GPU_COUNT=4 --gres=gpu:4 slurm/submit_gpu.slurm
# ==============================================================

#SBATCH --job-name=harx_gpu
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="V100|A100|A40|H100"
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

# --- CONFIGURATION (override via --export or edit here) ---
EXPERIMENT="${EXPERIMENT:-patchts}"        # patchts | ae_ridge
INPUT_PATH="${INPUT_PATH:-all30min}"
OUTPUT="${OUTPUT:-}"                       # empty = use config default
GPU_COUNT="${GPU_COUNT:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
TRAIN_WINDOW="${TRAIN_WINDOW:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"            # ae_ridge only

# --- ENVIRONMENT SETUP ---
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Hostname:   $(hostname)"
echo "Experiment: $EXPERIMENT"
echo "GPUs:       $GPU_COUNT"
echo "============================================"

module purge
module load python
module load cuda

# Bind CPU threads to allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

mkdir -p logs

# --- BUILD COMMAND ---
CMD="python3 -m src.cli.gpu_executor --experiment $EXPERIMENT --input-path $INPUT_PATH --gpu-count $GPU_COUNT"

[ -n "$OUTPUT" ]        && CMD="$CMD --output $OUTPUT"
[ -n "$BATCH_SIZE" ]    && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$EPOCHS" ]        && CMD="$CMD --epochs $EPOCHS"
[ -n "$LEARNING_RATE" ] && CMD="$CMD --learning-rate $LEARNING_RATE"
[ -n "$TRAIN_WINDOW" ]  && CMD="$CMD --train-window $TRAIN_WINDOW"
[ -n "$WEIGHTS_DIR" ]   && CMD="$CMD --weights-dir $WEIGHTS_DIR"

echo "Running: $CMD"
echo "============================================"

time $CMD

echo "Job finished."
