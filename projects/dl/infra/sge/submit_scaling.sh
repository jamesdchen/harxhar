#!/bin/bash

# ==============================================================
# Scaling experiment SGE array script for Hoffman2.
#
# Submit with:
#   qsub -t 1-50 -v MULTIPLIERS="1 2 4 8",REPEATS=5,RESULTS_DIR=results/scaling projects/dl/infra/sge/submit_scaling.sh
#   qsub -t 1-20 -v MULTIPLIERS="1 2 4",REPEATS=3,RESULTS_DIR=results/scaling,BATCH_SIZE=64 projects/dl/infra/sge/submit_scaling.sh
#
# Supported GPUs: H200, H100, A100, A6000, V100, RTX2080Ti
# NOT supported:  P4 (4GB VRAM too small)
# ==============================================================

# --- SGE directives ---
#$ -cwd
#$ -j y
#$ -o /u/scratch/j/jamesdc1/
#$ -l gpu,A100,cuda=1
#$ -l h_data=16G,h_rt=3600
#$ -pe shared 4

# --- Configuration (override via -v or edit here) ---
MULTIPLIERS="${MULTIPLIERS:-1 2 4 8}"
REPEATS="${REPEATS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
TRAIN_FRAC="${TRAIN_FRAC:-}"
RESULTS_DIR="${RESULTS_DIR:-results/scaling}"
INPUT_PATH="${INPUT_PATH:-all30min}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"

# --- Task mapping ---
if [ -n "$TOTAL_TASKS" ]; then
    TOTAL_TASKS="$TOTAL_TASKS"
else
    TOTAL_TASKS=1
fi
TASK_ID=$((SGE_TASK_ID - 1))             # Convert 1-based to 0-based

# --- Environment Setup ---
echo "============================================"
echo "Job ID:      $JOB_ID"
echo "Array ID:    $SGE_TASK_ID"
echo "Hostname:    $(hostname)"
echo "Task:        $TASK_ID / $TOTAL_TASKS"
echo "Multipliers: $MULTIPLIERS"
echo "Repeats:     $REPEATS"
echo "============================================"

source /u/local/Modules/default/init/modules.sh
module load conda cuda/12.3

# Activate DL conda environment (Python 3.11 + torch-cuda + transformers)
conda activate harxhar-dl

# Bind CPU threads to allocated cores
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# --- Build Command ---
CMD="python3 -m projects.dl.scripts.run_scaling_experiment --task-id $TASK_ID --total-tasks $TOTAL_TASKS"
CMD="$CMD --multipliers $MULTIPLIERS --repeats $REPEATS --results-dir $RESULTS_DIR --input-path $INPUT_PATH"

[ -n "$BLOCK_SIZE" ]     && CMD="$CMD --block-size $BLOCK_SIZE"
[ -n "$TRAIN_FRAC" ]     && CMD="$CMD --train-frac $TRAIN_FRAC"
[ -n "$BATCH_SIZE" ]     && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$EPOCHS" ]         && CMD="$CMD --epochs $EPOCHS"
[ -n "$LEARNING_RATE" ]  && CMD="$CMD --learning-rate $LEARNING_RATE"

echo "Running: $CMD"
echo "============================================"

time $CMD

echo "Job finished."
