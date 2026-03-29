#!/bin/bash

# ==============================================================
# Scaling experiment SGE array script for Hoffman2.
#
# Submit with:
#   qsub -t 1-54 -v RESULTS_DIR=results/scaling projects/dl/infra/sge/submit_scaling.sh
#   qsub -t 1-54 -v MULTIPLIERS="0 1 2 5 10 50",REPEATS=3,BLOCK_SIZES="48 96 240" projects/dl/infra/sge/submit_scaling.sh
#
# Grid size = len(BLOCK_SIZES) * len(MULTIPLIERS) * REPEATS
# Default:   3 * 6 * 3 = 54 tasks
#
# Supported GPUs: H200, H100, A100, A6000, V100, RTX2080Ti
# NOT supported:  P4 (4GB VRAM too small)
# ==============================================================

# --- SGE directives ---
#$ -cwd
#$ -j y
#$ -o /u/scratch/j/jamesdc1/
#$ -l gpu,A100,cuda=1
#$ -l h_data=16G,h_rt=14400
#$ -pe shared 4

# --- Configuration (override via -v or edit here) ---
MULTIPLIERS="${MULTIPLIERS:-0 1 2 5 10 50}"
REPEATS="${REPEATS:-3}"
BLOCK_SIZES="${BLOCK_SIZES:-48 96 240}"
MAX_SYNTH_RATIO="${MAX_SYNTH_RATIO:-5.0}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
RESULTS_DIR="${RESULTS_DIR:-results_scaling_laws}"
INPUT_PATH="${INPUT_PATH:-all30min}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"

# --- Task mapping ---
TASK_ID=$((SGE_TASK_ID - 1))             # Convert 1-based to 0-based

# --- Environment Setup ---
echo "============================================"
echo "Job ID:       $JOB_ID"
echo "Array ID:     $SGE_TASK_ID"
echo "Hostname:     $(hostname)"
echo "Task ID:      $TASK_ID"
echo "Multipliers:  $MULTIPLIERS"
echo "Repeats:      $REPEATS"
echo "Block sizes:  $BLOCK_SIZES"
echo "Max ratio:    $MAX_SYNTH_RATIO"
echo "Results dir:  $RESULTS_DIR"
echo "============================================"

source /u/local/Modules/default/init/modules.sh
module load conda cuda/12.3

conda activate harxhar-dl

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# --- Build Command ---
CMD="python3 -m projects.dl.scripts.run_scaling_experiment --task-id $TASK_ID"
CMD="$CMD --multipliers $MULTIPLIERS --repeats $REPEATS --results-dir $RESULTS_DIR --input-path $INPUT_PATH"
CMD="$CMD --block-sizes $BLOCK_SIZES"
CMD="$CMD --max-synth-ratio $MAX_SYNTH_RATIO"
CMD="$CMD --train-frac $TRAIN_FRAC"

[ -n "$BATCH_SIZE" ]     && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$EPOCHS" ]         && CMD="$CMD --epochs $EPOCHS"
[ -n "$LEARNING_RATE" ]  && CMD="$CMD --learning-rate $LEARNING_RATE"

echo "Running: $CMD"
echo "============================================"

time $CMD

echo "Job finished."
