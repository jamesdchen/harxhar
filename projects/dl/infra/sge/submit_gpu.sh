#!/bin/bash

# ==============================================================
# GPU experiment SGE array script (chunked) for Hoffman2.
#
# Submit with:
#   qsub -t 1-10 projects/dl/infra/sge/submit_gpu.sh                              # PatchTST default, 10 chunks
#   qsub -t 1-20 -v EXPERIMENT=ae_ridge projects/dl/infra/sge/submit_gpu.sh        # AE+Ridge, 20 chunks
#   qsub -t 1-10 -v EXPERIMENT=patchts,GPU_COUNT=4 -l gpu,A100,cuda=4 projects/dl/infra/sge/submit_gpu.sh
#
# Supported GPUs: H200, H100, A100, A6000, V100, RTX2080Ti
# NOT supported:  P4 (4GB VRAM too small)
# ==============================================================

# --- SGE directives ---
#$ -cwd
#$ -j y
#$ -o /u/scratch/j/jamesdc1/
#$ -l gpu,A100,cuda=2
#$ -l h_data=16G,h_rt=21600
#$ -pe shared 8

# --- Configuration (override via -v or edit here) ---
EXPERIMENT="${EXPERIMENT:-patchts}"        # patchts | ae_ridge
INPUT_PATH="${INPUT_PATH:-all30min}"
RESULT_DIR="${RESULT_DIR:-}"              # empty = use config default

GPU_COUNT="${GPU_COUNT:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
TRAIN_WINDOW="${TRAIN_WINDOW:-}"
CONTEXT_LEN="${CONTEXT_LEN:-}"
PATCH_LEN="${PATCH_LEN:-}"
STRIDE="${STRIDE:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"            # ae_ridge only

# --- Chunking ---
if [ -n "$TOTAL_CHUNKS" ]; then
    TOTAL_CHUNKS="$TOTAL_CHUNKS"
else
    TOTAL_CHUNKS=1
fi
CHUNK_ID=$((SGE_TASK_ID - 1))            # Convert 1-based to 0-based

# Output file per chunk
if [ -n "$RESULT_DIR" ]; then
    mkdir -p "$RESULT_DIR"
    OUTPUT_FILE="$RESULT_DIR/results_chunk_${SGE_TASK_ID}.csv"
else
    OUTPUT_FILE=""
fi

# --- Environment Setup ---
echo "============================================"
echo "Job ID:     $JOB_ID"
echo "Array ID:   $SGE_TASK_ID"
echo "Hostname:   $(hostname)"
echo "Experiment: $EXPERIMENT"
echo "GPUs:       $GPU_COUNT"
echo "Chunk:      $CHUNK_ID / $TOTAL_CHUNKS"
echo "============================================"

source /u/local/Modules/default/init/modules.sh
module load conda cuda/12.3

# Activate DL conda environment (Python 3.11 + torch-cuda + transformers)
conda activate harxhar-dl

# Bind CPU threads to allocated cores
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# --- Build Command ---
CMD="python3 -m projects.dl.cli.gpu_executor --experiment $EXPERIMENT --input-path $INPUT_PATH --gpu-count $GPU_COUNT"
CMD="$CMD --chunk-id $CHUNK_ID --total-chunks $TOTAL_CHUNKS"

[ -n "$OUTPUT_FILE" ]   && CMD="$CMD --output $OUTPUT_FILE"
[ -n "$RESULT_DIR" ]    && CMD="$CMD --progress-path $RESULT_DIR/progress_chunk_${SGE_TASK_ID}.json"
[ -n "$BATCH_SIZE" ]    && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$EPOCHS" ]        && CMD="$CMD --epochs $EPOCHS"
[ -n "$LEARNING_RATE" ] && CMD="$CMD --learning-rate $LEARNING_RATE"
[ -n "$TRAIN_WINDOW" ]  && CMD="$CMD --train-window $TRAIN_WINDOW"
[ -n "$CONTEXT_LEN" ]   && CMD="$CMD --context-len $CONTEXT_LEN"
[ -n "$PATCH_LEN" ]     && CMD="$CMD --patch-len $PATCH_LEN"
[ -n "$STRIDE" ]        && CMD="$CMD --stride $STRIDE"
[ -n "$WEIGHTS_DIR" ]   && CMD="$CMD --weights-dir $WEIGHTS_DIR"

echo "Running: $CMD"
echo "============================================"

time $CMD

echo "Job finished."
