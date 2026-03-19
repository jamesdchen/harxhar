#!/bin/bash

# --- SGE (qsub) options ---
#$ -cwd
#$ -j y
#$ -l h_data=16G

# --- Environment Setup ---
echo "Job ID: $JOB_ID"
echo "Array Task ID: $SGE_TASK_ID"
echo "Hostname: $(hostname)"

# 1. Load Modules (Hoffman2)
export PATH=$PATH:/u/systems/UGE8.6.4/bin/lx-amd64
source /u/local/Modules/default/init/modules.sh
ml python
ml gcc

# 2. Check Variables (Passed from the python submission script via -v)
if [ -z "$RESULT_DIR" ]; then
    echo "WARNING: RESULT_DIR not set. Defaulting to current directory."
    RESULT_DIR="."
fi

if [ -z "$TOTAL_CHUNKS" ]; then
    TOTAL_CHUNKS=1
fi

if [ -z "$EXOG_COLS" ]; then
    EXOG_COLS="NONE"
fi

if [ -z "$MODEL_TYPE" ]; then
    echo "WARNING: MODEL_TYPE not set. Defaulting to naive."
    MODEL_TYPE="naive"
fi

# 3. Prepare Output
mkdir -p "$RESULT_DIR"
OUTPUT_FILE="$RESULT_DIR/results_chunk_${SGE_TASK_ID}.csv"

echo "Processing Chunk: $SGE_TASK_ID"
echo "Output File: $OUTPUT_FILE"
echo "Model Type: $MODEL_TYPE"
echo "Exog Vars: $EXOG_COLS"

# --- Run the Python Script ---
# SGE_TASK_ID is 1-based (1..100).
# Python lists are 0-based (0..99). We subtract 1.

time python3 -m src.cli.executor \
    --model "$MODEL_TYPE" \
    --output-file "$OUTPUT_FILE" \
    --chunk-id $((SGE_TASK_ID-1)) \
    --total-chunks $TOTAL_CHUNKS \
    --exog-cols "$EXOG_COLS" \
    ${EXTRA_ARGS:-}

echo "Job finished successfully."
