#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --nodes=1
#SBATCH --ntasks=1               # Run a single process (correct for most Python scripts)
#SBATCH --cpus-per-task=1        # CPU cores per task. (See note below*)
#SBATCH --mem=64G                # Memory per node
#SBATCH --partition=main         # 'main' partition on discovery
#SBATCH --time=4:00:00

# *Note on CPUs: If your script uses heavy Numpy/Pandas matrix math, 
# you can increase --cpus-per-task to 4 or 8 to speed it up.

# --- Environment Setup ---
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"

# 1. Load Modules
module purge
module load python        # Adjust version if needed

# 2. Bind Python Threading to Slurm CPUs
# This prevents Python from trying to use cores it doesn't have permission to touch.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3. Check Variables (Passed from the python submission script)
# We set defaults just in case you run this manually for testing.
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

# 4. Prepare Output
mkdir -p "$RESULT_DIR"
OUTPUT_FILE="$RESULT_DIR/results_chunk_${SLURM_ARRAY_TASK_ID}.csv"

echo "Processing Chunk: $SLURM_ARRAY_TASK_ID"
echo "Output File: $OUTPUT_FILE"
echo "Model Type: $MODEL_TYPE"
echo "Exog Vars: $EXOG_COLS"

# --- Run the Python Script ---
# Note: SLURM_ARRAY_TASK_ID is usually 1-based (1..100).
# Python lists are 0-based (0..99). We subtract 1 here.

time python3 -m projects.ml.cli.executor \
    --model "$MODEL_TYPE" \
    --output-file "$OUTPUT_FILE" \
    --chunk-id $((SLURM_ARRAY_TASK_ID-1)) \
    --total-chunks $TOTAL_CHUNKS \
    --exog-cols "$EXOG_COLS" \
    ${EXTRA_ARGS:-}

echo "Job finished successfully."