#!/bin/bash
#SBATCH --job-name=harx_single
#SBATCH --output=logs/single_%a.out
#SBATCH --error=logs/single_%a.err
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

# ==============================================================
# Single-experiment SLURM script.
# Edit the variables below, then submit:
#   sbatch run_single.slurm
# ==============================================================

# --- CONFIGURATION (edit these) ---
MODEL="ridge"
FEATURES="raw"                         # raw | har | pca | ae
EXOG_COLS=""                           # pipe-separated, or "" for none
RESULT_DIR="results_single"
TOTAL_CHUNKS=100
TRAIN_WINDOW=500
EXTRA_ARGS=""                          # e.g. "--save-coefs" or "--segment morning"

# --- SETUP ---
mkdir -p "$RESULT_DIR"
OUTPUT_FILE="${RESULT_DIR}/results_chunk_${SLURM_ARRAY_TASK_ID}.csv"

echo "=========================================="
echo "Model: ${MODEL}  Features: ${FEATURES}"
echo "Chunk: ${SLURM_ARRAY_TASK_ID} / ${TOTAL_CHUNKS}"
echo "=========================================="

ml python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- RUN ---
python3 -m src.cli.executor \
    --model "$MODEL" \
    --features "$FEATURES" \
    --output-file "$OUTPUT_FILE" \
    --chunk-id "$((SLURM_ARRAY_TASK_ID - 1))" \
    --total-chunks "$TOTAL_CHUNKS" \
    --train-window "$TRAIN_WINDOW" \
    ${EXOG_COLS:+--exog-cols "$EXOG_COLS"} \
    $EXTRA_ARGS

echo "Done."
