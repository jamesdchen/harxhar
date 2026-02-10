#!/bin/bash
#SBATCH --job-name=harx_best
#SBATCH --output=logs/best_model_%a.out
#SBATCH --error=logs/best_model_%a.err
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

# --- CONFIGURATION ---
# The "Four Horsemen" of Realized Volatility
# We drop Volume, Sentiment, and VolDemand entirely.
BEST_FEATURES=""#"sumabsret|sumpret2|sumbipow"

# Output Directory
RESULT_DIR="results_best_model"
mkdir -p $RESULT_DIR

# Script Parameters
TOTAL_CHUNKS=100
TRAIN_WINDOW=500
OUTPUT_FILE="${RESULT_DIR}/results_chunk_${SLURM_ARRAY_TASK_ID}.csv"

echo "=========================================="
echo "Running Best Model (Exp 6 Config)"
echo "Chunk: ${SLURM_ARRAY_TASK_ID} / ${TOTAL_CHUNKS}"
echo "Features: ${BEST_FEATURES}"
echo "=========================================="

ml python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run Python Script
python harx_coef.py \
    --output-file "$OUTPUT_FILE" \
    --chunk-id $SLURM_ARRAY_TASK_ID \
    --total-chunks $TOTAL_CHUNKS \
    --train-window $TRAIN_WINDOW \
    --exog-cols "$BEST_FEATURES"

echo "Done."