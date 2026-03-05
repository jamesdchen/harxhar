#!/bin/bash
#SBATCH --job-name=harx_all
#SBATCH --output=logs/all_model_%a.out
#SBATCH --error=logs/all_model_%a.err
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

# --- CONFIGURATION ---
BEST_FEATURES="endbartime|sumret|sumabsret|sumret2|sumret3|sumret4|sumpret2|sumbipow|sumautocov|sumvolume|numobs|sumret2_ewstock|sumret3_ewstock|sumret4_ewstock|sumabsret_ewstock|sumbipow_ewstock|sumpret2_ewstock|turnover_ewstock|buyturnover_ewstock|sellturnover_ewstock|effspread_ewstock|spread_ewstock|sumret2_vwstock|sumret3_vwstock|sumret4_vwstock|sumabsret_vwstock|sumbipow_vwstock|sumpret2_vwstock|turnover_vwstock|buyturnover_vwstock|sellturnover_vwstock|effspread_vwstock|spread_vwstock|turnover_spy|buyturnover_spy|sellturnover_spy|stocktwits_attention|stocktwits_sentiment|stocktwits_sentcount|vix|vvix|vix3m|voldemand_spx_open_and_close|voldemand_spx_open_only|voldemand_all_open_and_close|voldemand_all_open_only|hour|DOW"

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