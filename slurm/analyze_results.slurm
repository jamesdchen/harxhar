#!/bin/bash

# ==============================================================
# Analyze scaling law results SLURM script.
#
# Submit with:
#   sbatch slurm/analyze_results.slurm
#   sbatch --export=RESULTS_DIR=my_results slurm/analyze_results.slurm
# ==============================================================

#SBATCH --job-name=analyze
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=main
#SBATCH --time=00:30:00

# --- CONFIGURATION ---
RESULTS_DIR="${RESULTS_DIR:-results_scaling_laws}"

# --- ENVIRONMENT SETUP ---
echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Hostname:    $(hostname)"
echo "Results dir: $RESULTS_DIR"
echo "============================================"

module purge
module load python

mkdir -p logs

python3 -m scripts.analyze_scaling --results-dir "$RESULTS_DIR"

echo "Job finished."
