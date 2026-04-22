#!/bin/bash
# SGE CPU Array Job Template with TASK_OFFSET support
#
# Submit with:
#   qsub -t 1-100 -v TASK_OFFSET=0,MODULES=...,EXECUTOR=...,... _cpu_array.sh

#$ -cwd
#$ -j y
#$ -l h_data=16G
#$ -l h_rt=10:00:00

set -e

echo "============================================"
echo "Job ID:       $JOB_ID"
echo "Array Task:   $SGE_TASK_ID"
echo "Hostname:     $(hostname)"
echo "============================================"

RESULT_DIR="${RESULT_DIR:-.}"
REPO_DIR="${REPO_DIR:-.}"

# Module setup (Hoffman2)
if [ -f /u/local/Modules/default/init/modules.sh ]; then
    source /u/local/Modules/default/init/modules.sh
fi
if [ -n "$MODULES" ]; then
    for mod in $MODULES; do
        module load "$mod"
    done
fi

# Conda
if [ -n "$CONDA_SOURCE" ]; then
    source "$CONDA_SOURCE"
fi
if [ -n "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Convert 1-based SGE_TASK_ID to 0-based, add offset for batched submission
TASK_ID=$((SGE_TASK_ID - 1 + ${TASK_OFFSET:-0}))

# Per-task manifest lookup — re-derive RESULT_DIR + CHUNK_ID from the manifest.
# The submitted job env carries a single global RESULT_DIR; without this lookup
# every task in the array would write into that one dir using the global
# TASK_ID as the chunk filename (cross-trial chunk leakage).
MANIFEST_PATH="${HPC_MANIFEST:-_hpc_dispatch.json}"
read RESULT_DIR CHUNK_ID < <(python3 -c "
import json
t = json.load(open('${MANIFEST_PATH}'))['tasks'][str(${TASK_ID})]
print(t['result_dir'], t['chunk_id'])
")
export RESULT_DIR CHUNK_ID
mkdir -p "$RESULT_DIR"

echo "Task:         $TASK_ID (offset=${TASK_OFFSET:-0})"
echo "Chunk ID:     $CHUNK_ID"
echo "Result dir:   $RESULT_DIR"
echo "Executor:     $EXECUTOR"
echo "============================================"

export TASK_ID
time $EXECUTOR ${EXTRA_ARGS:-}

echo "Job finished."
