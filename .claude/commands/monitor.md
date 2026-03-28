Monitor a running DL SGE experiment on Hoffman2 via SSH and take corrective action.

## Context

You are monitoring a DL experiment on UCLA Hoffman2 (SGE scheduler).
All cluster commands run remotely via `ssh jamesdc1@hoffman2.idre.ucla.edu`.
Remote repo: `$HPC_REPO` (default `/u/project/project-cucuringu/harxhar`)

## Arguments

$ARGUMENTS formats (pick one):

1. **Submit + monitor** (no job-ids -> submits first):
   `<experiment> <total_chunks>` or `<experiment> <total_chunks> --result-dir <dir> [--batch-size N] [--epochs N] ...`
   Example: `patchts 10`

2. **Monitor existing** (job-ids provided):
   `<experiment> <result_dir> <job_ids> <total_chunks>`
   Example: `patchts results/dl_patchts 12345678,12345679 10`

3. **Auto-discover** (empty):
   Check lifecycle.jsonl files in results/dl_* dirs to find active experiments
   (look for "submitted" events without a corresponding "all_complete" or "abandoned" event).

## Step 0: Submit (if no job-ids provided)

If $ARGUMENTS has no job-ids (format 1), sync code and submit the experiment first:

```bash
# Sync code first
rsync -az --delete \
    --exclude='.git/' --exclude='results/' --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='.mypy_cache/' --exclude='all30min/' \
    --exclude='.claude/' \
    . jamesdc1@hoffman2.idre.ucla.edu:$HPC_REPO/

# Submit
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python -m projects.dl.cli.lifecycle submit \
    --experiment <experiment> --total-chunks <total_chunks> [--result-dir <dir>] [pass-through flags]"
```

This prints JSON with `{"job_ids": [...]}`. Parse the job IDs and result_dir (defaults to `results/dl_<experiment>`).
Then proceed to Step 1 with the captured job-ids.

## Step 1: Get Status

Run the status reporter via SSH:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python -m projects.dl.cli.lifecycle status \
    --result-dir <result_dir> --job-ids <job_ids> --total-chunks <total_chunks>"
```

Parse the JSON output. Determine overall state:
- **all_complete**: all chunks have CSVs -> go to Step 4 (aggregate)
- **still_running**: some chunks RUNNING/PENDING -> report count, exit (next loop iteration checks again)
- **has_failures**: no active jobs, some chunks failed -> go to Step 2
- **all_failed**: every chunk failed -> go to Step 3 (triage carefully)

Alternative quick check:
```bash
# Check SGE queue for running jobs
ssh jamesdc1@hoffman2.idre.ucla.edu "qstat -u jamesdc1"

# Count completed chunks
ssh jamesdc1@hoffman2.idre.ucla.edu "ls $HPC_REPO/<result_dir>/results_chunk_*.csv 2>/dev/null | wc -l"
```

## Step 2: Diagnose Failures

For each failed chunk, read the log via SSH:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "tail -100 $HPC_REPO/logs/<job_name>.o<JOBID>.<TASKID>"
```

Or check job accounting:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "qacct -j <JOBID> -t <TASKID>"
```

Classify the failure:

| Pattern | Category | Action |
|---------|----------|--------|
| `CUDA out of memory` or `OutOfMemoryError` | GPU OOM | Resubmit with more memory + smaller batch |
| `exit_status` != 0 + high `maxvmem` | System OOM | Resubmit with `-l h_vmem=96G` |
| `failed` = 100 (time limit) | Walltime exceeded | Resubmit with `-l h_rt=14400` |
| Node failure / `Eqw` state | Infra issue | Resubmit as-is |
| `Traceback` with a clear Python error | Code bug | **Do NOT resubmit. Report to user.** |
| Unrecognized error | Unknown | Read full log. If unclear, report to user, do NOT auto-resubmit. |

**IMPORTANT**: Never auto-resubmit code bugs or unrecognized errors. Stop and report.

## Step 3: Resubmit Failed Chunks

First, check retry count in lifecycle.jsonl:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "grep '\"resubmitted\"' $HPC_REPO/<result_dir>/lifecycle.jsonl" 2>/dev/null
```
Count how many times each chunk has been resubmitted. **Max 3 retries per chunk.** If exceeded, log "abandoned" and report to user.

Build the qsub resubmit command:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && qsub -t <failed_task_ids> \
    -N dl_<experiment> \
    -o logs -j y \
    [-l h_vmem=96G] [-l h_rt=14400] \
    -v EXPERIMENT=<exp>,RESULT_DIR=<dir>,TOTAL_CHUNKS=<n>,INPUT_PATH=all30min \
    projects/dl/infra/sge/submit_gpu.sh"
```

### Resource overrides by failure type:

| Failure | Retry 1 | Retry 2+ |
|---------|---------|----------|
| GPU OOM | h_vmem=192G, BATCH_SIZE=original/2 | h_vmem=256G, BATCH_SIZE=original/4 |
| System OOM | h_vmem=192G | h_vmem=256G |
| Timeout | h_rt=36000 (10hrs) | h_rt=50400 (14hrs) |
| Node fail | no overrides | no overrides |

After qsub, parse the job ID from stdout and log to lifecycle.jsonl:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python -c \"
import json, time
entry = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'action': 'resubmitted',
         'experiment': '<experiment>', 'job_id': '<new_job_id>',
         'task_ids': [<failed_ids>], 'reason': '<category>', 'overrides': '<changes>'}
with open('<result_dir>/lifecycle.jsonl', 'a') as f:
    f.write(json.dumps(entry) + '\n')
\""
```

**Update your job-ids list** for subsequent status checks to include the new job ID.

## Step 4: Aggregate

When all chunks complete, run aggregation on the cluster:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python -m projects.ml.scripts.aggregate --base-dir <result_dir>"
```

After aggregation:
1. Verify output files exist:
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "ls $HPC_REPO/<result_dir>/*summary*.csv 2>/dev/null"
   ```
2. Download summaries locally:
   ```bash
   rsync -az \
       --include='*/' --include='*_summary*.csv' --include='metadata.json' \
       --include='config.txt' --include='lifecycle.jsonl' --exclude='*' \
       jamesdc1@hoffman2.idre.ucla.edu:$HPC_REPO/<result_dir>/ ./<result_dir>/
   ```
3. Read and report key metrics from the local summary CSV
4. Log "aggregated" event to lifecycle.jsonl via SSH
5. If aggregation fails, read stderr and report to user

## Step 5: Report

Always end with a concise summary:
- Chunks: X/Y complete, Z running, W failed
- Actions taken this iteration (if any)
- Next: waiting / needs attention / done

If fully aggregated, note that monitoring can be stopped (`/loop stop`).
