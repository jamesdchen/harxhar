Monitor a running DL SLURM experiment and take corrective action.

## Context

You are monitoring a DL experiment on the USC CARC Discovery cluster (SLURM).
All SLURM commands require `--clusters=discovery`. scancel is intentionally blocked.
Working directory: /home1/jc_905/harxhar

## Arguments

$ARGUMENTS formats (pick one):

1. **Submit + monitor** (no job-ids → submits first):
   `<experiment> <total_chunks>` or `<experiment> <total_chunks> --result-dir <dir> [--batch-size N] [--epochs N] ...`
   Example: `patchts 10`

2. **Monitor existing** (job-ids provided):
   `<experiment> <result_dir> <job_ids> <total_chunks>`
   Example: `patchts results/dl_patchts 12345678,12345679 10`

3. **Auto-discover** (empty):
   Check lifecycle.jsonl files in results/dl_* dirs to find active experiments
   (look for "submitted" events without a corresponding "all_complete" or "abandoned" event).

## Step 0: Submit (if no job-ids provided)

If $ARGUMENTS has no job-ids (format 1), submit the experiment first:

```bash
cd /home1/jc_905/harxhar && python -m projects.dl.cli.lifecycle submit \
    --experiment <experiment> --total-chunks <total_chunks> [--result-dir <dir>] [pass-through flags]
```

This prints JSON with `{"job_ids": [...]}`. Parse the job IDs and result_dir (defaults to `results/dl_<experiment>`).
Then proceed to Step 1 with the captured job-ids.

## Step 1: Get Status

Run the status reporter:
```bash
cd /home1/jc_905/harxhar && python -m projects.dl.cli.lifecycle status \
    --result-dir <result_dir> --job-ids <job_ids> --total-chunks <total_chunks>
```

Parse the JSON output. Determine overall state:
- **all_complete**: all chunks have CSVs → go to Step 4 (aggregate)
- **still_running**: some chunks RUNNING/PENDING → report count, exit (next loop iteration checks again)
- **has_failures**: no active jobs, some chunks failed → go to Step 2
- **all_failed**: every chunk failed → go to Step 3 (triage carefully)

## Step 2: Diagnose Failures

For each failed chunk, read the .err log path from the status JSON:
```bash
tail -100 <err_log_path>
```

Classify the failure:

| Pattern | Category | Action |
|---------|----------|--------|
| `CUDA out of memory` or `OutOfMemoryError` | GPU OOM | Resubmit with more memory + smaller batch |
| sacct state = `OUT_OF_MEMORY` | System OOM | Resubmit with more memory |
| sacct state = `TIMEOUT` | Walltime exceeded | Resubmit with longer time |
| sacct state = `NODE_FAIL` or `CANCELLED` | Infra issue | Resubmit as-is |
| `Traceback` with a clear Python error | Code bug | **Do NOT resubmit. Report to user.** |
| Unrecognized error | Unknown | Read full log. If unclear, report to user, do NOT auto-resubmit. |

**IMPORTANT**: Never auto-resubmit code bugs or unrecognized errors. Stop and report.

## Step 3: Resubmit Failed Chunks

First, check retry count in lifecycle.jsonl:
```bash
grep '"resubmitted"' <result_dir>/lifecycle.jsonl | grep '"task_ids"'
```
Count how many times each chunk has been resubmitted. **Max 3 retries per chunk.** If exceeded, log "abandoned" and report to user.

Build the sbatch command:
```bash
sbatch --clusters=discovery \
    --array=<failed_task_ids> \
    --job-name dl_<experiment> \
    --account pollok_1603 \
    --output /scratch1/jc_905/logs/slurm-%A_%a.out \
    --error /scratch1/jc_905/logs/slurm-%A_%a.err \
    [resource overrides] \
    --export=ALL \
    projects/dl/infra/slurm/submit_gpu.slurm
```

### Resource overrides by failure type:

| Failure | Retry 1 | Retry 2+ |
|---------|---------|----------|
| GPU OOM | --mem=192G, set BATCH_SIZE=original/2 | --mem=256G, set BATCH_SIZE=original/4 |
| System OOM | --mem=192G | --mem=256G |
| Timeout | --time=10:00:00 | --time=14:00:00 |
| Node fail | no overrides | no overrides |

Set env vars:
```
--export=EXPERIMENT=<exp>,RESULT_DIR=<dir>,TOTAL_CHUNKS=<n>,INPUT_PATH=all30min
```

After sbatch, parse the job ID from stdout and log to lifecycle.jsonl:
```bash
python -c "
import json, time
entry = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'action': 'resubmitted',
         'experiment': '<experiment>', 'job_id': '<new_job_id>',
         'task_ids': [<failed_ids>], 'reason': '<category>', 'overrides': '<changes>'}
with open('<result_dir>/lifecycle.jsonl', 'a') as f:
    f.write(json.dumps(entry) + '\n')
"
```

**Update your job-ids list** for subsequent status checks to include the new job ID.

## Step 4: Aggregate

When all chunks complete:
```bash
cd /home1/jc_905/harxhar && python -m projects.ml.scripts.aggregate --base-dir <result_dir>
```

After aggregation:
1. Verify output files exist (ls for *summary*.csv or similar)
2. If found, read and report key metrics
3. Log "aggregated" event to lifecycle.jsonl
4. If aggregation fails, read stderr and report to user

## Step 5: Report

Always end with a concise summary:
- Chunks: X/Y complete, Z running, W failed
- Actions taken this iteration (if any)
- Next: waiting / needs attention / done

If fully aggregated, note that monitoring can be stopped (`/loop stop`).
