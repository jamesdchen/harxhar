import os
import subprocess
import numpy as np

# --- CONFIGURATION ---
TOTAL_CHUNKS = 100
TASKS_PER_ARRAY = 100
BASE_RESULT_DIR = "results_ridge_individual_moments"
SUBMISSION_SCRIPT = "submit_carc.slurm"

# --- 1. DEFINE THE FEATURE UNIVERSE ---
FULL_FEATURE_STRING = (
    "endbartime|sumret|sumabsret|sumret3|sumret4|sumpret2|sumbipow|sumautocov|sumvolume|numobs|"
    "sumret2_ewstock|sumret3_ewstock|sumret4_ewstock|sumabsret_ewstock|sumbipow_ewstock|sumpret2_ewstock|"
    "turnover_ewstock|buyturnover_ewstock|sellturnover_ewstock|effspread_ewstock|spread_ewstock|"
    "sumret2_vwstock|sumret3_vwstock|sumret4_vwstock|sumabsret_vwstock|sumbipow_vwstock|sumpret2_vwstock|"
    "turnover_vwstock|buyturnover_vwstock|sellturnover_vwstock|effspread_vwstock|spread_vwstock|"
    "turnover_spy|buyturnover_spy|sellturnover_spy|"
    "stocktwits_attention|stocktwits_sentiment|stocktwits_sentcount|"
    "vix|vvix|vix3m|"
    "voldemand_spx_open_and_close|voldemand_spx_open_only|voldemand_all_open_and_close|voldemand_all_open_only|"
    "hour|DOW"
)

ALL_FEATURES = FULL_FEATURE_STRING.split("|")

# --- 2. ISOLATE INDIVIDUAL MOMENT FEATURES ---
# Extract moments using your original subgroup logic
MOMENT_FEATURES = [f for f in ALL_FEATURES if f.startswith("sum") and "stock" not in f and "volume" not in f]

def main():
    print(f"Generating Experiment for {len(MOMENT_FEATURES)} Individual Moments + 1 Naive Baseline...")
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    
    # ==========================================
    # --- SUBMIT NAIVE BASELINE FIRST (ID 0) ---
    # ==========================================
    naive_dir = os.path.abspath(os.path.join(BASE_RESULT_DIR, "exp_naive"))
    os.makedirs(naive_dir, exist_ok=True)
    
    with open(os.path.join(naive_dir, "config.txt"), "w") as f:
        f.write("Experiment ID: 0\n")
        f.write("Experiment Name: naive_baseline\n")
        f.write("Model Type: naive\n")
        f.write("Num Variables: 0\n")
        f.write("Variables: []\n")
        
    print(f"--- Submitting ID 0: NAIVE_BASELINE ---")
    
    start_task = 1
    while start_task <= TOTAL_CHUNKS:
        end_task = min(start_task + TASKS_PER_ARRAY - 1, TOTAL_CHUNKS)
        task_range = f"{start_task}-{end_task}"
        
        job_env = os.environ.copy()
        job_env["TOTAL_CHUNKS"] = str(TOTAL_CHUNKS)
        job_env["EXOG_COLS"] = "None" 
        job_env["RESULT_DIR"] = naive_dir
        job_env["MODEL_TYPE"] = "naive"
        
        cmd = [
            "sbatch",
            "--array", task_range,
            "--job-name", "naive_0",
            SUBMISSION_SCRIPT
        ]
        
        subprocess.run(cmd, env=job_env)
        start_task = end_task + 1

    # ==========================================
    # --- SUBMIT INDIVIDUAL MOMENTS ---
    # ==========================================
    for i, feature in enumerate(MOMENT_FEATURES, start=1):
        
        exp_name = f"moment_{feature}"
        variables = [feature]
        exog_str = feature # Since it's just one feature, no need to join with '|'
        
        # Directory setup
        exp_dir = os.path.abspath(os.path.join(BASE_RESULT_DIR, f"exp_{i}"))
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(exp_dir, "config.txt"), "w") as f:
            f.write(f"Experiment ID: {i}\n")
            f.write(f"Experiment Name: {exp_name}\n")
            f.write(f"Model Type: ridge\n")
            f.write(f"Num Variables: 1\n")
            f.write(f"Variables: {variables}\n")
            
        print(f"--- Submitting ID {i}: {exp_name.upper()} ---")
        
        start_task = 1
        while start_task <= TOTAL_CHUNKS:
            end_task = min(start_task + TASKS_PER_ARRAY - 1, TOTAL_CHUNKS)
            task_range = f"{start_task}-{end_task}"
            
            job_name = f"enet_{i}"
            
            job_env = os.environ.copy()
            job_env["TOTAL_CHUNKS"] = str(TOTAL_CHUNKS)
            job_env["EXOG_COLS"] = exog_str
            job_env["RESULT_DIR"] = exp_dir
            job_env["MODEL_TYPE"] = "ridge" 
            
            cmd = [
                "sbatch",
                "--array", task_range,
                "--job-name", job_name,
                SUBMISSION_SCRIPT
            ]
            
            subprocess.run(cmd, env=job_env)
            start_task = end_task + 1

    print(f"\n✅ All {len(MOMENT_FEATURES)} experiments + Naive Baseline submitted to {BASE_RESULT_DIR}.")

if __name__ == "__main__":
    main()