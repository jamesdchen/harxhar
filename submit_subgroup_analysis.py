import os
import subprocess
import numpy as np

# --- CONFIGURATION ---
TOTAL_CHUNKS = 100
TASKS_PER_ARRAY = 100
BASE_RESULT_DIR = "results_ridge_subgroups"
SUBMISSION_SCRIPT = "submit_carc.slurm"

# Define the models we want to iterate over
MODELS_TO_RUN = ["ridge", "xgboost"]

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

# --- 2. DEFINE SUBGROUPS ---
SUBGROUPS = {
    "baseline": [],
    "moments": [f for f in ALL_FEATURES if f.startswith("sum") and "stock" not in f and "volume" not in f],
    "liquidity": [f for f in ALL_FEATURES if any(x in f for x in ["volume", "turnover", "spread"])],
    "market_ew": [f for f in ALL_FEATURES if "ewstock" in f],
    "market_vw": [f for f in ALL_FEATURES if "vwstock" in f],
    "sentiment": [f for f in ALL_FEATURES if "stocktwits" in f],
    "macro_vol": [f for f in ALL_FEATURES if "vix" in f],
    "vol_demand": [f for f in ALL_FEATURES if "voldemand" in f],
    "all_features": ALL_FEATURES
}

def main():
    total_experiments = len(SUBGROUPS) * len(MODELS_TO_RUN)
    print(f"Generating Experiments for {len(SUBGROUPS)} Subgroups x {len(MODELS_TO_RUN)} Models + 1 Naive Baseline...")
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    
    # ==========================================
    # --- SUBMIT NAIVE BASELINE FIRST ---
    # ==========================================
    naive_dir = os.path.abspath(os.path.join(BASE_RESULT_DIR, "exp_0_naive_baseline"))
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
    # --- SUBMIT ML MODELS (RIDGE & XGBOOST) ---
    # ==========================================
    exp_id = 1
    for model_type in MODELS_TO_RUN:
        for exp_name, variables in SUBGROUPS.items():
            
            exog_str = "|".join(variables) if variables else "None"
            
            # Make the directory name distinct so models don't overwrite each other
            exp_dir = os.path.abspath(os.path.join(BASE_RESULT_DIR, f"exp_{exp_id}_{model_type}_{exp_name}"))
            os.makedirs(exp_dir, exist_ok=True)
            
            with open(os.path.join(exp_dir, "config.txt"), "w") as f:
                f.write(f"Experiment ID: {exp_id}\n")
                f.write(f"Experiment Name: {exp_name}\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Num Variables: {len(variables)}\n")
                f.write(f"Variables: {variables}\n")
                
            print(f"--- Submitting ID {exp_id}: {model_type.upper()} - {exp_name.upper()} ({len(variables)} vars) ---")
            
            start_task = 1
            while start_task <= TOTAL_CHUNKS:
                end_task = min(start_task + TASKS_PER_ARRAY - 1, TOTAL_CHUNKS)
                task_range = f"{start_task}-{end_task}"
                
                # Dynamic job names: rdg_1, xgb_10, etc.
                short_model_name = "xgb" if model_type == "xgboost" else "rdg"
                job_name = f"{short_model_name}_{exp_id}"
                
                job_env = os.environ.copy()
                job_env["TOTAL_CHUNKS"] = str(TOTAL_CHUNKS)
                job_env["EXOG_COLS"] = exog_str
                job_env["RESULT_DIR"] = exp_dir
                job_env["MODEL_TYPE"] = model_type 
                
                cmd = [
                    "sbatch",
                    "--array", task_range,
                    "--job-name", job_name,
                    SUBMISSION_SCRIPT
                ]
                
                subprocess.run(cmd, env=job_env)
                start_task = end_task + 1
                
            exp_id += 1

    print(f"\n✅ All {total_experiments} experiments + Naive baseline submitted to {BASE_RESULT_DIR}.")

if __name__ == "__main__":
    main()