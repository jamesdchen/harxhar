import os
import subprocess
import numpy as np

# --- CONFIGURATION ---
TOTAL_CHUNKS = 100
TASKS_PER_ARRAY = 100
BASE_RESULT_DIR = "results_elasticnet_subgroups"
SUBMISSION_SCRIPT = "submit_carc.slurm"

# --- 1. DEFINE THE FEATURE UNIVERSE ---
FULL_FEATURE_STRING = (
    "endbartime|sumret|sumabsret|sumret2|sumret3|sumret4|sumpret2|sumbipow|sumautocov|sumvolume|numobs|"
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
    "microstructure": [f for f in ALL_FEATURES if f.startswith("sum") and "stock" not in f and "volume" not in f],
    "liquidity": [f for f in ALL_FEATURES if any(x in f for x in ["volume", "turnover", "spread"])],
    "market_ew": [f for f in ALL_FEATURES if "ewstock" in f],
    "market_vw": [f for f in ALL_FEATURES if "vwstock" in f],
    "sentiment": [f for f in ALL_FEATURES if "stocktwits" in f],
    "macro_vol": [f for f in ALL_FEATURES if "vix" in f],
    "vol_demand": [f for f in ALL_FEATURES if "voldemand" in f],
    "all_features": ALL_FEATURES
}

def main():
    print(f"Generating Experiment for {len(SUBGROUPS)} Subgroups...")
    
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    
    # Loop through each Subgroup with an INDEX
    for i, (exp_name, variables) in enumerate(SUBGROUPS.items(), start=1):
        
        # --- PREPARE EXOG STRING ---
        exog_str = "|".join(variables)
        
        # Directory setup (Indexed: exp_1, exp_2, etc.)
        exp_dir = os.path.abspath(os.path.join(BASE_RESULT_DIR, f"exp_{i}"))
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config (Important: Map the ID back to the Name here)
        with open(os.path.join(exp_dir, "config.txt"), "w") as f:
            f.write(f"Experiment ID: {i}\n")
            f.write(f"Experiment Name: {exp_name}\n")
            f.write(f"Model Type: elasticnet\n")
            f.write(f"Num Variables: {len(variables)}\n")
            f.write(f"Variables: {variables}\n")
            
        print(f"--- Submitting ID {i}: {exp_name.upper()} ({len(variables)} vars) ---")
        
        # --- SUBMIT ---
        start_task = 1
        while start_task <= TOTAL_CHUNKS:
            end_task = min(start_task + TASKS_PER_ARRAY - 1, TOTAL_CHUNKS)
            task_range = f"{start_task}-{end_task}"
            
            # Use ID in job name for shortness
            job_name = f"enet_{i}"
            
            job_env = os.environ.copy()
            job_env["TOTAL_CHUNKS"] = str(TOTAL_CHUNKS)
            job_env["EXOG_COLS"] = exog_str
            job_env["RESULT_DIR"] = exp_dir
            job_env["MODEL_TYPE"] = "elasticnet" 
            
            cmd = [
                "sbatch",
                "--array", task_range,
                "--job-name", job_name,
                SUBMISSION_SCRIPT
            ]
            
            subprocess.run(cmd, env=job_env)
            start_task = end_task + 1

    print(f"\n✅ All {len(SUBGROUPS)} experiments submitted to {BASE_RESULT_DIR}.")

if __name__ == "__main__":
    main()