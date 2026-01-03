import subprocess
import random
import numpy as np
import sys

# Configuration ranges from your sweep_config.yaml
LATENT_DIMS = 32
BETAS = 1
EPOCHS = 100
# MIN_LR = 0.0001
# MAX_LR = 0.01
CROP_SETTINGS = [0.0, 0.1, 0.2, 0.3]
# How many total runs do you want to perform?
# NUM_RUNS =4
SCRIPT_NAME = "Medical_VAE_Clustering.py"
# def get_random_lr():
#     """Simulate log_uniform_values distribution"""
#     # Log uniform sampling: exp(uniform(log(min), log(max)))
#     return np.exp(np.random.uniform(np.log(MIN_LR), np.log(MAX_LR)))

print(f"Starting Offline Sweep for {len(CROP_SETTINGS)} crop settings...")

for i, crop in enumerate(CROP_SETTINGS):
    
    print(f"\n--- Run {i+1}/{len(CROP_SETTINGS)} ---")
    print(f"Setting: Crop Top={crop*100}%")

    # 2. Construct the command
    # matches arguments in train_sweep.py
    cmd = [
        sys.executable, SCRIPT_NAME,
        "--latent_dim", str(LATENT_DIMS),
        "--beta", str(BETAS),
        "--epochs", str(EPOCHS),
        "--crop_percent", str(crop)
    ]

    # 3. Execute the training script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run {i+1} failed with error: {e}")


print("\nAll offline runs completed.")
print("To upload results, run: wandb sync --include-offline wandb/offline-run-*")

# to run use nohup python offline_launcher_for_crop_sweep.py &> log &