import subprocess
import random
import numpy as np
import sys

# Configuration ranges from your sweep_config.yaml
LATENT_DIMS = [27, 23]
BETAS = [1,5]
EPOCHS = 60
# MIN_LR = 0.0001
# MAX_LR = 0.01
CROP_SETTINGS = 0.1
# How many total runs do you want to perform?
# NUM_RUNS =4
SCRIPT_NAME = "Medical_VAE_Clustering.py"
# def get_random_lr():
#     """Simulate log_uniform_values distribution"""
#     # Log uniform sampling: exp(uniform(log(min), log(max)))
#     return np.exp(np.random.uniform(np.log(MIN_LR), np.log(MAX_LR)))

print(f"Starting Offline Sweep for {len(BETAS)} betas...")

for i, beta in enumerate(BETAS):
    
    print(f"\n--- Run {i+1}/{len(BETAS)} ---")
    print(f"Setting: Beta = {beta}")
    print(f"Latent Dim = {LATENT_DIMS[i]}")

    latent_dim = LATENT_DIMS[i]
    # 2. Construct the command
    # matches arguments in train_sweep.py
    cmd = [
        sys.executable, SCRIPT_NAME,
        "--latent_dim", str(latent_dim),
        "--beta", str(beta),
        "--epochs", str(EPOCHS),
        "--crop_percent", str(CROP_SETTINGS)
    ]

    # 3. Execute the training script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run {i+1} failed with error: {e}")


print("\nAll offline runs completed.")
print("To upload results, run: wandb sync --include-offline wandb/offline-run-*")

# to run use nohup python offline_launcher_for_beta.py &> log &