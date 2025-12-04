import subprocess
import random
import numpy as np
import sys

# Configuration ranges from your sweep_config.yaml
LATENT_DIMS = [16, 24, 32, 48, 64]
BETAS = [1, 2, 5, 10]
MIN_LR = 0.0001
MAX_LR = 0.01

# How many total runs do you want to perform?
NUM_RUNS = 20 

def get_random_lr():
    """Simulate log_uniform_values distribution"""
    # Log uniform sampling: exp(uniform(log(min), log(max)))
    return np.exp(np.random.uniform(np.log(MIN_LR), np.log(MAX_LR)))

print(f"Starting Offline Sweep for {NUM_RUNS} runs...")

for i in range(NUM_RUNS):
    # 1. Sample Hyperparameters
    latent_dim = random.choice(LATENT_DIMS)
    beta = random.choice(BETAS)
    lr = get_random_lr()
    
    print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
    print(f"Params: Latent={latent_dim}, Beta={beta}, LR={lr:.6f}")

    # 2. Construct the command
    # matches arguments in train_sweep.py
    cmd = [
        sys.executable, "train_sweep.py",
        "--latent_dim", str(latent_dim),
        "--beta", str(beta),
        "--learning_rate", str(lr),
        "--batch_size", "64",
        "--epochs", "10"
    ]

    # 3. Execute the training script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run {i+1} failed with error: {e}")

print("\nAll offline runs completed.")
print("To upload results, run: wandb sync --include-offline wandb/offline-run-*")