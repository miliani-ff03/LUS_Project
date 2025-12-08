import subprocess
import random
import numpy as np
import sys

# Configuration ranges from your sweep_config.yaml
LATENT_DIMS = 64
BETAS = [0.0001, 0.001, 0.01, 0.1,  1, 5, 10, 50, 100, 1000, 10000]
MIN_LR = 0.0001
MAX_LR = 0.01

# How many total runs do you want to perform?
NUM_RUNS =11

def get_random_lr():
    """Simulate log_uniform_values distribution"""
    # Log uniform sampling: exp(uniform(log(min), log(max)))
    return np.exp(np.random.uniform(np.log(MIN_LR), np.log(MAX_LR)))

print(f"Starting Offline Sweep for {NUM_RUNS} runs...")
b=0
for i in range(NUM_RUNS):
    # 1. Sample Hyperparameters
    # latent_dim = random.choice(LATENT_DIMS)
    beta = BETAS[b]
    lr = get_random_lr()
    
    print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
    print(f"Params: Beta={beta}, LR={lr:.6f}")

    # 2. Construct the command
    # matches arguments in train_sweep.py
    cmd = [
        sys.executable, "train_sweep.py",
        "--latent_dim", "64",
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

    b+=1

print("\nAll offline runs completed.")
print("To upload results, run: wandb sync --include-offline wandb/offline-run-*")

# to run use nohup python offline_launcher.py &> log &