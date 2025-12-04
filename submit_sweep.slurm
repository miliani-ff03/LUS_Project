#!/bin/bash
#SBATCH --job-name=vae_sweep
#SBATCH --output=logs/run_%a.out
#SBATCH --error=logs/run_%a.err
#SBATCH --time=02:00:00
#SBATCH --array=1-12%4        # Run lines 1-12 from params.txt, max 4 at a time
#SBATCH --partition=dine2     # <--- CHANGE THIS IF NEEDED (e.g. cosma8-shm)
#SBATCH --account=do015       # <--- REQUIRED: Project code for dine2 (or do016 for cosma8)
#SBATCH --gres=gpu:1          # Request 1 GPU

# Load modules (Adjust based on your COSMA environment)
module purge
module load python/3.10.12
# source ~/your_virtual_env/bin/activate 

# Read the specific parameters for THIS job ID
PARAM_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" params.txt)
read -r LD BETA LR <<< "$PARAM_LINE"

echo "Running on $(hostname) with Params: LD=$LD Beta=$BETA LR=$LR"

# Run Python
python train_sweep.py \
    --latent_dim "$LD" \
    --beta "$BETA" \
    --learning_rate "$LR" \
    --epochs 15