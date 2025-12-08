import argparse
import wandb
import torch
import torch.optim as optim
import torchvision
from Medical_VAE_Clustering import (
    ConvVAE, get_dataloader, vae_loss_function, log_reconstruction, perform_clustering_and_log, log_loss_graphs, DEVICE, LossTracker
)

# 1. Setup Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=32)
parser.add_argument("--beta", type=float, default=5)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

def train():
    # 2. Init WandB in Offline Mode with 'args'
    run = wandb.init(project="medical-vae-sweep", config=args, mode="offline")
    config = wandb.config

    # 3. Use config parameters
    print(f"Running: Latent={config.latent_dim}, Beta={config.beta}, LR={config.learning_rate}")
    
    # --- DATA LOADING ---
    # COSMA paths are distinct, ensure this path is correct for the COMPUTE nodes
    data_path = "/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae"
    dataloader, dataset = get_dataloader(use_real_data=True, data_path=data_path, batch_size=config.batch_size)

    # --- MODEL SETUP ---
    model = ConvVAE(latent_dim=config.latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_batches = len(dataloader)
    tracker = LossTracker()

    # --- TRAINING LOOP ---
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0
        
        for data, _ in dataloader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon, data, mu, logvar, beta=config.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        # Correct averaging: divide by number of batches
        avg_loss = total_loss / num_batches
        avg_bce = total_bce / num_batches
        avg_kld = total_kld / num_batches
        tracker.add(avg_loss, avg_bce, avg_kld)
        
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "recon_loss": avg_bce,
            "kl_loss": avg_kld,
            "latent_dim": config.latent_dim
        })
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    # 1. Log Clusters
    perform_clustering_and_log(model, dataloader, config.latent_dim)
    
    # 2. Log Reconstructions (NEW)
    log_reconstruction(model, dataloader, config.latent_dim)

    # 3. Log Loss Curves
    log_loss_graphs(tracker, config.latent_dim)

    run.finish()

if __name__ == "__main__":
    train()