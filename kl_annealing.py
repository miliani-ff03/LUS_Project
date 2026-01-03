
import wandb
import torch
import torch.optim as optim
import torchvision
from Medical_VAE_Clustering import (
    ConvVAE, get_dataloader, vae_loss_function, log_reconstruction, perform_clustering_and_log, log_loss_graphs, extract_latent_features, plot_latent_distributions, DEVICE, LossTracker
)

# KL Annealing Class
class KLAnnealer:
    def __init__(self, total_epochs, start_beta=0.0, end_beta=1.0):
        self.total_epochs = total_epochs
        self.warmup_epochs = total_epochs // 4  
        self.start_beta = start_beta
        self.end_beta = end_beta
    
    def get_beta(self, epoch):
        """Linear annealing from start_beta to end_beta"""
        if epoch < self.warmup_epochs:
            return self.start_beta + (self.end_beta - self.start_beta) * (epoch / self.warmup_epochs)
        else:
            return self.end_beta
       
      
       


latent_dim = 32
learning_rate = 1e-3
batch_size = 64
epochs = 50


def train(end_beta):
    # 2. Init WandB in Offline Mode with 'args'
    run = wandb.init(project="medical-vae-kl-annealing", config={
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "end_beta": end_beta
    }, mode="offline")
    config = wandb.config

    # 3. Use config parameters
    print(f"Running: Latent={config.latent_dim}, LR={config.learning_rate}")
    
    # --- DATA LOADING ---
    # COSMA paths are distinct, ensure this path is correct for the COMPUTE nodes
    data_path = "/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae"
    dataloader, dataset = get_dataloader(use_real_data=True, data_path=data_path, batch_size=config.batch_size)

    # --- MODEL SETUP ---
    model = ConvVAE(latent_dim=config.latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_batches = len(dataloader)
    tracker = LossTracker()
    kl_annealer = KLAnnealer(total_epochs=config.epochs, start_beta=0.0, end_beta=config.end_beta)

    best_loss = float('inf')
    best_epoch = 0
    best_model_path = f"best_vae_latent{config.latent_dim}_beta{config.end_beta}.pth"
    # --- TRAINING LOOP ---
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0
        
        current_beta = kl_annealer.get_beta(epoch)
        
        for data, _ in dataloader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon, data, mu, logvar, beta=current_beta)
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

        print(f"Epoch [{epoch+1}/{config.epochs}] - Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}, Beta: {current_beta:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> New best model saved with loss {best_loss:.4f}")

        # Log reconstructions every 10 epochs
        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     log_reconstruction(model, dataloader, epoch + 1)
       
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "recon_loss": avg_bce,
            "kl_loss": avg_kld,
            "beta": current_beta,
            "latent_dim": config.latent_dim
        })          

    print(f"Training completed. Best model saved at {best_model_path} (Loss: {best_loss:.4f})")
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))  
    model.eval()

    log_reconstruction(model, dataloader, config.latent_dim, config.end_beta)
    # 1. Log Clusters
    perform_clustering_and_log(model, dataloader, config.latent_dim, config.end_beta)
    # 2. Log Loss Curves
    log_loss_graphs(tracker, config.latent_dim)
    # 3. Plot Latent Distributions
    print("Extracting features for distribution plot...")
    X_latent, _ = extract_latent_features(model, dataloader)
    plot_latent_distributions(X_latent, config.latent_dim, epoch=config.epochs)

    run.finish()

if __name__ == "__main__":
    end_beta_values = [0.01, 0.1, 1.0, 2.0, 5.0, 10]
    for end_beta in end_beta_values:
        train(end_beta)


#  run using nohup python kl_annealing.py &> log &