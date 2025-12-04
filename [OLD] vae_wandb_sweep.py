# for wandb agent sweep
# wandb sync /cosma/home/durham/dc-fras4/code/wandb/offline-run-*

# %%
import os
import torch
import torch.optim as optim
import torchvision
import wandb
from dotenv import load_dotenv
# %%

# Import shared components from the main script
from Medical_VAE_Clustering import (
    ConvVAE, 
    get_dataloader, 
    vae_loss_function, 
    DEVICE, 
    IMAGE_SIZE, 
    CHANNELS
)

def train_one_run(config=None):
    # Initialize W&B run
    run = wandb.init(config=config, mode="offline")
    config = wandb.config

    # local logging
    print(f"\n{'='*50}")
    print(f"Starting run with config:")
    print(f"  latent_dim: {config.get('latent_dim', 32)}")
    print(f"  learning_rate: {config.get('learning_rate', 1e-3)}")
    print(f"  beta: {config.get('beta', 5)}")
    print(f"  batch_size: {config.get('batch_size', 64)}")
    print(f"{'='*50}\n")

    # Get hyperparameters from the sweep config
    latent_dim = config.get("latent_dim", 32)
    learning_rate = config.get("learning_rate", 1e-3)
    beta = config.get("beta", 5)
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 10)
    data_path = config.get("data_path", "/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae")

    # Use the imported get_dataloader with the sweep's batch_size
    dataloader, dataset = get_dataloader(use_real_data=True, data_path=data_path, batch_size=batch_size)

    # Initialize model
    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_batches = len(dataloader)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0
        
        for data, _ in dataloader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon, data, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_bce = total_bce / num_batches
        avg_kld = total_kld / num_batches

        # Print progress to console
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | KLD: {avg_kld:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "loss/total": avg_loss,
            "loss/reconstruction": avg_bce,
            "loss/kl": avg_kld,
            "latent_dim": latent_dim,
            "beta": beta,
            "learning_rate": learning_rate,
        })

        # Log sample reconstructions once per epoch (using first batch)
        model.eval()
        with torch.no_grad():
            sample_batch, _ = next(iter(dataloader))
            sample_batch = sample_batch.to(DEVICE)[:8]
            recon_batch, _, _ = model(sample_batch)
            
            comparison = torch.cat([sample_batch, recon_batch])
            grid = torchvision.utils.make_grid(comparison.cpu(), nrow=8)
            
            wandb.log({"samples/reconstruction": wandb.Image(grid.permute(1, 2, 0))})

    # Optionally save model artifact if requested by config
    if config.get("save_model", False):
        artifact_path = f"vae_ld{latent_dim}_beta{beta}.pt"
        torch.save(model.state_dict(), artifact_path)
        
        artifact = wandb.Artifact(
            name=f"vae-ld{latent_dim}-beta{beta}", 
            type="model",
            metadata={"latent_dim": latent_dim, "beta": beta, "learning_rate": learning_rate}
        )
        artifact.add_file(artifact_path)
        run.log_artifact(artifact)

    print(f"\nRun complete! Offline data saved to: {run.dir}")
    run.finish()


def main():
    load_dotenv()

    # Only login if we have internet (login node)
    # Compute nodes will run in offline mode
    if not os.getenv("WANDB_MODE") == "offline":
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            try:
                wandb.login(key=api_key)
            except Exception as e:
                print(f"W&B login failed, switching to offline: {e}")
                os.environ["WANDB_MODE"] = "offline"
            
    # This function is called by the wandb agent repeatedly
    train_one_run()


if __name__ == "__main__":
    main()
