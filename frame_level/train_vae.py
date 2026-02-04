"""
Frame-Level VAE Training Script.

Trains a Variational Autoencoder on individual medical images (frames),
without any video-level aggregation.

Usage:
    python -m frame_level.train_vae --beta 2.0 --latent_dim 32 --epochs 60
    
    # Or run directly
    python frame_level/train_vae.py --crop_percent 0.1 --beta 2.0
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import (
    ConvVAE,
    KLAnnealer,
    CyclicalAnnealer,
    EarlyStopping,
    LossTracker,
    vae_loss_function
)
from shared.config import Config

# Setup wandb (optional)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ==========================================
# CONFIGURATION
# ==========================================

# Paths for this module
MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"
CHECKPOINTS_DIR = MODULE_DIR / "checkpoints"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Image settings
IMAGE_SIZE = 64
CHANNELS = 1  # Grayscale

os.environ["WANDB_MODE"] = "offline"


# ==========================================
# DATASET CLASSES
# ==========================================

class FlatImageDataset(Dataset):
    """Dataset for images in a flat directory (no class subfolders)."""
    
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths."""
    
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        image = original_tuple[0]
        image_path = self.samples[index][0]
        return image, image_path


# ==========================================
# DATA LOADING
# ==========================================

def get_dataloader(data_path, batch_size=64, crop_percent=0.1, val_split=0.2):
    """
    Returns train and validation dataloaders for frame-level data.
    """
    transform_ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if crop_percent > 0:
        transform_ops.append(
            transforms.Lambda(lambda img: transforms.functional.crop(
                img,
                top=int(img.size[1] * crop_percent),
                left=0,
                height=int(img.size[1] * (1 - crop_percent)),
                width=img.size[0]
            ))
        )
    
    transform_ops.extend([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=CHANNELS),
        transforms.ToTensor()
    ])
    
    transform = transforms.Compose(transform_ops)

    print(f"Loading frame data from {data_path} with crop_percent={crop_percent}")
    
    try:
        full_dataset = ImageFolderWithPaths(root=data_path, transform=transform)
    except (FileNotFoundError, RuntimeError):
        print("ImageFolder failed - trying flat directory structure...")
        full_dataset = FlatImageDataset(root_dir=data_path, transform=transform)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Data Split: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader


# ==========================================
# TRAINING
# ==========================================

def train_vae(model, train_loader, val_loader, epochs, end_beta, learning_rate, 
              save_path, patience=10, use_cyclical=False, device='cuda'):
    """Train VAE model on frame-level data."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tracker = LossTracker()

    if use_cyclical:
        annealer = CyclicalAnnealer(total_epochs=epochs, cycles=4, start_beta=0.0, end_beta=end_beta)
    else:
        annealer = KLAnnealer(total_epochs=epochs, start_beta=0.0, end_beta=end_beta)

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=str(save_path))

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0

        current_beta = annealer.get_beta(epoch)

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=current_beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(device)
                recon_val, mu_val, logvar_val = model(val_data)
                v_loss, _, _ = vae_loss_function(recon_val, val_data, mu_val, logvar_val, beta=current_beta)
                val_loss += v_loss.item()

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        tracker.add(avg_train_loss, avg_bce, avg_kld, avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f} | Beta: {current_beta:.4f}")
        
        # Early stopping check
        if use_cyclical:
            total_cycles = epochs // annealer.cycle_length
            current_cycle = epoch // annealer.cycle_length
            if current_cycle >= total_cycles - 1:
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
        else:
            if epoch >= annealer.warmup_epochs:
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
            else:
                if avg_val_loss < early_stopping.val_loss_min - early_stopping.delta:
                    early_stopping.save_checkpoint(avg_val_loss, model)

    # Load best model
    print(f"Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path))

    return tracker


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_latent_features(model, dataloader, device='cuda'):
    """
    Passes all data through encoder to get the latent vectors (mu).
    """
    model.eval()
    latent_vectors = []
    image_paths = []
    
    with torch.no_grad():
        for data, paths in dataloader:
            data = data.to(device)
            mu = model.encode(data)
            latent_vectors.append(mu.cpu().numpy())
            image_paths.extend(paths)
            
    return np.concatenate(latent_vectors), image_paths


# ==========================================
# VISUALIZATION
# ==========================================

def log_loss_graphs(tracker, latent_dim, beta, crop_suffix=""):
    """Plot and save loss curves."""
    epochs = range(1, len(tracker.history["loss"]) + 1)
    
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, tracker.history["loss"], label="Total Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Total Loss (ld={latent_dim}, beta={beta})")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, tracker.history["reconstruction_loss"], label="Recon Loss", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Reconstruction Loss")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, tracker.history["kl_loss"], label="KL Divergence", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"KL Divergence")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save locally
    save_path = RESULTS_DIR / f"loss_curves_ld{latent_dim}_beta{beta}_{crop_suffix}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to: {save_path}")
    
    # Log to wandb if available
    if HAS_WANDB and wandb.run is not None:
        wandb.log({"plots/loss_curves": wandb.Image(fig)})


def plot_latent_distributions(latent_vectors, latent_dim, save_suffix=""):
    """Plot the distribution of each latent feature dimension."""
    n_cols = 6
    n_rows = (latent_dim + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(latent_dim):
        ax = axes[i]
        feature_data = latent_vectors[:, i]
        
        mu = np.mean(feature_data)
        std = np.std(feature_data)
        
        ax.hist(feature_data, bins=30, density=True, alpha=0.6, color='gray', edgecolor='none')
        
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=1)
        
        ax.set_title(f"Feat {i+1}\n({mu:.3f}, {std:.3f})", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Hide unused subplots
    for j in range(latent_dim, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    save_path = RESULTS_DIR / f"latent_distributions_{save_suffix}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Latent distributions saved to: {save_path}")


def perform_clustering_and_log(model, dataloader, latent_dim, beta, crop_suffix="", device='cuda'):
    """Runs feature extraction, KMeans, t-SNE, and saves plots."""
    print(f"Generating clustering plots for LD={latent_dim}...")
    
    X_latent, image_paths = extract_latent_features(model, dataloader, device)
    
    for N_CLUSTERS in [2, 3, 4]:
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_latent)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        limit = min(len(X_latent), 1000)
        X_embedded = tsne.fit_transform(X_latent[:limit])
        clusters_plot = clusters[:limit]
        
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_plot, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f"Frame Clustering LD={latent_dim}, Beta={beta}, K={N_CLUSTERS}")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.tight_layout()
        
        save_path = RESULTS_DIR / f"clustering_ld{latent_dim}_beta{beta}_{crop_suffix}_k{N_CLUSTERS}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Clustering plot saved to: {save_path}")
        
        # Log to wandb if available
        if HAS_WANDB and wandb.run is not None:
            table_data = list(zip(
                image_paths[:limit],
                clusters[:limit],
                X_embedded[:, 0],
                X_embedded[:, 1]
            ))
            wandb.log({
                f"plots/clustering_k{N_CLUSTERS}": wandb.Image(fig),
                f"cluster_labels_k{N_CLUSTERS}": wandb.Table(
                    data=table_data,
                    columns=["image_path", "cluster_label", "tsne_x", "tsne_y"]
                )
            })


def log_reconstruction(model, dataloader, latent_dim, beta, crop_suffix="", device='cuda'):
    """Log reconstruction comparison."""
    model.eval()
    with torch.no_grad():
        sample_data, _ = next(iter(dataloader))
        sample_data = sample_data.to(device)[:8]
        recon, _, _ = model(sample_data)
        
        comparison = torch.cat([sample_data, recon])
        grid = torchvision.utils.make_grid(comparison.cpu(), nrow=8)
        
        fig = plt.figure(figsize=(15, 5))
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f"Top: Original | Bottom: Reconstructed (LD={latent_dim}, Beta={beta})")
        plt.axis('off')
        
        save_path = RESULTS_DIR / f"reconstruction_ld{latent_dim}_beta{beta}_{crop_suffix}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Reconstruction saved to: {save_path}")


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Frame-Level VAE Training")
    parser.add_argument("--crop_percent", type=float, default=0.1, help="Percentage to crop from top")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta parameter for KL Divergence")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension size")
    parser.add_argument("--annealing", type=str, default="cyclical", choices=["cyclical", "linear"])
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Config: crop={args.crop_percent}, beta={args.beta}, epochs={args.epochs}, ld={args.latent_dim}")
    
    # Load environment
    load_dotenv()
    
    # Setup wandb (optional)
    if HAS_WANDB:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            try:
                wandb.login(key=wandb_key)
            except Exception as e:
                print(f"W&B login failed: {e}")
    
    # Data path
    data_path = args.data_path or str(Config.VAE_DATA_PATH)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloader(
        data_path=data_path,
        batch_size=args.batch_size,
        crop_percent=args.crop_percent
    )
    
    # Initialize wandb run
    if HAS_WANDB:
        wandb.init(
            project="lus-medical-vae",
            group="frame_level",
            config={
                "crop_percent": args.crop_percent,
                "beta": args.beta,
                "epochs": args.epochs,
                "latent_dim": args.latent_dim,
                "level": "frame"
            },
            reinit=True
        )
    
    # Create model
    vae = ConvVAE(latent_dim=args.latent_dim, channels=CHANNELS)
    
    # Model save path
    suffix = f"crop{int(args.crop_percent*100)}"
    full_suffix = f"ld{args.latent_dim}_{suffix}_beta{args.beta}"
    model_save_path = CHECKPOINTS_DIR / f"Best_VAE_{full_suffix}_{args.annealing}.pth"
    
    # Train
    tracker = train_vae(
        vae, train_loader, val_loader,
        epochs=args.epochs,
        end_beta=args.beta,
        learning_rate=args.learning_rate,
        save_path=model_save_path,
        patience=10,
        use_cyclical=(args.annealing == "cyclical"),
        device=device
    )
    
    # Generate visualizations
    perform_clustering_and_log(vae, train_loader, args.latent_dim, args.beta, suffix, device)
    log_reconstruction(vae, val_loader, args.latent_dim, args.beta, suffix, device)
    log_loss_graphs(tracker, args.latent_dim, args.beta, suffix)
    
    # Save latent features
    print("Extracting features for distribution plot...")
    X_latent, image_paths = extract_latent_features(vae, train_loader, device)
    plot_latent_distributions(X_latent, args.latent_dim, full_suffix)
    
    # Save latent vectors
    latent_dir = RESULTS_DIR / "latent_features"
    latent_dir.mkdir(parents=True, exist_ok=True)
    np.save(latent_dir / f"latent_vectors_{full_suffix}.npy", X_latent)
    np.save(latent_dir / f"image_paths_{full_suffix}.npy", np.array(image_paths))
    print(f"Saved {X_latent.shape[0]} latent vectors of dimension {X_latent.shape[1]}")
    
    if HAS_WANDB and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"frame_latent_vectors_{full_suffix}",
            type="latent_features",
            description=f"Frame-level latent vectors for ld={args.latent_dim}, beta={args.beta}"
        )
        artifact.add_file(str(latent_dir / f"latent_vectors_{full_suffix}.npy"))
        artifact.add_file(str(latent_dir / f"image_paths_{full_suffix}.npy"))
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print(f"\nDone! Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
