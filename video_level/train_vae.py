"""
Video-Level VAE Training Script.

Trains a Variational Autoencoder on medical images grouped by video,
with various aggregation methods (mean, max, transformer).

Usage:
    python -m video_level.train_vae --beta 2.0 --latent_dim 32 --aggregation mean
    
    # With transformer aggregation
    python video_level/train_vae.py --aggregation transformer --beta 2.0
    
    # Load pre-trained VAE and just do clustering
    python video_level/train_vae.py --load_model path/to/vae.pth --aggregation mean
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import torch.nn as nn
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
    TransformerVideoAggregator,
    vae_loss_function,
    aggregate_frame_latents
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

MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"
CHECKPOINTS_DIR = MODULE_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64
CHANNELS = 1

os.environ["WANDB_MODE"] = "offline"


# ==========================================
# VIDEO-GROUPED DATASETS
# ==========================================

class VideoGroupedDataset(Dataset):
    """Dataset that groups frames by video ID."""
    
    def __init__(self, root_dir, transform=None, frames_per_video=10):
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        all_images = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        
        self.video_frames = defaultdict(list)
        for img_path in all_images:
            video_id = self._extract_video_id(img_path)
            self.video_frames[video_id].append(img_path)
        
        self.video_ids = []
        for video_id, frames in self.video_frames.items():
            frames.sort(key=lambda x: self._extract_frame_number(x))
            if len(frames) == frames_per_video:
                self.video_ids.append(video_id)
            else:
                print(f"Warning: Video {video_id} has {len(frames)} frames, expected {frames_per_video}. Skipping.")
        
        print(f"Found {len(self.video_ids)} complete videos with {frames_per_video} frames each.")
    
    def _extract_video_id(self, path):
        """Extract video ID prefixed with hospital."""
        filename = os.path.basename(path)
        
        hospital = None
        for h in ['JCUH', 'MFT', 'UHW']:
            if h in path:
                hospital = h
                break
        
        match = re.match(r"(.+?)_selected_frame_\d+\.png", filename, re.IGNORECASE)
        if match:
            base_id = match.group(1)
        else:
            base_id = os.path.basename(os.path.dirname(path))
        
        if hospital:
            return f"{hospital}_{base_id}"
        return base_id
    
    def _extract_frame_number(self, path):
        """Extract frame number from filename for sorting."""
        filename = os.path.basename(path)
        match = re.search(r'_selected_frame_(\d+)\.png', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'(\d+)', filename)
        if numbers:
            return int(numbers[-1])
        return 0
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frame_paths = self.video_frames[video_id][:self.frames_per_video]
        
        frames = []
        for path in frame_paths:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        frames_tensor = torch.stack(frames)
        return frames_tensor, video_id


class VideoGroupedImageFolder(Dataset):
    """Alternative dataset using ImageFolder structure."""
    
    def __init__(self, root_dir, transform=None, frames_per_video=10):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        self.image_folder = ImageFolder(root=root_dir)
        
        self.video_frames = defaultdict(list)
        for idx, (path, _) in enumerate(self.image_folder.samples):
            video_id = self._extract_video_id(path)
            self.video_frames[video_id].append((path, idx))
        
        self.video_ids = []
        for video_id, frames in self.video_frames.items():
            frames.sort(key=lambda x: self._extract_frame_number(x[0]))
            if len(frames) == frames_per_video:
                self.video_ids.append(video_id)
        
        print(f"Found {len(self.video_ids)} complete videos with {frames_per_video} frames each.")
    
    def _extract_video_id(self, path):
        """Extract video ID prefixed with hospital."""
        filename = os.path.basename(path)
        
        hospital = None
        for h in ['JCUH', 'MFT', 'UHW']:
            if h in path:
                hospital = h
                break
        
        match = re.match(r"(.+?)_selected_frame_\d+\.png", filename, re.IGNORECASE)
        if match:
            base_id = match.group(1)
        else:
            base_id = os.path.basename(os.path.dirname(path))
        
        if hospital:
            return f"{hospital}_{base_id}"
        return base_id
    
    def _extract_frame_number(self, path):
        filename = os.path.basename(path)
        match = re.search(r'_selected_frame_(\d+)\.png', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'(\d+)', filename)
        if numbers:
            return int(numbers[-1])
        return 0
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frame_data = self.video_frames[video_id][:self.frames_per_video]
        
        frames = []
        for path, _ in frame_data:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        frames_tensor = torch.stack(frames)
        return frames_tensor, video_id


# ==========================================
# DATA LOADING
# ==========================================

def get_video_dataloader(data_path, batch_size=16, crop_percent=0.1, val_split=0.2, frames_per_video=10):
    """Returns dataloaders for video-grouped data."""
    
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

    print(f"Loading video data from {data_path} with crop_percent={crop_percent}")
    
    try:
        full_dataset = VideoGroupedImageFolder(
            root_dir=data_path,
            transform=transform,
            frames_per_video=frames_per_video
        )
    except (FileNotFoundError, RuntimeError):
        print("ImageFolder structure failed - trying flat directory...")
        full_dataset = VideoGroupedDataset(
            root_dir=data_path,
            transform=transform,
            frames_per_video=frames_per_video
        )
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Data Split: {train_size} training videos, {val_size} validation videos.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ==========================================
# TRAINING
# ==========================================

def train_vae_video(model, train_loader, val_loader, epochs, end_beta, learning_rate,
                    save_path, patience=10, use_cyclical=False, device='cuda'):
    """Train VAE on video data (processes all frames but trains at frame level)."""
    
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
        n_frames = 0

        current_beta = annealer.get_beta(epoch)

        for batch_idx, (video_frames, _) in enumerate(train_loader):
            batch_size, n_f, C, H, W = video_frames.shape
            frames_flat = video_frames.view(-1, C, H, W).to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(frames_flat)
            loss, bce, kld = vae_loss_function(recon_batch, frames_flat, mu, logvar, beta=current_beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * frames_flat.size(0)
            total_bce += bce.item() * frames_flat.size(0)
            total_kld += kld.item() * frames_flat.size(0)
            n_frames += frames_flat.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_frames = 0
        with torch.no_grad():
            for video_frames, _ in val_loader:
                batch_size, n_f, C, H, W = video_frames.shape
                frames_flat = video_frames.view(-1, C, H, W).to(device)
                recon_val, mu_val, logvar_val = model(frames_flat)
                v_loss, _, _ = vae_loss_function(recon_val, frames_flat, mu_val, logvar_val, beta=current_beta)
                val_loss += v_loss.item() * frames_flat.size(0)
                val_frames += frames_flat.size(0)

        avg_train_loss = total_loss / n_frames
        avg_bce = total_bce / n_frames
        avg_kld = total_kld / n_frames
        avg_val_loss = val_loss / val_frames

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

    print(f"Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path))

    return tracker


def train_transformer_aggregator(vae_model, train_loader, transformer_model, 
                                  epochs=20, learning_rate=1e-3, device='cuda'):
    """Train the transformer aggregator using contrastive or reconstruction objective."""
    
    vae_model.eval()
    transformer_model = transformer_model.to(device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)
    
    # Simple reconstruction objective: predict mean of frame latents
    criterion = nn.MSELoss()
    
    print("Training Transformer Aggregator...")
    
    for epoch in range(epochs):
        transformer_model.train()
        total_loss = 0
        
        for video_frames, _ in train_loader:
            batch_size, n_f, C, H, W = video_frames.shape
            frames_flat = video_frames.view(-1, C, H, W).to(device)
            
            with torch.no_grad():
                frame_latents = vae_model.encode(frames_flat)
                frame_latents = frame_latents.view(batch_size, n_f, -1)
                target = frame_latents.mean(dim=1)  # Target is mean
            
            optimizer.zero_grad()
            pred = transformer_model(frame_latents)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Transformer Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    return transformer_model


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_video_latent_features(model, dataloader, aggregation_method="mean", 
                                   transformer_model=None, device='cuda'):
    """Extract video-level latent features by aggregating frame embeddings."""
    
    model.eval()
    if transformer_model is not None:
        transformer_model.eval()
    
    video_embeddings = []
    video_ids = []
    
    with torch.no_grad():
        for video_frames, vid_ids in dataloader:
            batch_size, n_frames, C, H, W = video_frames.shape
            
            frames_flat = video_frames.view(-1, C, H, W).to(device)
            frame_latents = model.encode(frames_flat)
            frame_latents = frame_latents.view(batch_size, n_frames, -1)
            
            video_emb = aggregate_frame_latents(
                frame_latents,
                method=aggregation_method,
                transformer_model=transformer_model
            )
            
            video_embeddings.append(video_emb.cpu().numpy())
            video_ids.extend(vid_ids)
    
    return np.concatenate(video_embeddings), video_ids


# ==========================================
# VISUALIZATION
# ==========================================

def log_loss_graphs(tracker, latent_dim, beta, aggregation=""):
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
    plt.title("Reconstruction Loss")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, tracker.history["kl_loss"], label="KL Divergence", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KL Divergence")
    plt.grid(True)
    
    plt.tight_layout()
    
    save_path = RESULTS_DIR / f"loss_curves_ld{latent_dim}_beta{beta}_{aggregation}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to: {save_path}")


def perform_video_clustering_and_log(model, dataloader, latent_dim, beta, aggregation_method="mean",
                                      transformer_model=None, selected_dims=None, device='cuda'):
    """Runs video-level clustering and saves plots."""
    print(f"Generating video-level clustering plots (aggregation={aggregation_method})...")
    
    video_embeddings, video_ids = extract_video_latent_features(
        model, dataloader, aggregation_method, transformer_model=transformer_model, device=device
    )
    
    print(f"Extracted {len(video_embeddings)} video embeddings of shape {video_embeddings.shape}")
    
    # Filter to selected dimensions if specified
    if selected_dims is not None:
        print(f"Using only selected latent dimensions: {selected_dims}")
        embeddings_for_clustering = video_embeddings[:, selected_dims]
        dims_label = f"dims{'-'.join(map(str, selected_dims))}"
    else:
        embeddings_for_clustering = video_embeddings
        dims_label = "all_dims"
    
    for N_CLUSTERS in [2, 3, 4]:
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(embeddings_for_clustering)

        # t-SNE
        perplexity = min(30, len(embeddings_for_clustering) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        limit = min(len(embeddings_for_clustering), 1000)
        X_embedded = tsne.fit_transform(embeddings_for_clustering[:limit])
        clusters_plot = clusters[:limit]
        
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_plot, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f"Video Clustering (LD={latent_dim}, Beta={beta}, Agg={aggregation_method}, K={N_CLUSTERS})")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.tight_layout()
        
        save_path = RESULTS_DIR / f"video_clustering_ld{latent_dim}_beta{beta}_{aggregation_method}_k{N_CLUSTERS}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Clustering plot saved to: {save_path}")
        
        # Log to wandb if available
        if HAS_WANDB and wandb.run is not None:
            table_data = list(zip(
                video_ids[:limit],
                clusters[:limit],
                X_embedded[:, 0],
                X_embedded[:, 1]
            ))
            wandb.log({
                f"plots/video_clustering_k{N_CLUSTERS}": wandb.Image(fig),
                f"video_cluster_labels_k{N_CLUSTERS}": wandb.Table(
                    data=table_data,
                    columns=["video_id", "cluster_label", "tsne_x", "tsne_y"]
                )
            })
    
    return video_embeddings, video_ids, clusters


def log_video_reconstruction(model, dataloader, latent_dim, beta, device='cuda'):
    """Log reconstructions for a sample video."""
    model.eval()
    with torch.no_grad():
        video_frames, video_id = next(iter(dataloader))
        video_frames = video_frames[0].to(device)  # First video
        
        recon, _, _ = model(video_frames)
        
        comparison = torch.cat([video_frames, recon])
        grid = torchvision.utils.make_grid(comparison.cpu(), nrow=video_frames.size(0))
        
        fig = plt.figure(figsize=(15, 5))
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f"Video: {video_id} | Top: Original | Bottom: Reconstructed")
        plt.axis('off')
        
        save_path = RESULTS_DIR / f"video_reconstruction_ld{latent_dim}_beta{beta}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Video reconstruction saved to: {save_path}")


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Video-Level VAE Training")
    parser.add_argument("--crop_percent", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--annealing", type=str, default="cyclical", choices=["cyclical", "linear"])
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max", "concat", "transformer"])
    parser.add_argument("--frames_per_video", type=int, default=10)
    parser.add_argument("--load_model", type=str, default=None, help="Path to pre-trained VAE")
    parser.add_argument("--selected_dims", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Config: crop={args.crop_percent}, beta={args.beta}, epochs={args.epochs}, ld={args.latent_dim}, agg={args.aggregation}")
    
    load_dotenv()
    
    if HAS_WANDB:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            try:
                wandb.login(key=wandb_key)
            except Exception as e:
                print(f"W&B login failed: {e}")
    
    data_path = args.data_path or str(Config.VAE_DATA_PATH)
    
    train_loader, val_loader = get_video_dataloader(
        data_path=data_path,
        batch_size=args.batch_size,
        crop_percent=args.crop_percent,
        frames_per_video=args.frames_per_video
    )
    
    if HAS_WANDB:
        wandb.init(
            project="lus-medical-vae",
            group="video_level",
            config={
                "crop_percent": args.crop_percent,
                "beta": args.beta,
                "epochs": args.epochs,
                "latent_dim": args.latent_dim,
                "aggregation": args.aggregation,
                "level": "video"
            },
            reinit=True
        )
    
    vae = ConvVAE(latent_dim=args.latent_dim, channels=CHANNELS)
    
    suffix = f"crop{int(args.crop_percent*100)}"
    full_suffix = f"ld{args.latent_dim}_{suffix}_beta{args.beta}_{args.aggregation}"
    model_save_path = CHECKPOINTS_DIR / f"Best_VAE_{full_suffix}.pth"
    
    # Train or load VAE
    if args.load_model:
        print(f"Loading pre-trained VAE from: {args.load_model}")
        vae.load_state_dict(torch.load(args.load_model, map_location=device))
        vae = vae.to(device)
        tracker = None
    else:
        tracker = train_vae_video(
            vae, train_loader, val_loader,
            epochs=args.epochs,
            end_beta=args.beta,
            learning_rate=args.learning_rate,
            save_path=model_save_path,
            patience=10,
            use_cyclical=(args.annealing == "cyclical"),
            device=device
        )
    
    # Train transformer if needed
    transformer_model = None
    if args.aggregation == "transformer":
        transformer_model = TransformerVideoAggregator(
            latent_dim=args.latent_dim,
            n_frames=args.frames_per_video
        )
        transformer_model = train_transformer_aggregator(
            vae, train_loader, transformer_model,
            epochs=20, learning_rate=1e-3, device=device
        )
        # Save transformer
        trans_save_path = CHECKPOINTS_DIR / f"Transformer_Aggregator_{full_suffix}.pth"
        torch.save(transformer_model.state_dict(), trans_save_path)
        print(f"Transformer saved to: {trans_save_path}")
    
    # Generate visualizations
    perform_video_clustering_and_log(
        vae, train_loader, args.latent_dim, args.beta,
        aggregation_method=args.aggregation,
        transformer_model=transformer_model,
        selected_dims=args.selected_dims,
        device=device
    )
    
    log_video_reconstruction(vae, val_loader, args.latent_dim, args.beta, device)
    
    if tracker:
        log_loss_graphs(tracker, args.latent_dim, args.beta, args.aggregation)
    
    # Save video embeddings
    print("Extracting video embeddings...")
    video_emb, video_ids = extract_video_latent_features(
        vae, train_loader, args.aggregation, transformer_model, device
    )
    
    emb_dir = RESULTS_DIR / "video_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / f"video_embeddings_{full_suffix}.npy", video_emb)
    np.save(emb_dir / f"video_ids_{full_suffix}.npy", np.array(video_ids))
    print(f"Saved {video_emb.shape[0]} video embeddings of dimension {video_emb.shape[1]}")
    
    if HAS_WANDB and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"video_embeddings_{full_suffix}",
            type="video_embeddings",
            description=f"Video-level embeddings for ld={args.latent_dim}, beta={args.beta}, agg={args.aggregation}"
        )
        artifact.add_file(str(emb_dir / f"video_embeddings_{full_suffix}.npy"))
        artifact.add_file(str(emb_dir / f"video_ids_{full_suffix}.npy"))
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print(f"\nDone! Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
