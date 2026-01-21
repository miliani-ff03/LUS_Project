# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
import glob
import re
from PIL import Image
from dotenv import load_dotenv
import wandb
import scipy.stats as stats
import argparse
import sys
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_MODE"] = "offline"

# %%
# ==========================================
# 1. CONFIGURATION
# ==========================================
parser = argparse.ArgumentParser(description="VAE Training with Video-Level Aggregation")

parser.add_argument("--crop_percent", type=float, default=0.1, help="Percentage to crop from top (0.0 to 1.0)")
parser.add_argument("--beta", type=float, default=2.0, help="Beta parameter for KL Divergence")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension size")
parser.add_argument("--annealing", type=str, default="cyclical", choices=["cyclical", "linear"], help="Type of beta annealing")
parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max", "concat", "transformer"], help="Video aggregation method")
parser.add_argument("--frames_per_video", type=int, default=10, help="Number of frames per video")
parser.add_argument("--load_model", type=str, default=None, help="Path to pre-trained VAE model (skips training if provided)")

if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
    args = parser.parse_args([])
else:
    args = parser.parse_args()

# Image settings
IMAGE_SIZE = 64
CHANNELS = 1
LATENT_DIM = args.latent_dim
FRAMES_PER_VIDEO = args.frames_per_video
AGGREGATION_METHOD = args.aggregation

# Training settings
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = args.epochs
BETA = args.beta
CROP_PERCENT = args.crop_percent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# %%
# ==========================================
# 2. VAE MODEL ARCHITECTURE (Same as original)
# ==========================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        
        z_input = self.decoder_input(z)
        z_matrix = z_input.view(z_input.size(0), 256, 4, 4)
        reconstruction = self.decoder(z_matrix)
        
        return reconstruction, mu, logvar
    
    def encode(self, x):
        """Encode images to latent space (returns mu only for inference)."""
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        return mu

# ==========================================
# 3. VIDEO AGGREGATION (Pooling + Transformer)
# ==========================================

class TransformerVideoAggregator(nn.Module):
    """
    Aggregates frame-level latent vectors into a single video embedding using self-attention.
    
    Uses a learnable CLS token (like BERT) to aggregate information from all frames.
    The transformer learns which frames are most important for the final representation.
    
    Args:
        latent_dim: Dimension of frame latent vectors (default: 32)
        n_frames: Number of frames per video (default: 10)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer encoder layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, latent_dim=32, n_frames=10, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        
        # Learnable positional encoding for temporal order
        self.pos_encoding = nn.Parameter(torch.randn(1, n_frames, latent_dim) * 0.02)
        
        # CLS token for aggregation (prepended to sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Optional: projection head for the output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, frame_latents):
        """
        Args:
            frame_latents: Tensor of shape (batch, n_frames, latent_dim)
        
        Returns:
            video_embedding: Tensor of shape (batch, latent_dim)
        """
        B = frame_latents.size(0)
        
        # Add positional encoding to frame latents
        x = frame_latents + self.pos_encoding
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, latent_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_frames+1, latent_dim)
        
        # Self-attention across frames
        x = self.transformer(x)
        
        # Extract CLS token as video embedding
        cls_output = x[:, 0, :]  # (B, latent_dim)
        
        # Project output
        video_embedding = self.output_proj(cls_output)
        
        return video_embedding
    
    def get_attention_weights(self, frame_latents):
        """Get attention weights to see which frames the model focuses on."""
        # This is a simplified version - full implementation would extract from transformer layers
        B = frame_latents.size(0)
        x = frame_latents + self.pos_encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Use hooks or manual computation for actual attention weights
        # For now, return uniform weights as placeholder
        return torch.ones(B, self.n_frames) / self.n_frames


def aggregate_frame_latents(frame_latents, method="mean", transformer_model=None):
    """
    Aggregate frame-level latent vectors into a single video-level embedding.
    
    Args:
        frame_latents: Tensor of shape (n_frames, latent_dim) or (batch, n_frames, latent_dim)
        method: 'mean', 'max', 'concat', or 'transformer'
        transformer_model: Required if method='transformer', instance of TransformerVideoAggregator
    
    Returns:
        video_embedding: Tensor of shape (latent_dim,) or (latent_dim * n_frames,) for concat
    """
    if method == "mean":
        return frame_latents.mean(dim=-2)
    elif method == "max":
        return frame_latents.max(dim=-2)[0]
    elif method == "concat":
        if frame_latents.dim() == 2:
            return frame_latents.flatten()
        else:
            return frame_latents.flatten(start_dim=-2)
    elif method == "transformer":
        if transformer_model is None:
            raise ValueError("transformer_model must be provided when method='transformer'")
        # Ensure 3D input (batch, n_frames, latent_dim)
        if frame_latents.dim() == 2:
            frame_latents = frame_latents.unsqueeze(0)
        return transformer_model(frame_latents)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

# ==========================================
# 4. VIDEO-GROUPED DATASET
# ==========================================
class VideoGroupedDataset(Dataset):
    """
    Dataset that groups frames by video ID.
    Assumes filenames follow pattern: videoID_frameN.png or similar.
    """
    def __init__(self, root_dir, transform=None, frames_per_video=10):
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Find all images
        all_images = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        
        # Group images by video ID
        self.video_frames = defaultdict(list)
        for img_path in all_images:
            video_id = self._extract_video_id(img_path)
            self.video_frames[video_id].append(img_path)
        
        # Sort frames within each video and filter videos with correct frame count
        self.video_ids = []
        for video_id, frames in self.video_frames.items():
            # Sort frames by frame number
            frames.sort(key=lambda x: self._extract_frame_number(x))
            if len(frames) == frames_per_video:
                self.video_ids.append(video_id)
            else:
                print(f"Warning: Video {video_id} has {len(frames)} frames, expected {frames_per_video}. Skipping.")
        
        print(f"Found {len(self.video_ids)} complete videos with {frames_per_video} frames each.")
    
    def _extract_video_id(self, path):
        """
        Extract video ID from filename, prefixed with hospital to avoid collisions.
        
        Handles formats:
        - JCUH/MFT: .../JCUH/27_LU_4_RPB_selected_frame_0002.png -> JCUH_27_LU_4_RPB
        - UHW: .../UHW/YOUNG-SCAN-4_0010_selected_frame_0009.png -> UHW_YOUNG-SCAN-4_0010
        """
        filename = os.path.basename(path)
        
        # Detect hospital from path
        hospital = None
        for h in ['JCUH', 'MFT', 'UHW']:
            if h in path:
                hospital = h
                break
        
        # Extract base video ID
        match = re.match(r"(.+?)_selected_frame_\d+\.png", filename, re.IGNORECASE)
        if match:
            base_id = match.group(1)
        else:
            base_id = os.path.basename(os.path.dirname(path))
        
        # Prefix with hospital to ensure uniqueness
        if hospital:
            return f"{hospital}_{base_id}"
        return base_id
    
    def _extract_frame_number(self, path):
        """
        Extract frame number from filename for sorting.
        Extracts NNNN from '_selected_frame_NNNN.png'
        """
        filename = os.path.basename(path)
        match = re.search(r'_selected_frame_(\d+)\.png', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Fallback: use last number in filename
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
        
        # Stack frames: (n_frames, C, H, W)
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, video_id


class VideoGroupedImageFolder(Dataset):
    """
    Alternative dataset that uses ImageFolder structure but groups by video.
    Assumes: root/class/videoID_frameN.png
    """
    def __init__(self, root_dir, transform=None, frames_per_video=10):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Use ImageFolder to get all images
        self.image_folder = ImageFolder(root=root_dir)
        
        # Group by video ID
        self.video_frames = defaultdict(list)
        for idx, (path, _) in enumerate(self.image_folder.samples):
            video_id = self._extract_video_id(path)
            self.video_frames[video_id].append((path, idx))
        
        # Filter and sort
        self.video_ids = []
        for video_id, frames in self.video_frames.items():
            frames.sort(key=lambda x: self._extract_frame_number(x[0]))
            if len(frames) == frames_per_video:
                self.video_ids.append(video_id)
        
        print(f"Found {len(self.video_ids)} complete videos with {frames_per_video} frames each.")
    
    def _extract_video_id(self, path):
        """Extract video ID prefixed with hospital: HOSPITAL_videoId"""
        filename = os.path.basename(path)
        
        # Detect hospital from path
        hospital = None
        for h in ['JCUH', 'MFT', 'UHW']:
            if h in path:
                hospital = h
                break
        
        # Extract base video ID
        match = re.match(r"(.+?)_selected_frame_\d+\.png", filename, re.IGNORECASE)
        if match:
            base_id = match.group(1)
        else:
            base_id = os.path.basename(os.path.dirname(path))
        
        # Prefix with hospital
        if hospital:
            return f"{hospital}_{base_id}"
        return base_id
    
    def _extract_frame_number(self, path):
        """Extract NNNN from '_selected_frame_NNNN.png'"""
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
# 5. LOSS AND TRAINING UTILITIES (Same as original)
# ==========================================
class KLAnnealer:
    def __init__(self, total_epochs, start_beta=0.0, end_beta=1.0):
        self.total_epochs = total_epochs
        self.warmup_epochs = max(1, total_epochs // 5)
        self.start_beta = start_beta
        self.end_beta = end_beta
    
    def get_beta(self, epoch):
        if epoch < self.warmup_epochs:
            return self.start_beta + (self.end_beta - self.start_beta) * (epoch / self.warmup_epochs)
        return self.end_beta

class CyclicalAnnealer:
    def __init__(self, total_epochs, cycles=4, start_beta=0.0, end_beta=1.0):
        self.total_epochs = total_epochs
        self.cycles = cycles
        self.cycle_length = total_epochs // cycles
        self.start_beta = start_beta
        self.end_beta = end_beta
    
    def get_beta(self, epoch):
        cycle_epoch = epoch % self.cycle_length
        half_cycle = self.cycle_length / 2
        if cycle_epoch < half_cycle:
            return self.start_beta + (self.end_beta - self.start_beta) * (cycle_epoch / half_cycle)
        return self.end_beta

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.01, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class LossTracker:
    def __init__(self):
        self.history = {"loss": [], "reconstruction_loss": [], "kl_loss": [], "validation_loss": []}

    def add(self, total, recon, kl, val):
        self.history["loss"].append(total)
        self.history["reconstruction_loss"].append(recon)
        self.history["kl_loss"].append(kl)
        self.history["validation_loss"].append(val)

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    batch_size = x.size(0)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = BCE + beta * KLD
    return total_loss, BCE, KLD

# ==========================================
# 6. DATA LOADING
# ==========================================
def get_video_dataloader(data_path, batch_size=16, crop_percent=0.25, val_split=0.2, frames_per_video=10):
    """
    Returns dataloaders for video-grouped data.
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
# 7. TRAINING WITH VIDEO DATA
# ==========================================
def train_vae_video(model, train_loader, val_loader, epochs, end_beta, learning_rate, 
                    save_path="Best_VAE.pth", patience=10, use_cyclical=False):
    """
    Train VAE on video data (processes all frames but trains at frame level).
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tracker = LossTracker()

    if use_cyclical:
        annealer = CyclicalAnnealer(total_epochs=epochs, cycles=4, start_beta=0.0, end_beta=end_beta)
    else:
        annealer = KLAnnealer(total_epochs=epochs, start_beta=0.0, end_beta=end_beta)

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0
        n_frames = 0

        current_beta = annealer.get_beta(epoch)

        for batch_idx, (video_frames, _) in enumerate(train_loader):
            # video_frames: (batch, n_frames, C, H, W)
            batch_size, n_f, C, H, W = video_frames.shape
            
            # Flatten to process all frames through VAE
            frames_flat = video_frames.view(-1, C, H, W).to(DEVICE)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(frames_flat)
            loss, bce, kld = vae_loss_function(recon_batch, frames_flat, mu, logvar, beta=current_beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * frames_flat.size(0)
            total_bce += bce.item() * frames_flat.size(0)
            total_kld += kld.item() * frames_flat.size(0)
            n_frames += frames_flat.size(0)

        # Validation Phase
        model.eval()
        val_loss = 0
        val_frames = 0
        with torch.no_grad():
            for video_frames, _ in val_loader:
                batch_size, n_f, C, H, W = video_frames.shape
                frames_flat = video_frames.view(-1, C, H, W).to(DEVICE)
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

# ==========================================
# 8. VIDEO-LEVEL FEATURE EXTRACTION
# ==========================================
def extract_video_latent_features(model, dataloader, aggregation_method="mean", transformer_model=None):
    """
    Extract video-level latent features by aggregating frame embeddings.
    
    Args:
        model: VAE model for encoding frames
        dataloader: Video dataloader
        aggregation_method: 'mean', 'max', 'concat', or 'transformer'
        transformer_model: Required if aggregation_method='transformer'
    
    Returns:
        video_embeddings: (n_videos, latent_dim) or (n_videos, latent_dim * n_frames) for concat
        video_ids: list of video identifiers
    """
    model.eval()
    if transformer_model is not None:
        transformer_model.eval()
    
    video_embeddings = []
    video_ids = []
    
    with torch.no_grad():
        for video_frames, vid_ids in dataloader:
            # video_frames: (batch, n_frames, C, H, W)
            batch_size, n_frames, C, H, W = video_frames.shape
            
            # Process all frames through encoder
            frames_flat = video_frames.view(-1, C, H, W).to(DEVICE)
            frame_latents = model.encode(frames_flat)  # (batch * n_frames, latent_dim)
            
            # Reshape back to (batch, n_frames, latent_dim)
            frame_latents = frame_latents.view(batch_size, n_frames, -1)
            
            # Aggregate to video level
            video_emb = aggregate_frame_latents(
                frame_latents, 
                method=aggregation_method,
                transformer_model=transformer_model
            )
            
            video_embeddings.append(video_emb.cpu().numpy())
            video_ids.extend(vid_ids)
    
    return np.concatenate(video_embeddings), video_ids

# ==========================================
# 9. VIDEO-LEVEL CLUSTERING AND VISUALIZATION
# ==========================================
def perform_video_clustering_and_log(model, dataloader, latent_dim, beta, aggregation_method="mean", crop_suffix="", transformer_model=None):
    """
    Runs video-level feature extraction, KMeans, t-SNE, and logs plots to WandB.
    """
    print(f"Generating video-level clustering plots (aggregation={aggregation_method})...")
    
    # Extract video-level features
    video_embeddings, video_ids = extract_video_latent_features(
        model, dataloader, aggregation_method, transformer_model=transformer_model
    )
    
    print(f"Extracted {len(video_embeddings)} video embeddings of shape {video_embeddings.shape}")
    
    for N_CLUSTERS in [2, 3, 4]:
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(video_embeddings)

        # t-SNE Visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(video_embeddings) - 1))
        limit = min(len(video_embeddings), 1000)
        X_embedded = tsne.fit_transform(video_embeddings[:limit])
        clusters_plot = clusters[:limit]
        
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_plot, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f"Video Clustering (LD={latent_dim}, Beta={beta}, Agg={aggregation_method}, K={N_CLUSTERS})")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.tight_layout()

        table_data = list(zip(
            video_ids[:limit],
            clusters[:limit],
            X_embedded[:, 0],
            X_embedded[:, 1]
        ))

        if wandb.run is not None:
            wandb.log({
                f"plots/video_clustering_k{N_CLUSTERS}": wandb.Image(fig),
                f"video_cluster_labels_k{N_CLUSTERS}": wandb.Table(
                    data=table_data,
                    columns=["video_id", "cluster_label", "tsne_x", "tsne_y"]
                )
            })
        
        plt.close(fig)
    
    return video_embeddings, video_ids, clusters

def log_loss_graphs(tracker, latent_dim, beta, crop_suffix=""):
    epochs = range(1, len(tracker.history["loss"]) + 1)
    
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, tracker.history["loss"], label="Total Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Total Loss (ld={latent_dim}, beta={beta})")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, tracker.history["reconstruction_loss"], label="Reconstruction Loss", color='green')
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
    
    if wandb.run is not None:
        wandb.log({
            "plots/loss_curves": wandb.Image(fig),
            "latent_dim": latent_dim,
            "beta": beta,
        })
    plt.close(fig)

def log_video_reconstruction(model, dataloader, latent_dim, beta, crop_suffix=""):
    """Log reconstructions for a sample video (showing all frames)."""
    print(f"Generating video reconstruction plots...")
    
    model.eval()
    with torch.no_grad():
        video_frames, video_id = next(iter(dataloader))
        
        # Take first video only
        frames = video_frames[0].to(DEVICE)  # (n_frames, C, H, W)
        
        # Reconstruct
        recon, _, _ = model(frames)
        
        # Create grid: Top row = Original, Bottom row = Reconstructed
        comparison = torch.cat([frames, recon])
        grid = torchvision.utils.make_grid(comparison.cpu(), nrow=frames.size(0))
        
        fig = plt.figure(figsize=(15, 5))
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f"Video {video_id} | Top: Original | Bottom: Reconstructed")
        plt.axis('off')
        
        if wandb.run is not None:
            wandb.log({
                "plots/video_reconstruction": wandb.Image(fig),
            })
        plt.close(fig)

# ==========================================
# 10. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY:
        try:
            wandb.login(key=WANDB_API_KEY)
        except Exception as e:
            print(f"W&B login failed: {e}")
    else:
        print("WANDB_API_KEY not found. Proceeding without explicit login.")

    print(f"Running on device: {DEVICE}")
    print(f"Config: crop={CROP_PERCENT}, beta={BETA}, epochs={EPOCHS}, latent_dim={LATENT_DIM}")
    print(f"Video config: frames_per_video={FRAMES_PER_VIDEO}, aggregation={AGGREGATION_METHOD}")
    
    # --- STEP 1: DATA ---
    DATA_PATH = os.getenv("DATA_PATH", "/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae/all_images")
    
    # Note: Using smaller batch size since each sample is now 10 frames
    train_loader, val_loader = get_video_dataloader(
        data_path=DATA_PATH,
        batch_size=16,  # Smaller since each sample is a video
        crop_percent=CROP_PERCENT,
        val_split=0.2,
        frames_per_video=FRAMES_PER_VIDEO
    )
    
    # --- STEP 2: TRAIN MODEL ---
    run = wandb.init(
        project="lus-medical-vae-video",
        group="video_aggregation",
        config={
            "crop_percent": CROP_PERCENT,
            "end_beta": BETA,
            "epochs": EPOCHS,
            "latent_dim": LATENT_DIM,
            "frames_per_video": FRAMES_PER_VIDEO,
            "aggregation_method": AGGREGATION_METHOD
        },
        reinit=True
    )

    vae = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
    suffix = f"crop{int(CROP_PERCENT*100)}"
    full_suffix = f"ld{LATENT_DIM}_crop{int(CROP_PERCENT*100)}_beta{BETA}_{AGGREGATION_METHOD}"

    annealing_type = args.annealing
    model_save_name = f"Best_VAE_video_{full_suffix}_{annealing_type}.pth"

    # --- STEP 2: LOAD OR TRAIN MODEL ---
    if args.load_model:
        # Load pre-trained model (skip training)
        print(f"\n=== LOADING PRE-TRAINED MODEL ===")
        print(f"Loading from: {args.load_model}")
        vae.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
        print("Model loaded successfully! Skipping training.")
        tracker = None  # No training history
    else:
        # Train from scratch
        print(f"\n=== TRAINING NEW MODEL ===")
        tracker = train_vae_video(
            vae, train_loader, val_loader,
            epochs=EPOCHS,
            end_beta=BETA,
            learning_rate=LEARNING_RATE,
            save_path=model_save_name,
            patience=10,
            use_cyclical=(annealing_type == "cyclical")
        )
    
    # --- STEP 3: TRANSFORMER AGGREGATOR (if using transformer method) ---
    transformer_aggregator = None
    if AGGREGATION_METHOD == "transformer":
        print(f"\n=== TRAINING TRANSFORMER AGGREGATOR ===")
        transformer_aggregator = TransformerVideoAggregator(
            latent_dim=LATENT_DIM,
            n_frames=FRAMES_PER_VIDEO,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(DEVICE)
        
        # Train transformer using reconstruction objective
        # (learn to produce embeddings that can reconstruct frame features)
        transformer_optimizer = optim.Adam(transformer_aggregator.parameters(), lr=1e-4)
        transformer_epochs = 20
        
        print(f"Training transformer for {transformer_epochs} epochs...")
        vae.eval()  # Freeze VAE during transformer training
        
        for epoch in range(transformer_epochs):
            transformer_aggregator.train()
            epoch_loss = 0
            n_batches = 0
            
            for video_frames, _ in train_loader:
                batch_size, n_f, C, H, W = video_frames.shape
                
                # Get frame latents from frozen VAE
                with torch.no_grad():
                    frames_flat = video_frames.view(-1, C, H, W).to(DEVICE)
                    frame_latents = vae.encode(frames_flat)
                    frame_latents = frame_latents.view(batch_size, n_f, -1)
                
                # Forward through transformer
                transformer_optimizer.zero_grad()
                video_emb = transformer_aggregator(frame_latents)
                
                # Loss: video embedding should be close to mean of frames but more informative
                # Using variance-preserving loss to encourage diverse embeddings
                mean_latent = frame_latents.mean(dim=1)
                
                # Reconstruction loss to mean (ensures alignment)
                recon_loss = nn.functional.mse_loss(video_emb, mean_latent)
                
                # Variance loss (encourage video embeddings to maintain variance across batch)
                batch_var = video_emb.var(dim=0).mean()
                var_loss = torch.relu(0.5 - batch_var)  # Penalize if variance < 0.5
                
                loss = recon_loss + 0.1 * var_loss
                loss.backward()
                transformer_optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{transformer_epochs} | Loss: {epoch_loss/n_batches:.4f}")
        
        print("Transformer aggregator training complete!")
        
        # Save transformer
        transformer_save_path = f"Transformer_Aggregator_{full_suffix}.pth"
        torch.save(transformer_aggregator.state_dict(), transformer_save_path)
        print(f"Saved transformer to: {transformer_save_path}")
    
    # --- STEP 4: VIDEO-LEVEL CLUSTERING ---
    video_embeddings, video_ids, clusters = perform_video_clustering_and_log(
        vae, train_loader, LATENT_DIM, BETA,
        aggregation_method=AGGREGATION_METHOD,
        crop_suffix=suffix,
        transformer_model=transformer_aggregator
    )
    
    log_video_reconstruction(vae, val_loader, LATENT_DIM, BETA, crop_suffix=suffix)
    
    # Only log loss graphs if we trained (not when loading pre-trained)
    if tracker is not None:
        log_loss_graphs(tracker, LATENT_DIM, BETA, crop_suffix=suffix)
    
    # --- STEP 5: SAVE RESULTS ---
    os.makedirs("results/video_latent_features", exist_ok=True)
    np.save(f"results/video_latent_features/video_embeddings_{full_suffix}.npy", video_embeddings)
    np.save(f"results/video_latent_features/video_ids_{full_suffix}.npy", np.array(video_ids))
    np.save(f"results/video_latent_features/video_clusters_{full_suffix}.npy", clusters)
    
    print(f"Saved {len(video_embeddings)} video embeddings of dimension {video_embeddings.shape[1]}")
    print(f"Aggregation method: {AGGREGATION_METHOD}")
    
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"video_embeddings_{full_suffix}",
            type="video_latent_features",
            description=f"Video-level embeddings (agg={AGGREGATION_METHOD})"
        )
        artifact.add_file(f"results/video_latent_features/video_embeddings_{full_suffix}.npy")
        artifact.add_file(f"results/video_latent_features/video_ids_{full_suffix}.npy")
        artifact.add_file(f"results/video_latent_features/video_clusters_{full_suffix}.npy")
        if transformer_aggregator is not None:
            artifact.add_file(transformer_save_path)
        wandb.log_artifact(artifact)
    
    wandb.finish()
    
    print("\n" + "="*50)
    print("VIDEO-LEVEL CLUSTERING COMPLETE")
    print("="*50)
    print(f"Total videos processed: {len(video_ids)}")
    print(f"Embedding dimension: {video_embeddings.shape[1]}")
    print(f"Cluster distribution: {np.bincount(clusters)}")
