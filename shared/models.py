"""
Shared model architectures for Medical VAE Clustering project.

This module contains all neural network models and training utilities
used by both frame_level and video_level analysis packages.

Usage:
    from shared.models import ConvVAE, TransformerVideoAggregator, LatentScoreClassifier
"""

import torch
import torch.nn as nn
from typing import List, Optional


# ==========================================
# 1. CONVOLUTIONAL VAE
# ==========================================

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for encoding medical images to latent space.
    
    Args:
        latent_dim: Size of the latent space (default: 32)
        channels: Number of input channels, 1 for grayscale (default: 1)
        image_size: Input image size (default: 64)
    """
    def __init__(self, latent_dim: int = 32, channels: int = 1, image_size: int = 64):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels
        
        # ENCODER: Compresses image -> hidden representation
        self.encoder = nn.Sequential(
            # Input: (B, C, 64, 64)
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 4, 4)
            nn.ReLU(),
        )
        
        # LATENT SPACE: Mean and Log-Variance
        # Flatten size: 256 channels * 4 * 4 pixels = 4096
        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # DECODER: Expands hidden representation -> reconstructed image
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),  # -> (C, 64, 64)
            nn.Sigmoid()  # Pixel values between 0 and 1
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: adds noise during training to force continuity,
        but allows backpropagation to work.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            reconstruction: Reconstructed image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        # Latent sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z_input = self.decoder_input(z)
        z_matrix = z_input.view(z_input.size(0), 256, 4, 4)
        reconstruction = self.decoder(z_matrix)
        
        return reconstruction, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space (returns mu only for inference)."""
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        return mu


# ==========================================
# 2. LOSS FUNCTION
# ==========================================

def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor, 
                      beta: float = 1.0):
    """
    VAE Loss = Reconstruction Error + KL Divergence
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
        
    Returns:
        total_loss, reconstruction_loss, kl_divergence
    """
    batch_size = x.size(0)
    # BCE: sum over pixels per image, then mean over batch
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size

    # KLD: sum over latent dims per image, then mean over batch
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = BCE + beta * KLD
    return total_loss, BCE, KLD


# ==========================================
# 3. TRAINING UTILITIES
# ==========================================

class KLAnnealer:
    """Linearly anneals beta from start_beta to end_beta over warmup epochs."""
    
    def __init__(self, total_epochs: int, start_beta: float = 0.0, end_beta: float = 1.0):
        self.total_epochs = total_epochs
        self.warmup_epochs = max(1, total_epochs // 5)
        self.start_beta = start_beta
        self.end_beta = end_beta
    
    def get_beta(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.start_beta + (self.end_beta - self.start_beta) * (epoch / self.warmup_epochs)
        return self.end_beta


class CyclicalAnnealer:
    """Cyclically anneals beta between start_beta and end_beta."""
    
    def __init__(self, total_epochs: int, cycles: int = 4, 
                 start_beta: float = 0.0, end_beta: float = 1.0):
        self.total_epochs = total_epochs
        self.cycles = cycles
        self.cycle_length = total_epochs // cycles
        self.start_beta = start_beta
        self.end_beta = end_beta
    
    def get_beta(self, epoch: int) -> float:
        cycle_epoch = epoch % self.cycle_length
        half_cycle = self.cycle_length / 2
        if cycle_epoch < half_cycle:
            return self.start_beta + (self.end_beta - self.start_beta) * (cycle_epoch / half_cycle)
        return self.end_beta


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after given patience."""
    
    def __init__(self, patience: int = 10, verbose: bool = False, 
                 delta: float = 0.01, path: str = 'best_model.pth'):
        """
        Args:
            patience: How long to wait after last improvement.
            verbose: If True, prints messages for each improvement.
            delta: Minimum change to qualify as an improvement.
            path: Path to save the checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module):
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

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LossTracker:
    """Tracks training losses over epochs."""
    
    def __init__(self):
        self.history = {
            "loss": [], 
            "reconstruction_loss": [], 
            "kl_loss": [], 
            "validation_loss": []
        }

    def add(self, total: float, recon: float, kl: float, val: float):
        self.history["loss"].append(total)
        self.history["reconstruction_loss"].append(recon)
        self.history["kl_loss"].append(kl)
        self.history["validation_loss"].append(val)


# ==========================================
# 4. VIDEO AGGREGATION
# ==========================================

class TransformerVideoAggregator(nn.Module):
    """
    Aggregates frame-level latent vectors into a single video embedding using self-attention.
    
    Uses a learnable CLS token (like BERT) to aggregate information from all frames.
    
    Args:
        latent_dim: Dimension of frame latent vectors (default: 32)
        n_frames: Number of frames per video (default: 10)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer encoder layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, latent_dim: int = 32, n_frames: int = 10, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
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
        
        # Output projection head
        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, frame_latents: torch.Tensor) -> torch.Tensor:
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
    
    def get_attention_weights(self, frame_latents: torch.Tensor) -> torch.Tensor:
        """Get attention weights to see which frames the model focuses on."""
        B = frame_latents.size(0)
        # Placeholder - full implementation would extract from transformer layers
        return torch.ones(B, self.n_frames) / self.n_frames


class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning.
    Learns to assign weights to frames based on their importance.
    """
    
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_frames, latent_dim)
        Returns:
            video_embedding: (batch_size, latent_dim)
        """
        # Calculate attention scores
        a_v = self.attention_V(x)  # (B, N, H)
        a_u = self.attention_U(x)  # (B, N, H)
        a = self.attention_weights(a_v * a_u)  # (B, N, 1)
        
        weights = torch.softmax(a, dim=1)  # Normalize weights across frames
        
        # Weighted sum of frame embeddings
        video_embedding = torch.sum(x * weights, dim=1)
        
        return video_embedding


def aggregate_frame_latents(frame_latents: torch.Tensor, method: str = "mean", 
                           transformer_model: Optional[nn.Module] = None) -> torch.Tensor:
    """
    Aggregate frame-level latent vectors into a single video-level embedding.
    
    Args:
        frame_latents: Tensor of shape (n_frames, latent_dim) or (batch, n_frames, latent_dim)
        method: 'mean', 'max', 'concat', or 'transformer'
        transformer_model: Required if method='transformer'
    
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
# 5. CLASSIFIER
# ==========================================

class LatentScoreClassifier(nn.Module):
    """
    MLP classifier that predicts LUS scores from video latent embeddings.
    Supports both Classification (4 outputs) and Regression (1 output).
    Supports internal aggregation (Attention) if input is a sequence.
    
    Args:
        latent_dim: Dimension of input embeddings (default: 32)
        hidden_dims: List of hidden layer dimensions (default: [64, 32])
        output_dim: 4 for classification, 1 for regression (default: 4)
        dropout: Dropout rate (default: 0.3)
        aggregator: Optional module to aggregate sequence inputs
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 4,
        dropout: float = 0.3,
        aggregator: Optional[nn.Module] = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.aggregator = aggregator
        self.latent_dim = latent_dim
        
        # Build MLP layers
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # Output layer
        self.head = nn.Linear(in_dim, output_dim)
        self.features = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Latent embeddings of shape (batch, dim) or (batch, frames, dim)
        
        Returns:
            Logits of shape (batch, output_dim)
        """
        # Handle sequence input if aggregator is present
        if x.dim() == 3:
            if self.aggregator is None:
                # Default to mean if 3D input but no aggregator provided
                x = x.mean(dim=1)
            else:
                x = self.aggregator(x)
        
        # Pass through MLP
        feat = self.features(x)
        logits = self.head(feat)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
