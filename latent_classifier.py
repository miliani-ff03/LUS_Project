"""
Latent Score Classifier for LUS Video Analysis.

This module provides a trainable neural network classifier that predicts
LUS scores (0-3) from VAE video latent representations.

Components:
    - LatentScoreClassifier: MLP classifier for score prediction
    - VideoScoreDataset: Dataset that maps videos to scores
    - Training/evaluation utilities

Usage:
    # Train a classifier from pre-computed embeddings
    python latent_classifier.py --vae-checkpoint Best_VAE_ld32_crop10_beta2.0_cyclical.pth

    # Or use as a module
    from latent_classifier import LatentScoreClassifier, train_classifier
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, List
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from config import Config
from utils import find_score


# ==========================================
# 1. CLASSIFIER MODEL
# ==========================================

class LatentScoreClassifier(nn.Module):
    """
    MLP classifier that predicts LUS scores from video latent embeddings.
    
    Architecture:
        latent_dim -> hidden_dims -> num_classes
        With BatchNorm and Dropout for regularization.
    
    Args:
        latent_dim: Dimension of input latent vectors (default: 32)
        hidden_dims: List of hidden layer dimensions (default: [64, 32])
        num_classes: Number of output classes (default: 4 for scores 0-3)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
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
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Latent embeddings of shape (batch_size, latent_dim)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)
    
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


# ==========================================
# 2. TRANSFORMER VIDEO AGGREGATOR
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


# ==========================================
# 3. VAE MODEL (copied from Medical_VAE_Video_Clustering)
# ==========================================

class ConvVAE(nn.Module):
    """Convolutional VAE for encoding images to latent space."""
    
    def __init__(self, latent_dim, channels=1, image_size=64):
        super(ConvVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
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
# 4. VIDEO SCORE DATASET
# ==========================================

class VideoScoreDataset(Dataset):
    """
    Dataset that loads video frames and maps them to LUS scores.
    
    Each sample returns:
        - video_frames: Tensor of shape (n_frames, C, H, W)
        - score: Integer label (0-3)
        - video_id: String identifier
    
    Args:
        root_dir: Path to image directory
        metadata_df: DataFrame with 'File Path' and 'Score' columns
        transform: Image transforms to apply
        frames_per_video: Number of frames per video
    """
    
    def __init__(
        self,
        root_dir: str,
        metadata_df: pd.DataFrame,
        transform=None,
        frames_per_video: int = 10
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Build video_id -> score mapping from metadata
        self.score_map = self._build_score_map(metadata_df)
        
        # Find all images and group by video
        self.video_frames = self._group_frames_by_video(root_dir)
        
        # Filter to videos with valid scores
        self.video_ids = [
            vid for vid in self.video_frames.keys()
            if self._get_score_for_video(vid) is not None
            and len(self.video_frames[vid]) == frames_per_video
        ]
        
        print(f"Found {len(self.video_ids)} videos with valid scores")
        self._print_score_distribution()
    
    def _build_score_map(self, df: pd.DataFrame) -> Dict[str, int]:
        """Build mapping from video basename to score."""
        score_map = {}
        for _, row in df.iterrows():
            if pd.isna(row.get('Score')) or row.get('no_score', False):
                continue
            
            # Get video basename from file path
            file_path = str(row['File Path'])
            video_id = Path(file_path).stem
            
            # Round score to nearest integer
            try:
                score = int(round(float(row['Score'])))
                score = max(0, min(3, score))  # Clamp to 0-3
                score_map[video_id] = score
            except (ValueError, TypeError):
                continue
        
        return score_map
    
    def _group_frames_by_video(self, root_dir: str) -> Dict[str, List[str]]:
        """Group image paths by video ID."""
        import glob
        
        all_images = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        
        video_frames = defaultdict(list)
        for img_path in all_images:
            video_id = self._extract_video_id(img_path)
            video_frames[video_id].append(img_path)
        
        # Sort frames within each video
        for video_id in video_frames:
            video_frames[video_id].sort(key=lambda x: self._extract_frame_number(x))
        
        return video_frames
    
    def _extract_video_id(self, path: str) -> str:
        """Extract video ID from image path (hospital-prefixed for uniqueness)."""
        filename = os.path.basename(path)
        
        # Detect hospital
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
        
        if hospital:
            return f"{hospital}_{base_id}"
        return base_id
    
    def _extract_frame_number(self, path: str) -> int:
        """Extract frame number from filename."""
        filename = os.path.basename(path)
        match = re.search(r'_selected_frame_(\d+)\.png', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'(\d+)', filename)
        if numbers:
            return int(numbers[-1])
        return 0
    
    def _get_score_for_video(self, video_id: str) -> Optional[int]:
        """Get score for a video ID, handling hospital prefix variations."""
        # Direct match
        if video_id in self.score_map:
            return self.score_map[video_id]
        
        # Try without hospital prefix
        for hospital in ['JCUH', 'MFT', 'UHW']:
            if video_id.startswith(f"{hospital}_"):
                base_id = video_id[len(hospital) + 1:]
                if base_id in self.score_map:
                    return self.score_map[base_id]
        
        return None
    
    def _print_score_distribution(self):
        """Print the distribution of scores."""
        scores = [self._get_score_for_video(vid) for vid in self.video_ids]
        unique, counts = np.unique(scores, return_counts=True)
        print("Score distribution:")
        for s, c in zip(unique, counts):
            print(f"  Score {s}: {c} videos ({100*c/len(scores):.1f}%)")
    
    def __len__(self) -> int:
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        video_id = self.video_ids[idx]
        frame_paths = self.video_frames[video_id][:self.frames_per_video]
        score = self._get_score_for_video(video_id)
        
        # Load and transform frames
        frames = []
        for path in frame_paths:
            image = Image.open(path).convert('L')  # Load as grayscale directly
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        frames_tensor = torch.stack(frames)
        return frames_tensor, score, video_id


class EmbeddingScoreDataset(Dataset):
    """
    Dataset for pre-computed latent embeddings with scores.
    
    This is faster than VideoScoreDataset since embeddings are pre-computed.
    
    Args:
        embeddings: Array of shape (n_videos, latent_dim)
        scores: Array of integer labels (n_videos,)
        video_ids: List of video identifiers
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        video_ids: List[str]
    ):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.long)
        self.video_ids = video_ids
        
        assert len(self.embeddings) == len(self.scores) == len(video_ids)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        return self.embeddings[idx], self.scores[idx], self.video_ids[idx]


# ==========================================
# 5. EMBEDDING EXTRACTION
# ==========================================

def extract_video_embeddings(
    vae_model: nn.Module,
    dataloader: DataLoader,
    aggregation: str = "mean",
    device: str = "cuda",
    transformer_model: Optional[nn.Module] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract video-level embeddings from a VAE.
    
    Args:
        vae_model: Trained VAE with encode() method
        dataloader: DataLoader yielding (video_frames, scores, video_ids)
        aggregation: 'mean', 'max', 'concat', or 'transformer'
        device: Device to use for computation
        transformer_model: Required if aggregation='transformer', instance of TransformerVideoAggregator
    
    Returns:
        embeddings: Array of shape (n_videos, latent_dim)
        scores: Array of integer labels
        video_ids: List of video identifiers
    """
    vae_model.eval()
    if transformer_model is not None:
        transformer_model.eval()
    
    all_embeddings = []
    all_scores = []
    all_video_ids = []
    
    with torch.no_grad():
        for video_frames, scores, video_ids in dataloader:
            batch_size, n_frames, C, H, W = video_frames.shape
            
            # Flatten frames and encode
            frames_flat = video_frames.view(-1, C, H, W).to(device)
            frame_latents = vae_model.encode(frames_flat)
            
            # Reshape to (batch, n_frames, latent_dim)
            frame_latents = frame_latents.view(batch_size, n_frames, -1)
            
            # Aggregate to video level
            if aggregation == "mean":
                video_emb = frame_latents.mean(dim=1)
            elif aggregation == "max":
                video_emb = frame_latents.max(dim=1)[0]
            elif aggregation == "concat":
                video_emb = frame_latents.view(batch_size, -1)
            elif aggregation == "transformer":
                if transformer_model is None:
                    raise ValueError("transformer_model required for transformer aggregation")
                video_emb = transformer_model(frame_latents)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            all_embeddings.append(video_emb.cpu().numpy())
            all_scores.extend(scores.numpy() if isinstance(scores, torch.Tensor) else scores)
            all_video_ids.extend(video_ids)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    scores = np.array(all_scores)
    
    return embeddings, scores, all_video_ids


# ==========================================
# 6. TRAINING AND EVALUATION
# ==========================================

def train_classifier(
    classifier: LatentScoreClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    save_path: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None
) -> Dict:
    """
    Train the latent score classifier.
    
    Args:
        classifier: LatentScoreClassifier model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to use
        save_path: Path to save best model
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Dictionary with training history
    """
    classifier = classifier.to(device)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for embeddings, scores, _ in train_loader:
            embeddings = embeddings.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            logits = classifier(embeddings)
            loss = criterion(logits, scores)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * embeddings.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == scores).sum().item()
            train_total += embeddings.size(0)
        
        # Validation phase
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, scores, _ in val_loader:
                embeddings = embeddings.to(device)
                scores = scores.to(device)
                
                logits = classifier(embeddings)
                loss = criterion(logits, scores)
                
                val_loss += loss.item() * embeddings.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == scores).sum().item()
                val_total += embeddings.size(0)
        
        # Compute metrics
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1%} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1%}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(classifier.state_dict(), save_path)
                print(f"  -> New best model saved (val_acc={val_acc:.1%})")
    
    # Load best model
    if save_path and os.path.exists(save_path):
        classifier.load_state_dict(torch.load(save_path))
    
    return history


def evaluate_classifier(
    classifier: LatentScoreClassifier,
    test_loader: DataLoader,
    device: str = "cuda",
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Evaluate the classifier and generate reports.
    
    Args:
        classifier: Trained classifier
        test_loader: Test dataloader
        device: Device to use
        output_dir: Directory to save plots
    
    Returns:
        Dictionary with evaluation metrics
    """
    classifier.eval()
    classifier = classifier.to(device)
    
    all_preds = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        for embeddings, scores, video_ids in test_loader:
            embeddings = embeddings.to(device)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(scores.numpy() if isinstance(scores, torch.Tensor) else scores)
            all_video_ids.extend(video_ids)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    # Specify labels explicitly to handle cases where not all classes appear in test set
    report = classification_report(
        all_labels, all_preds, 
        labels=[0, 1, 2, 3],
        target_names=['Score 0', 'Score 1', 'Score 2', 'Score 3'],
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(f"\nOverall Accuracy: {accuracy:.1%}")
    print(f"\n{report}")
    
    # Plot confusion matrix
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1', '2', '3'],
            yticklabels=['0', '1', '2', '3'],
            ax=ax
        )
        ax.set_xlabel('Predicted Score')
        ax.set_ylabel('True Score')
        ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.1%})')
        
        save_path = output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\nSaved confusion matrix to: {save_path}")
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'video_ids': all_video_ids,
        'confusion_matrix': cm,
        'report': report
    }


def plot_training_history(history: Dict, output_dir: Path):
    """Plot and save training curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train', color='blue')
    axes[0].plot(epochs, history['val_loss'], label='Validation', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], label='Train', color='blue')
    axes[1].plot(epochs, history['val_acc'], label='Validation', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved training history to: {save_path}")


def compute_class_weights(scores: np.ndarray) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced data."""
    unique, counts = np.unique(scores, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(unique)  # Normalize
    
    # Create full weight tensor (in case some classes are missing)
    full_weights = torch.ones(4)
    for cls, weight in zip(unique, weights):
        full_weights[int(cls)] = weight
    
    return full_weights


# ==========================================
# 7. MAIN SCRIPT
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train latent score classifier")
    
    # Model paths
    parser.add_argument("--vae-checkpoint", type=str, required=True,
                       help="Path to trained VAE checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/classifier",
                       help="Output directory for results")
    
    # Model settings
    parser.add_argument("--latent-dim", type=int, default=32,
                       help="Latent dimension of VAE")
    parser.add_argument("--aggregation", type=str, default="mean",
                       choices=["mean", "max", "concat", "transformer"],
                       help="Video aggregation method")
    parser.add_argument("--frames-per-video", type=int, default=10,
                       help="Frames per video")
    parser.add_argument("--transformer-checkpoint", type=str, default=None,
                       help="Path to trained transformer aggregator (required if aggregation=transformer)")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    
    # Data settings
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to image data (default: from Config)")
    parser.add_argument("--crop-percent", type=float, default=0.1,
                       help="Crop percentage for images")
    
    # Options
    parser.add_argument("--use-class-weights", action="store_true",
                       help="Use inverse frequency class weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    # Handle interactive environments (Jupyter, IPython)
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        return parser.parse_args([])
    else:
        return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LATENT SCORE CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = args.data_path or str(Config.VAE_DATA_PATH)
    
    print(f"\nConfiguration:")
    print(f"  VAE checkpoint: {args.vae_checkpoint}")
    print(f"  Data path: {data_path}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Device: {args.device}")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata_df = pd.read_csv(Config.METADATA_PATH)
    
    # Build image transforms
    IMAGE_SIZE = 64
    CHANNELS = 1  # Must match VAE training (grayscale)
    transform_ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    
    if args.crop_percent > 0:
        transform_ops.append(
            transforms.Lambda(lambda img: transforms.functional.crop(
                img,
                top=int(img.size[1] * args.crop_percent),
                left=0,
                height=int(img.size[1] * (1 - args.crop_percent)),
                width=img.size[0]
            ))
        )
    
    transform_ops.extend([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=CHANNELS),
        transforms.ToTensor()
    ])
    
    transform = transforms.Compose(transform_ops)
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = VideoScoreDataset(
        root_dir=data_path,
        metadata_df=metadata_df,
        transform=transform,
        frames_per_video=args.frames_per_video
    )
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nData split: {train_size} train, {val_size} validation")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )
    
    # Load VAE
    print(f"\nLoading VAE from {args.vae_checkpoint}...")
    
    vae = ConvVAE(latent_dim=args.latent_dim).to(args.device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=args.device))
    vae.eval()
    
    # Load transformer aggregator if needed
    transformer_model = None
    if args.aggregation == "transformer":
        if args.transformer_checkpoint is None:
            raise ValueError("--transformer-checkpoint required when using transformer aggregation")
        
        print(f"\nLoading transformer aggregator from {args.transformer_checkpoint}...")
        transformer_model = TransformerVideoAggregator(
            latent_dim=args.latent_dim,
            n_frames=args.frames_per_video,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(args.device)
        transformer_model.load_state_dict(torch.load(args.transformer_checkpoint, map_location=args.device))
        transformer_model.eval()
        print("Transformer aggregator loaded successfully!")
    
    # Extract embeddings
    print("\nExtracting video embeddings...")
    
    train_embeddings, train_scores, train_ids = extract_video_embeddings(
        vae, train_loader, aggregation=args.aggregation, device=args.device,
        transformer_model=transformer_model
    )
    val_embeddings, val_scores, val_ids = extract_video_embeddings(
        vae, val_loader, aggregation=args.aggregation, device=args.device,
        transformer_model=transformer_model
    )
    
    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Val embeddings: {val_embeddings.shape}")
    
    # Create embedding datasets
    train_emb_dataset = EmbeddingScoreDataset(train_embeddings, train_scores, train_ids)
    val_emb_dataset = EmbeddingScoreDataset(val_embeddings, val_scores, val_ids)
    
    train_emb_loader = DataLoader(train_emb_dataset, batch_size=args.batch_size, shuffle=True)
    val_emb_loader = DataLoader(val_emb_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_scores)
        print(f"\nClass weights: {class_weights.tolist()}")
    
    # Create classifier
    latent_dim = train_embeddings.shape[1]
    classifier = LatentScoreClassifier(
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        num_classes=4,
        dropout=0.3
    )
    
    print(f"\nClassifier architecture:")
    print(classifier)
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    save_path = output_dir / "best_classifier.pth"
    
    history = train_classifier(
        classifier=classifier,
        train_loader=train_emb_loader,
        val_loader=val_emb_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        save_path=str(save_path),
        class_weights=class_weights
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    results = evaluate_classifier(
        classifier=classifier,
        test_loader=val_emb_loader,
        device=args.device,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
