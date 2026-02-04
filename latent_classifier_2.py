"""
Latent Score Classifier for LUS Video Analysis.
IMPROVED VERSION: Includes Gated Attention, Regression, and Latent Augmentation.

Usage:
    # Train with Regression (MSE) and Attention Pooling
    python latent_classifier.py --vae-checkpoint Best_VAE.pth --regression --aggregation attention --noise-level 0.05

    # Train standard Classifier with Mean Pooling
    python latent_classifier.py --vae-checkpoint Best_VAE.pth --aggregation mean
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error

from config import Config
# from utils import find_score # Assuming this exists in your utils, otherwise removed

# ==========================================
# 1. ATTENTION MODULES (NEW)
# ==========================================

class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning.
    Learns to assign weights to frames based on their importance.
    """
    def __init__(self, latent_dim=32, hidden_dim=128):
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

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_frames, latent_dim)
        Returns:
            video_embedding: (batch_size, latent_dim)
            attention_scores: (batch_size, n_frames, 1)
        """
        # Calculate attention scores
        a_v = self.attention_V(x)  # (B, N, H)
        a_u = self.attention_U(x)  # (B, N, H)
        a = self.attention_weights(a_v * a_u)  # (B, N, 1)
        
        weights = torch.softmax(a, dim=1)  # Normalize weights across frames
        
        # Weighted sum of frame embeddings
        video_embedding = torch.sum(x * weights, dim=1)
        
        return video_embedding

# ==========================================
# 2. CLASSIFIER MODEL (UPDATED)
# ==========================================

class LatentScoreClassifier(nn.Module):
    """
    Predicts LUS scores from video latent embeddings.
    Supports both Classification (4 outputs) and Regression (1 output).
    Supports internal aggregation (Attention) if input is a sequence.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        output_dim: int = 4,  # 4 for classification, 1 for regression
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
        Args:
            x: Latent embeddings. 
               Shape (Batch, Dim) if aggregated externally.
               Shape (Batch, Frames, Dim) if using internal aggregator.
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


# ==========================================
# 3. VAE MODEL (UNCHANGED)
# ==========================================

class ConvVAE(nn.Module):
    """Convolutional VAE for encoding images to latent space."""
    def __init__(self, latent_dim, channels=1, image_size=64):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
        )
        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        return mu

# ==========================================
# 4. DATASETS
# ==========================================

class VideoScoreDataset(Dataset):
    """Standard dataset loading images from disk."""
    def __init__(self, root_dir, metadata_df, transform=None, frames_per_video=10):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.score_map = self._build_score_map(metadata_df)
        self.video_frames = self._group_frames_by_video(root_dir)
        self.video_ids = [
            vid for vid in self.video_frames.keys()
            if self._get_score_for_video(vid) is not None
            and len(self.video_frames[vid]) == frames_per_video
        ]
        print(f"Found {len(self.video_ids)} videos with valid scores")

    def _build_score_map(self, df):
        score_map = {}
        for _, row in df.iterrows():
            if pd.isna(row.get('Score')) or row.get('no_score', False): continue
            fp = str(row['File Path'])
            vid = Path(fp).stem
            try:
                score = int(round(float(row['Score'])))
                score_map[vid] = max(0, min(3, score))
            except: continue
        return score_map

    def _group_frames_by_video(self, root_dir):
        import glob
        all_imgs = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        v_frames = defaultdict(list)
        for img in all_imgs:
            vid = self._extract_video_id(img)
            v_frames[vid].append(img)
        for vid in v_frames:
            v_frames[vid].sort(key=lambda x: self._extract_frame_number(x))
        return v_frames

    def _extract_video_id(self, path):
        fname = os.path.basename(path)
        hospital = next((h for h in ['JCUH', 'MFT', 'UHW'] if h in path), None)
        match = re.match(r"(.+?)_selected_frame_\d+\.png", fname, re.IGNORECASE)
        base_id = match.group(1) if match else os.path.basename(os.path.dirname(path))
        return f"{hospital}_{base_id}" if hospital else base_id

    def _extract_frame_number(self, path):
        match = re.search(r'_selected_frame_(\d+)\.png', os.path.basename(path), re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _get_score_for_video(self, vid):
        if vid in self.score_map: return self.score_map[vid]
        for h in ['JCUH', 'MFT', 'UHW']:
            if vid.startswith(f"{h}_"):
                base = vid[len(h)+1:]
                if base in self.score_map: return self.score_map[base]
        return None

    def __len__(self): return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        paths = self.video_frames[vid][:self.frames_per_video]
        score = self._get_score_for_video(vid)
        frames = []
        for p in paths:
            img = Image.open(p).convert('L')
            if self.transform: img = self.transform(img)
            frames.append(img)
        return torch.stack(frames), score, vid

class EmbeddingScoreDataset(Dataset):
    """Dataset for pre-computed latent embeddings."""
    def __init__(self, embeddings: np.ndarray, scores: np.ndarray, video_ids: List[str]):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        # Keep scores as float for regression compatibility, convert to long later if needed
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.video_ids = video_ids
    
    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx): return self.embeddings[idx], self.scores[idx], self.video_ids[idx]

# ==========================================
# 5. EMBEDDING EXTRACTION (UPDATED)
# ==========================================

def extract_video_embeddings(
    vae_model: nn.Module,
    dataloader: DataLoader,
    aggregation: str = "mean",
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts embeddings. If aggregation is 'sequence' or 'attention', 
    returns (N, Frames, Dim). Otherwise returns (N, Dim).
    """
    vae_model.eval()
    all_emb, all_scores, all_ids = [], [], []
    
    with torch.no_grad():
        for frames, scores, vids in dataloader:
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(-1, C, H, W).to(device)
            latents = vae_model.encode(frames_flat) # (B*T, Latent)
            latents = latents.view(B, T, -1)        # (B, T, Latent)
            
            if aggregation == "mean":
                vid_emb = latents.mean(dim=1)
            elif aggregation == "max":
                vid_emb = latents.max(dim=1)[0]
            elif aggregation == "concat":
                vid_emb = latents.view(B, -1)
            elif aggregation in ["sequence", "attention", "transformer"]:
                # Keep temporal dimension for later processing
                vid_emb = latents
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            all_emb.append(vid_emb.cpu().numpy())
            all_scores.extend(scores.numpy())
            all_ids.extend(vids)
    
    return np.concatenate(all_emb, axis=0), np.array(all_scores), all_ids

# ==========================================
# 6. TRAINING (UPDATED)
# ==========================================

def train_classifier(
    classifier: LatentScoreClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    is_regression: bool = False,
    noise_level: float = 0.0
) -> Dict:
    
    classifier = classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Select Loss Function
    if is_regression:
        criterion = nn.MSELoss()
        print("Using MSE Loss (Regression)")
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropy Loss (Classification)")

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_metric = float('inf') if is_regression else 0
    
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0
        
        for embeddings, scores, _ in train_loader:
            embeddings = embeddings.to(device)
            
            # --- FEATURE 1: Latent Augmentation ---
            if noise_level > 0:
                noise = torch.randn_like(embeddings) * noise_level
                embeddings = embeddings + noise
            # --------------------------------------

            if is_regression:
                scores = scores.float().to(device).unsqueeze(1) # (B, 1)
            else:
                scores = scores.long().to(device) # (B,)

            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        classifier.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for embeddings, scores, _ in val_loader:
                embeddings = embeddings.to(device)
                outputs = classifier(embeddings)
                
                if is_regression:
                    scores_dev = scores.float().to(device).unsqueeze(1)
                    loss = criterion(outputs, scores_dev)
                    
                    # Round for accuracy calculation
                    preds = torch.round(outputs).clamp(0, 3)
                    all_preds.extend(preds.cpu().numpy().flatten())
                else:
                    scores_dev = scores.long().to(device)
                    loss = criterion(outputs, scores_dev)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                
                val_loss += loss.item()
                all_targets.extend(scores.numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate accuracy (works for both reg and clf)
        acc = accuracy_score(all_targets, all_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.1%}")
        
        # Save Best Model Logic
        saved = False
        if is_regression:
            if avg_val_loss < best_metric: # Minimize Loss for Regression
                best_metric = avg_val_loss
                saved = True
        else:
            if acc > best_metric: # Maximize Acc for Classification
                best_metric = acc
                saved = True
                
        if saved and save_path:
            torch.save(classifier.state_dict(), save_path)
            
    if save_path and os.path.exists(save_path):
        classifier.load_state_dict(torch.load(save_path))
    return history

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
    axes[1].plot(epochs, history['val_acc'], label='Validation', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'training_history_2.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved training history to: {save_path}")

def evaluate_classifier(
    classifier: LatentScoreClassifier,
    test_loader: DataLoader,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
    is_regression: bool = False
) -> Dict:
    """Evaluate the classifier and generate confusion matrix."""
    classifier.eval()
    classifier = classifier.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, scores, _ in test_loader:
            embeddings = embeddings.to(device)
            outputs = classifier(embeddings)
            
            if is_regression:
                preds = torch.round(outputs).clamp(0, 3)
                all_preds.extend(preds.cpu().numpy().flatten())
            else:
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(scores.numpy())
    
    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2, 3],
        target_names=['Score 0', 'Score 1', 'Score 2', 'Score 3'],
        zero_division=0
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.1%}")
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
        
        save_path = output_dir / 'confusion_matrix_2.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved confusion matrix to: {save_path}")
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm
    }

# ==========================================
# 7. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/classifier_2")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension of VAE")
    parser.add_argument("--aggregation", type=str, default="mean", 
                        choices=["mean", "max", "attention"], help="Method to combine frames")
    parser.add_argument("--regression", action="store_true", help="Use regression (MSE) instead of classification")
    parser.add_argument("--noise-level", type=float, default=0.0, help="Std dev of noise added to latents during training (e.g. 0.05)")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    
    # Notebook/Interactive support
    args = parser.parse_args([] if hasattr(sys, 'ps1') else None)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Data & VAE
    print(f"Loading VAE: {args.vae_checkpoint}")
    vae = ConvVAE(latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device), strict=False)
    
    data_path = args.data_path or str(Config.VAE_DATA_PATH)
    metadata_df = pd.read_csv(Config.METADATA_PATH)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    
    dataset = VideoScoreDataset(data_path, metadata_df, transform=transform)
    train_size = int(len(dataset) * 0.8)
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. Extract Embeddings
    # If using Attention, we need 'sequence' mode to keep (B, Frames, Dim)
    # If using Mean/Max, we collapse to (B, Dim) here
    extract_mode = "sequence" if args.aggregation == "attention" else args.aggregation
    print(f"Extracting embeddings (Mode: {extract_mode})...")
    
    train_emb, train_y, train_ids = extract_video_embeddings(vae, train_loader, extract_mode, device)
    val_emb, val_y, val_ids = extract_video_embeddings(vae, val_loader, extract_mode, device)
    
    print(f"Train Shape: {train_emb.shape}") # (N, 32) or (N, 10, 32)
    
    train_emb_ds = EmbeddingScoreDataset(train_emb, train_y, train_ids)
    val_emb_ds = EmbeddingScoreDataset(val_emb, val_y, val_ids)
    
    train_emb_loader = DataLoader(train_emb_ds, batch_size=32, shuffle=True)
    val_emb_loader = DataLoader(val_emb_ds, batch_size=32, shuffle=False)
    
    # 3. Setup Classifier
    aggregator = None
    if args.aggregation == "attention":
        print("Initializing Gated Attention Module...")
        aggregator = GatedAttention(latent_dim=args.latent_dim)
    
    # Output dim is 1 for regression, 4 for classification
    out_dim = 1 if args.regression else 4
    
    classifier = LatentScoreClassifier(
        latent_dim=args.latent_dim,
        hidden_dims=[64, 32],
        output_dim=out_dim,
        aggregator=aggregator
    ).to(device)
    
    # 4. Train
    print(f"Starting Training: Regression={args.regression}, Noise={args.noise_level}")
    history = train_classifier(
        classifier, 
        train_emb_loader, 
        val_emb_loader, 
        epochs=args.epochs, 
        device=device,
        save_path=str(output_dir / "best_model_2.pth"),
        is_regression=args.regression,
        noise_level=args.noise_level
    )
    
    # 5. Plot Training History
    plot_training_history(history, output_dir)
    
    # 6. Evaluate and Generate Confusion Matrix
    evaluate_classifier(
        classifier,
        val_emb_loader,
        device=device,
        output_dir=output_dir,
        is_regression=args.regression
    )
    
    print(f"\nDone! Results saved to {output_dir}")

if __name__ == "__main__":
    main()