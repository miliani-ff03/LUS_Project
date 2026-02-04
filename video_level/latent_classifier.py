"""
Video-Level Latent Classifier.

Loads a pre-trained VAE, extracts frame-level latent features,
aggregates them to video-level, and trains a classifier to predict LUS scores.

Usage:
    python -m video_level.latent_classifier --vae_path ../Best_VAE_ld32_crop10_beta2.0_cyclical.pth
    
    # With specific aggregation
    python video_level/latent_classifier.py --vae_path path/to/vae.pth --aggregation mean
    
    # With transformer aggregation
    python video_level/latent_classifier.py --vae_path path/to/vae.pth --aggregation attention
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import (
    ConvVAE, 
    LatentScoreClassifier, 
    GatedAttention,
    aggregate_frame_latents
)
from shared.config import Config
from shared.utils import find_score

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


# ==========================================
# DATASET
# ==========================================

class VideoLatentDataset(Dataset):
    """Dataset that extracts and aggregates video-level latent features from a pre-trained VAE."""
    
    def __init__(self, root_dir, vae_model, metadata_df, transform=None, 
                 frames_per_video=10, aggregation='mean', device='cuda'):
        self.transform = transform
        self.vae_model = vae_model
        self.device = device
        self.frames_per_video = frames_per_video
        self.aggregation = aggregation
        
        # Group images by video ID
        all_images = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        
        video_frames = defaultdict(list)
        for img_path in all_images:
            video_id = self._extract_video_id(img_path)
            video_frames[video_id].append(img_path)
        
        # Filter to complete videos with valid scores
        self.samples = []
        for video_id, frames in video_frames.items():
            frames.sort(key=lambda x: self._extract_frame_number(x))
            
            if len(frames) != frames_per_video:
                continue
            
            score = find_score(frames[0], metadata_df)
            if not np.isnan(score):
                self.samples.append((video_id, frames[:frames_per_video], int(score)))
        
        print(f"Found {len(self.samples)} complete videos with valid scores")
    
    def _extract_video_id(self, path):
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
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, frame_paths, score = self.samples[idx]
        
        # Load and transform all frames
        frames = []
        for path in frame_paths:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        frames_tensor = torch.stack(frames)
        
        # Extract latents and aggregate
        with torch.no_grad():
            frames_batch = frames_tensor.to(self.device)
            frame_latents = self.vae_model.encode(frames_batch)  # (n_frames, latent_dim)
            
            if self.aggregation == 'none':
                # Return full sequence for models with internal aggregation
                return frame_latents.cpu(), score
            else:
                # Aggregate to single vector
                video_latent = aggregate_frame_latents(frame_latents, method=self.aggregation)
                return video_latent.cpu(), score


class PrecomputedVideoLatentDataset(Dataset):
    """Dataset from pre-computed video-level latent vectors."""
    
    def __init__(self, latent_vectors, scores):
        self.latents = torch.tensor(latent_vectors, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.long)
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.scores[idx]


# ==========================================
# TRAINING
# ==========================================

def train_classifier(model, train_loader, val_loader, epochs=50, learning_rate=1e-3, device='cuda'):
    """Train the latent classifier."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for latents, labels in train_loader:
            latents, labels = latents.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(latents)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for latents, labels in val_loader:
                latents, labels = latents.to(device), labels.to(device)
                outputs = model(latents)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINTS_DIR / 'best_video_classifier.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / 'best_video_classifier.pth'))
    return model, history


def evaluate_classifier(model, test_loader, device='cuda'):
    """Evaluate classifier and generate reports."""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for latents, labels in test_loader:
            latents = latents.to(device)
            outputs = model(latents)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Score 0', 'Score 1', 'Score 2', 'Score 3']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2', '3'],
                yticklabels=['0', '1', '2', '3'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Video-Level Classifier Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'video_classifier_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to: {RESULTS_DIR / 'video_classifier_confusion_matrix.png'}")
    
    return accuracy_score(all_labels, all_preds)


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Video-Level Latent Classifier")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to pre-trained VAE")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--crop_percent", type=float, default=0.1)
    parser.add_argument("--frames_per_video", type=int, default=10)
    parser.add_argument("--aggregation", type=str, default="mean", 
                        choices=["mean", "max", "attention", "none"],
                        help="Aggregation method (none = use internal attention)")
    parser.add_argument("--precomputed_latents", type=str, default=None,
                        help="Path to pre-computed video latent vectors (.npy)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Aggregation method: {args.aggregation}")
    
    # Load metadata
    import pandas as pd
    metadata_df = pd.read_csv(Config.METADATA_PATH)
    metadata_df['video_id'] = metadata_df['File Path'].apply(lambda p: Path(str(p)).stem)
    
    # Determine if using internal or external aggregation
    use_internal_aggregation = args.aggregation in ['attention', 'none']
    
    if args.precomputed_latents:
        # Use pre-computed latents
        print(f"Loading pre-computed video latents from: {args.precomputed_latents}")
        latent_vectors = np.load(args.precomputed_latents)
        
        # Load corresponding video IDs to get scores
        ids_file = args.precomputed_latents.replace('video_embeddings', 'video_ids')
        video_ids = np.load(ids_file, allow_pickle=True)
        
        scores = []
        valid_indices = []
        for i, vid in enumerate(video_ids):
            score = find_score(str(vid), metadata_df)
            if not np.isnan(score):
                scores.append(int(score))
                valid_indices.append(i)
        
        latent_vectors = latent_vectors[valid_indices]
        print(f"Valid videos with scores: {len(scores)}")
        
        dataset = PrecomputedVideoLatentDataset(latent_vectors, scores)
    else:
        # Load VAE and extract latents on-the-fly
        print(f"Loading VAE from: {args.vae_path}")
        vae = ConvVAE(latent_dim=args.latent_dim, channels=CHANNELS)
        vae.load_state_dict(torch.load(args.vae_path, map_location=device))
        vae = vae.to(device)
        vae.eval()
        
        # Transform
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
        
        data_path = args.data_path or str(Config.VAE_DATA_PATH)
        
        # Use 'none' for internal aggregation, otherwise use specified method
        agg_for_dataset = 'none' if use_internal_aggregation else args.aggregation
        
        dataset = VideoLatentDataset(
            data_path, vae, metadata_df, transform,
            frames_per_video=args.frames_per_video,
            aggregation=agg_for_dataset,
            device=device
        )
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Data split: {train_size} train, {val_size} val, {test_size} test")
    
    # Create classifier with optional internal aggregation
    if use_internal_aggregation:
        print("Using classifier with internal Gated Attention aggregation")
        aggregator = GatedAttention(latent_dim=args.latent_dim, hidden_dim=128)
        classifier = LatentScoreClassifier(
            latent_dim=args.latent_dim,
            hidden_dims=[64, 32],
            output_dim=4,
            dropout=0.3,
            aggregator=aggregator
        )
    else:
        classifier = LatentScoreClassifier(
            latent_dim=args.latent_dim,
            hidden_dims=[64, 32],
            output_dim=4,
            dropout=0.3
        )
    
    # Train
    classifier, history = train_classifier(
        classifier, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Evaluate
    test_acc = evaluate_classifier(classifier, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training/Validation Loss')
    axes[0].legend()
    
    axes[1].plot(history['val_acc'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'video_classifier_training_{args.aggregation}.png')
    plt.close()
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
