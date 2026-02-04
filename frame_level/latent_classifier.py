"""
Frame-Level Latent Classifier.

Loads a pre-trained VAE, extracts frame-level latent features,
and trains a classifier to predict LUS scores.

Usage:
    python -m frame_level.latent_classifier --vae_path checkpoints/Best_VAE_ld32.pth --latent_dim 32
    
    # With specific beta
    python frame_level/latent_classifier.py --vae_path ../Best_VAE_ld32_crop10_beta2.0_cyclical.pth
"""

import argparse
import glob
import os
import sys
from pathlib import Path
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

from shared.models import ConvVAE, LatentScoreClassifier
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

class FrameLatentDataset(Dataset):
    """Dataset that extracts latent features from a pre-trained VAE."""
    
    def __init__(self, root_dir, vae_model, metadata_df, transform=None, device='cuda'):
        self.transform = transform
        self.vae_model = vae_model
        self.device = device
        
        image_paths = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        
        self.samples = []
        for img_path in image_paths:
            score = find_score(img_path, metadata_df)
            if not np.isnan(score):
                self.samples.append((img_path, int(score)))
        
        print(f"Found {len(self.samples)} frames with valid scores")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, score = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Extract latent
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(self.device)
            latent = self.vae_model.encode(image_tensor).squeeze(0).cpu()
        
        return latent, score


class PrecomputedLatentDataset(Dataset):
    """Dataset from pre-computed latent vectors."""
    
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
            torch.save(model.state_dict(), CHECKPOINTS_DIR / 'best_frame_classifier.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / 'best_frame_classifier.pth'))
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
    ax.set_title('Frame-Level Classifier Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'frame_classifier_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to: {RESULTS_DIR / 'frame_classifier_confusion_matrix.png'}")
    
    return accuracy_score(all_labels, all_preds)


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Frame-Level Latent Classifier")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to pre-trained VAE")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--crop_percent", type=float, default=0.1)
    parser.add_argument("--precomputed_latents", type=str, default=None, 
                        help="Path to pre-computed latent vectors (.npy)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Load metadata
    import pandas as pd
    metadata_df = pd.read_csv(Config.METADATA_PATH)
    metadata_df['video_id'] = metadata_df['File Path'].apply(lambda p: Path(str(p)).stem)
    
    if args.precomputed_latents:
        # Use pre-computed latents
        print(f"Loading pre-computed latents from: {args.precomputed_latents}")
        latent_vectors = np.load(args.precomputed_latents)
        
        # Load corresponding paths to get scores
        paths_file = args.precomputed_latents.replace('latent_vectors', 'image_paths')
        image_paths = np.load(paths_file, allow_pickle=True)
        
        scores = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            score = find_score(str(path), metadata_df)
            if not np.isnan(score):
                scores.append(int(score))
                valid_indices.append(i)
        
        latent_vectors = latent_vectors[valid_indices]
        print(f"Valid samples with scores: {len(scores)}")
        
        dataset = PrecomputedLatentDataset(latent_vectors, scores)
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
        dataset = FrameLatentDataset(data_path, vae, metadata_df, transform, device)
    
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
    
    # Create classifier
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
    plt.savefig(RESULTS_DIR / 'frame_classifier_training.png')
    plt.close()
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
