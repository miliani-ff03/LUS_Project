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
import pandas as pd

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
                 frames_per_video=10, aggregation='mean', device='cuda',
                 return_video_id=False):
        self.transform = transform
        self.vae_model = vae_model
        self.device = device
        self.frames_per_video = frames_per_video
        self.aggregation = aggregation
        self.return_video_id = return_video_id
        
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
                if self.return_video_id:
                    return frame_latents.cpu(), score, video_id
                return frame_latents.cpu(), score
            else:
                # Aggregate to single vector
                video_latent = aggregate_frame_latents(frame_latents, method=self.aggregation)
                if self.return_video_id:
                    return video_latent.cpu(), score, video_id
                return video_latent.cpu(), score


class PrecomputedVideoLatentDataset(Dataset):
    """Dataset from pre-computed video-level latent vectors."""
    
    def __init__(self, latent_vectors, scores, video_ids=None):
        self.latents = torch.tensor(latent_vectors, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.long)
        self.video_ids = video_ids if video_ids is not None else [f"video_{i}" for i in range(len(scores))]
        self.return_video_id = video_ids is not None
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        if self.return_video_id:
            return self.latents[idx], self.scores[idx], self.video_ids[idx]
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
        
        for batch in train_loader:
            # Handle both (latent, label) and (latent, label, video_id) formats
            latents, labels = batch[0], batch[1]
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
            for batch in val_loader:
                # Handle both (latent, label) and (latent, label, video_id) formats
                latents, labels = batch[0], batch[1]
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


def evaluate_classifier(model, test_loader, device='cuda', save_predictions=True):
    """Evaluate classifier and generate reports with confidence analysis."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store softmax probabilities
    all_video_ids = []  # Store video IDs if available
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle both (latent, label) and (latent, label, video_id) formats
            if len(batch) == 3:
                latents, labels, video_ids = batch
                all_video_ids.extend(video_ids)
            else:
                latents, labels = batch
            
            latents = latents.to(device)
            outputs = model(latents)
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
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
    
    # === CONFIDENCE ANALYSIS ===
    video_ids = all_video_ids if all_video_ids else None
    analyze_confidence(all_probs, all_preds, all_labels, RESULTS_DIR, 
                       prefix='video', video_ids=video_ids)
    
    return accuracy_score(all_labels, all_preds)


def analyze_confidence(probs, preds, labels, results_dir, prefix='video', video_ids=None):
    """
    Analyze prediction confidence to identify hard-to-classify videos.
    
    Args:
        probs: (N, 4) array of softmax probabilities
        preds: (N,) array of predicted labels
        labels: (N,) array of true labels
        results_dir: Path to save results
        prefix: 'frame' or 'video' for filenames
        video_ids: Optional list of video identifiers
    """
    # Get confidence (max probability) for each prediction
    confidence = np.max(probs, axis=1)
    
    # Create per-sample DataFrame
    df = pd.DataFrame({
        'true_label': labels,
        'predicted': preds,
        'correct': labels == preds,
        'confidence': confidence,
        'prob_score_0': probs[:, 0],
        'prob_score_1': probs[:, 1],
        'prob_score_2': probs[:, 2],
        'prob_score_3': probs[:, 3],
    })
    
    # Add video IDs if available
    if video_ids is not None:
        df.insert(0, 'video_id', video_ids)
    
    # Add entropy (uncertainty measure)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    df['entropy'] = entropy
    
    # Add margin (difference between top two probabilities)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    df['margin'] = margin
    
    # Sort by confidence (lowest first = hardest to classify)
    df_sorted = df.sort_values('confidence', ascending=True)
    df_sorted.to_csv(results_dir / f'{prefix}_predictions_with_confidence.csv', index=False)
    
    print(f"\n{'='*60}")
    print("CONFIDENCE ANALYSIS")
    print(f"{'='*60}")
    
    # 1. Overall confidence statistics
    print(f"\nOverall Confidence Statistics:")
    print(f"  Mean confidence: {confidence.mean():.4f}")
    print(f"  Std confidence:  {confidence.std():.4f}")
    print(f"  Min confidence:  {confidence.min():.4f}")
    print(f"  Max confidence:  {confidence.max():.4f}")
    
    # 2. Confidence by correctness
    correct_conf = confidence[labels == preds]
    incorrect_conf = confidence[labels != preds]
    print(f"\nConfidence by Correctness:")
    print(f"  Correct predictions:   {correct_conf.mean():.4f} ± {correct_conf.std():.4f}")
    if len(incorrect_conf) > 0:
        print(f"  Incorrect predictions: {incorrect_conf.mean():.4f} ± {incorrect_conf.std():.4f}")
    
    # 3. Confidence by true label
    print(f"\nConfidence by True Label:")
    for score in range(4):
        mask = labels == score
        if mask.sum() > 0:
            score_conf = confidence[mask]
            print(f"  Score {score}: {score_conf.mean():.4f} ± {score_conf.std():.4f} (n={mask.sum()})")
    
    # 4. Identify hardest samples (lowest 10% confidence)
    threshold = np.percentile(confidence, 10)
    hard_samples = df[df['confidence'] < threshold]
    print(f"\nHardest Samples (bottom 10%, confidence < {threshold:.4f}):")
    print(f"  Count: {len(hard_samples)}")
    if len(hard_samples) > 0:
        print(f"  Accuracy on hard samples: {hard_samples['correct'].mean():.4f}")
        print(f"  True label distribution:")
        for score in range(4):
            count = (hard_samples['true_label'] == score).sum()
            print(f"    Score {score}: {count} ({100*count/len(hard_samples):.1f}%)")
        
        # Show specific hard videos if IDs available
        if video_ids is not None and len(hard_samples) > 0:
            print(f"\n  Top 10 Hardest Videos:")
            for _, row in hard_samples.head(10).iterrows():
                print(f"    {row['video_id']}: true={int(row['true_label'])}, "
                      f"pred={int(row['predicted'])}, conf={row['confidence']:.3f}")
    
    # 5. Confusion pair analysis
    print(f"\nClass Confusion Analysis (when incorrect):")
    confusion_pairs = analyze_confusion_pairs(probs, preds, labels)
    
    # Save confusion pair analysis
    pd.DataFrame(confusion_pairs).to_csv(
        results_dir / f'{prefix}_confusion_pairs.csv', index=False
    )
    
    # 6. Boundary case analysis: samples near decision boundaries
    print(f"\nBoundary Cases (margin between top 2 classes < 0.2):")
    boundary_mask = margin < 0.2
    boundary_samples = df[boundary_mask]
    print(f"  Count: {len(boundary_samples)} ({100*len(boundary_samples)/len(df):.1f}% of total)")
    if len(boundary_samples) > 0:
        print(f"  Accuracy on boundary cases: {boundary_samples['correct'].mean():.4f}")
    
    # 7. Plot confidence distributions
    plot_confidence_analysis(df, results_dir, prefix)
    
    print(f"\nPredictions saved to: {results_dir / f'{prefix}_predictions_with_confidence.csv'}")
    print(f"Confusion pairs saved to: {results_dir / f'{prefix}_confusion_pairs.csv'}")
    
    return df


def analyze_confusion_pairs(probs, preds, labels):
    """
    Analyze which class pairs are most commonly confused.
    Returns statistics on confusion between each pair of classes.
    """
    confusion_data = []
    
    for true_class in range(4):
        for pred_class in range(4):
            if true_class == pred_class:
                continue
            
            # Find samples where true=true_class and pred=pred_class
            mask = (labels == true_class) & (preds == pred_class)
            count = mask.sum()
            
            if count > 0:
                # Average probability assigned to correct class when this confusion occurs
                avg_true_prob = probs[mask, true_class].mean()
                avg_pred_prob = probs[mask, pred_class].mean()
                
                confusion_data.append({
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'count': count,
                    'avg_prob_true_class': avg_true_prob,
                    'avg_prob_pred_class': avg_pred_prob,
                    'prob_difference': avg_pred_prob - avg_true_prob
                })
    
    # Sort by count (most common confusions first)
    confusion_data.sort(key=lambda x: -x['count'])
    
    print("  Top confusion pairs (true → predicted):")
    for item in confusion_data[:5]:
        print(f"    {item['true_class']} → {item['predicted_class']}: "
              f"{item['count']} cases, "
              f"avg prob diff: {item['prob_difference']:.3f}")
    
    return confusion_data


def plot_confidence_analysis(df, results_dir, prefix):
    """Generate confidence analysis plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Confidence histogram by correctness
    ax = axes[0, 0]
    correct_conf = df[df['correct']]['confidence']
    incorrect_conf = df[~df['correct']]['confidence']
    ax.hist(correct_conf, bins=30, alpha=0.7, label=f'Correct (n={len(correct_conf)})', color='green')
    ax.hist(incorrect_conf, bins=30, alpha=0.7, label=f'Incorrect (n={len(incorrect_conf)})', color='red')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution by Correctness')
    ax.legend()
    
    # 2. Confidence by true label (box plot)
    ax = axes[0, 1]
    data_by_label = [df[df['true_label'] == i]['confidence'].values for i in range(4)]
    bp = ax.boxplot(data_by_label, labels=['0', '1', '2', '3'], patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('True Score')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Distribution by True Label')
    
    # 3. Confusion heatmap with probabilities
    ax = axes[1, 0]
    # Create average probability matrix: for each (true, pred) pair,
    # show average probability assigned to predicted class
    prob_matrix = np.zeros((4, 4))
    count_matrix = np.zeros((4, 4))
    for i in range(len(df)):
        true_label = int(df.iloc[i]['true_label'])
        pred = int(df.iloc[i]['predicted'])
        prob_matrix[true_label, pred] += df.iloc[i][f'prob_score_{pred}']
        count_matrix[true_label, pred] += 1
    
    # Average probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_matrix = np.where(count_matrix > 0, prob_matrix / count_matrix, 0)
    
    sns.heatmap(prob_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=['0', '1', '2', '3'],
                yticklabels=['0', '1', '2', '3'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Avg Probability for Predicted Class')
    
    # 4. Accuracy vs confidence threshold (with coverage)
    ax = axes[1, 1]
    thresholds = np.linspace(0.3, 0.95, 20)
    accuracies = []
    coverages = []
    for thresh in thresholds:
        mask = df['confidence'] >= thresh
        if mask.sum() > 0:
            acc = df[mask]['correct'].mean()
            cov = mask.mean()
            accuracies.append(acc)
            coverages.append(cov)
        else:
            accuracies.append(np.nan)
            coverages.append(0)
    
    ax2 = ax.twinx()
    l1, = ax.plot(thresholds, accuracies, 'b-', linewidth=2, label='Accuracy')
    l2, = ax2.plot(thresholds, coverages, 'r--', linewidth=2, label='Coverage')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Accuracy', color='blue')
    ax2.set_ylabel('Coverage (% samples)', color='red')
    ax.set_title('Accuracy vs Coverage Trade-off')
    ax.legend(handles=[l1, l2], loc='center right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{prefix}_confidence_analysis.png', dpi=150)
    plt.close()
    print(f"Confidence plots saved to: {results_dir / f'{prefix}_confidence_analysis.png'}")


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
        valid_video_ids = [video_ids[i] for i in valid_indices]
        print(f"Valid videos with scores: {len(scores)}")
        
        dataset = PrecomputedVideoLatentDataset(latent_vectors, scores, video_ids=valid_video_ids)
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
            device=device,
            return_video_id=True  # Enable video ID tracking for confidence analysis
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
