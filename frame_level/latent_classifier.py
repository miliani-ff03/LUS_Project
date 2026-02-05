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
import pandas as pd

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


def evaluate_classifier(model, test_loader, device='cuda', save_predictions=True):
    """Evaluate classifier and generate reports with confidence analysis."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store softmax probabilities
    
    with torch.no_grad():
        for latents, labels in test_loader:
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
    ax.set_title('Frame-Level Classifier Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'frame_classifier_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to: {RESULTS_DIR / 'frame_classifier_confusion_matrix.png'}")
    
    # === CONFIDENCE ANALYSIS ===
    analyze_confidence(all_probs, all_preds, all_labels, RESULTS_DIR, prefix='frame')
    
    return accuracy_score(all_labels, all_preds)


def analyze_confidence(probs, preds, labels, results_dir, prefix='frame'):
    """
    Analyze prediction confidence to identify hard-to-classify samples.
    
    Args:
        probs: (N, 4) array of softmax probabilities
        preds: (N,) array of predicted labels
        labels: (N,) array of true labels
        results_dir: Path to save results
        prefix: 'frame' or 'video' for filenames
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
    
    # Add entropy (uncertainty measure)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    df['entropy'] = entropy
    
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
    
    # 5. Confusion pair analysis
    print(f"\nClass Confusion Analysis (when incorrect):")
    confusion_pairs = analyze_confusion_pairs(probs, preds, labels)
    
    # Save confusion pair analysis
    pd.DataFrame(confusion_pairs).to_csv(
        results_dir / f'{prefix}_confusion_pairs.csv', index=False
    )
    
    # 6. Plot confidence distributions
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
    
    # 3. Entropy vs Confidence scatter
    ax = axes[1, 0]
    scatter = ax.scatter(df['confidence'], df['entropy'], 
                         c=df['correct'].map({True: 'green', False: 'red'}),
                         alpha=0.5, s=10)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Entropy (Uncertainty)')
    ax.set_title('Confidence vs Entropy')
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Correct'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Incorrect')]
    ax.legend(handles=legend_elements)
    
    # 4. Accuracy vs confidence threshold
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
    l1, = ax.plot(thresholds, accuracies, 'b-', label='Accuracy')
    l2, = ax2.plot(thresholds, coverages, 'r--', label='Coverage')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Accuracy', color='blue')
    ax2.set_ylabel('Coverage (% samples)', color='red')
    ax.set_title('Accuracy vs Coverage Trade-off')
    ax.legend(handles=[l1, l2], loc='center right')
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{prefix}_confidence_analysis.png', dpi=150)
    plt.close()
    print(f"Confidence plots saved to: {results_dir / f'{prefix}_confidence_analysis.png'}")


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
