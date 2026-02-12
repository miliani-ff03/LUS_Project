"""
Frame-Level Supervised VAE Training Script.

Trains a Variational Autoencoder with an auxiliary classifier head that
predicts severity scores from the latent space. This encourages the VAE
to learn latent representations that are discriminative for classification.

Loss function:
    Total = BCE(reconstruction) + β * KL(latent) + γ * CrossEntropy(score)

Usage:
    python -m frame_level.train_vae_with_classifier --beta 2.0 --gamma 1.0 --latent_dim 32
    
    # With class weights for imbalanced data
    python frame_level/train_vae_with_classifier.py --beta 2.0 --gamma 0.5 --use_class_weights
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import (
    ConvVAE,
    KLAnnealer,
    CyclicalAnnealer,
    EarlyStopping,
    LossTracker,
)
from shared.config import Config
from shared.utils import find_score

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
RESULTS_DIR = MODULE_DIR / "results" / "supervised"
CHECKPOINTS_DIR = MODULE_DIR / "checkpoints"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Image settings
IMAGE_SIZE = 64
CHANNELS = 1  # Grayscale

os.environ["WANDB_MODE"] = "offline"


# ==========================================
# CLASSIFIER HEAD
# ==========================================

class LatentClassifierHead(nn.Module):
    """
    MLP classifier from latent space to severity scores (0-3).
    
    Architecture: latent_dim → 64 → 32 → num_classes
    """
    
    def __init__(self, latent_dim: int, num_classes: int = 4, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vectors (batch, latent_dim)
        Returns:
            Logits for each class (batch, num_classes)
        """
        return self.classifier(z)


# ==========================================
# DATASET WITH SCORES
# ==========================================

class FlatImageDatasetWithScores(Dataset):
    """
    Dataset for images with severity scores from metadata.
    
    Returns (image, path, score) tuples where score is -1 for unmatched samples.
    """
    
    def __init__(self, root_dir: str, metadata_path: str, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        self.transform = transform
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        if 'video_id' not in self.metadata.columns:
            self.metadata['video_id'] = self.metadata['File Path'].apply(
                lambda p: Path(str(p)).stem
            )
        
        # Pre-compute scores for all images
        print("Matching images to scores...")
        self.scores = {}
        matched = 0
        for path in self.image_paths:
            score = find_score(path, self.metadata)
            if not np.isnan(score):
                self.scores[path] = int(score)
                matched += 1
            else:
                self.scores[path] = -1  # Unknown score marker
        
        print(f"Matched {matched}/{len(self.image_paths)} images to scores")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Load as grayscale directly
        
        if self.transform:
            image = self.transform(image)
        
        score = self.scores.get(img_path, -1)
        return image, img_path, score


# ==========================================
# LOSS FUNCTION
# ==========================================

def supervised_vae_loss(
    recon_x, x, mu, logvar, score_logits, scores,
    beta=1.0, gamma=1.0, class_weights=None, device='cuda'
):
    # Reconstruction and KL (same as before)
    bce = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Filter for samples with valid scores
    valid_mask = scores >= 0
    n_valid = valid_mask.sum()
    
    if n_valid > 0:
        # Generate the soft distributions for valid samples
        targets = get_soft_labels(scores[valid_mask], device=device)
        
        # Manual cross-entropy computation for soft labels
        # CE = -sum(target_probs * log(predicted_probs))
        log_probs = torch.nn.functional.log_softmax(score_logits[valid_mask], dim=1)

        if class_weights is not None: 
            sample_weights = class_weights[scores[valid_mask]]
            ce_loss = -(targets * log_probs).sum(dim=1) * sample_weights 
            ce_loss = ce_loss.mean() * n_valid 

        else:
            ce_loss = -(targets * log_probs).sum(dim=1).mean()
            ce_loss = ce_loss * n_valid
    else:
        ce_loss = torch.tensor(0.0, device=device)
    
    total_loss = bce + beta * kld + gamma * ce_loss
    return total_loss, bce, kld, ce_loss

# ==========================================
# DATA LOADING
# ==========================================

def get_dataloader_with_scores(
    data_path: str, 
    metadata_path: str, 
    batch_size: int = 64, 
    crop_percent: float = 0.1, 
    val_split: float = 0.2
) -> tuple:
    """
    Returns train/val dataloaders that include severity scores.
    
    Returns:
        Tuple of (train_loader, val_loader, class_weights)
    """
    transform_ops = []

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
        transforms.ToTensor()
    ])
    
    transform = transforms.Compose(transform_ops)

    print(f"Loading frame data from {data_path}")
    print(f"Using metadata from {metadata_path}")

    full_dataset = FlatImageDatasetWithScores(
        root_dir=data_path, 
        metadata_path=metadata_path,
        transform=transform
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Data Split: {train_size} training, {val_size} validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Compute class weights for imbalanced data
    valid_scores = [s for s in full_dataset.scores.values() if s >= 0]
    if valid_scores:
        score_counts = np.bincount(valid_scores, minlength=4)
        print(f"Score distribution: {dict(enumerate(score_counts))}")
        
        # Inverse frequency weighting
        class_weights = torch.tensor(
            len(valid_scores) / (4 * (score_counts + 1e-6)), 
            dtype=torch.float32
        )
        print(f"Class weights: {class_weights.tolist()}")
    else:
        class_weights = None
        print("Warning: No valid scores found!")
    
    return train_loader, val_loader, class_weights


# ==========================================
# TRAINING
# ==========================================
def get_soft_labels(scores, num_classes=4, device='cuda'):
    """
    Converts hard scores (0-3) into soft probability distributions.
    0, 3 -> 90% primary, 10% neighbor.
    1, 2 -> 70% primary, 15% each neighbor.
    """
    batch_size = scores.size(0)
    soft_labels = torch.zeros(batch_size, num_classes, device=device)
    
    for i in range(batch_size):
        score = scores[i].item()
        if score == 0:
            soft_labels[i, 0] = 0.90
            soft_labels[i, 1] = 0.10
        elif score == 3:
            soft_labels[i, 3] = 0.90
            soft_labels[i, 2] = 0.10
        elif score == 1:
            soft_labels[i, 1] = 0.70
            soft_labels[i, 0] = 0.15
            soft_labels[i, 2] = 0.15
        elif score == 2:
            soft_labels[i, 2] = 0.70
            soft_labels[i, 1] = 0.15
            soft_labels[i, 3] = 0.15
            
    return soft_labels


class SupervisedLossTracker:
    """Extended loss tracker that includes classification loss."""
    
    def __init__(self):
        self.history = {
            "loss": [],
            "reconstruction_loss": [],
            "kl_loss": [],
            "classification_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
    
    def add(self, loss, recon, kl, cls_loss, val_loss, val_acc):
        self.history["loss"].append(loss)
        self.history["reconstruction_loss"].append(recon)
        self.history["kl_loss"].append(kl)
        self.history["classification_loss"].append(cls_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_acc)


def train_supervised_vae(
    model: ConvVAE, 
    classifier_head: LatentClassifierHead, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    epochs: int, 
    end_beta: float, 
    gamma: float, 
    learning_rate: float, 
    save_path: Path,
    patience: int = 10, 
    use_cyclical: bool = False, 
    class_weights: torch.Tensor = None,
    device: str = 'cuda'
) -> tuple:
    """
    Train VAE with classification loss.
    
    The classifier head predicts severity scores from the latent mean (mu).
    Backpropagation through the classifier loss encourages the encoder to
    produce discriminative latent representations.
    
    Returns:
        Tuple of (tracker, trained_classifier_head)
    """
    model = model.to(device)
    classifier_head = classifier_head.to(device)
    
    # Joint optimization of VAE and classifier
    all_params = list(model.parameters()) + list(classifier_head.parameters())
    optimizer = optim.Adam(all_params, lr=learning_rate)
    
    tracker = SupervisedLossTracker()
    
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Setup annealing
    if use_cyclical:
        annealer = CyclicalAnnealer(
            total_epochs=epochs, cycles=4, 
            start_beta=0.0, end_beta=end_beta
        )
    else:
        annealer = KLAnnealer(
            total_epochs=epochs, 
            start_beta=0.0, end_beta=end_beta
        )

    # Early stopping on validation loss
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=str(save_path))
    
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        classifier_head.train()
        
        total_loss = 0
        total_bce = 0
        total_kld = 0
        total_ce = 0
        train_correct = 0
        train_valid = 0

        current_beta = annealer.get_beta(epoch)

        for batch_idx, (data, _, scores) in enumerate(train_loader):
            data = data.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through VAE
            recon_batch, mu, logvar = model(data)
            
            # Forward pass through classifier (from latent mean)
            score_logits = classifier_head(mu)
            
            # Combined loss
            loss, bce, kld, ce = supervised_vae_loss(
                recon_batch, data, mu, logvar, score_logits, scores,
                beta=current_beta, gamma=gamma, device=device
            )
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
            total_ce += ce.item()
            
            # Training accuracy
            valid_mask = scores >= 0
            if valid_mask.sum() > 0:
                preds = score_logits[valid_mask].argmax(dim=1)
                train_correct += (preds == scores[valid_mask]).sum().item()
                train_valid += valid_mask.sum().item()

        # Validation
        model.eval()
        classifier_head.eval()
        val_loss = 0
        val_correct = 0
        val_valid = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for val_data, _, val_scores in val_loader:
                val_data = val_data.to(device)
                val_scores = val_scores.to(device)
                
                recon_val, mu_val, logvar_val = model(val_data)
                score_logits_val = classifier_head(mu_val)
                
                v_loss, _, _, _ = supervised_vae_loss(
                    recon_val, val_data, mu_val, logvar_val, 
                    score_logits_val, val_scores,
                    beta=current_beta, gamma=gamma, device=device
                )
                val_loss += v_loss.item()
                
                # Validation accuracy
                valid_mask = val_scores >= 0
                if valid_mask.sum() > 0:
                    preds = score_logits_val[valid_mask].argmax(dim=1)
                    val_correct += (preds == val_scores[valid_mask]).sum().item()
                    val_valid += valid_mask.sum().item()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(val_scores[valid_mask].cpu().numpy())

        # Compute averages
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        
        avg_train_loss = total_loss / n_train
        avg_bce = total_bce / n_train
        avg_kld = total_kld / n_train
        avg_ce = total_ce / n_train
        avg_val_loss = val_loss / n_val
        
        train_acc = train_correct / train_valid if train_valid > 0 else 0.0
        val_acc = val_correct / val_valid if val_valid > 0 else 0.0
        
        tracker.add(avg_train_loss, avg_bce, avg_kld, avg_ce, avg_val_loss, val_acc)
        
        # Update best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_train_loss:.2f} | "
              f"Val: {avg_val_loss:.2f} | "
              f"Train Acc: {train_acc:.2%} | "
              f"Val Acc: {val_acc:.2%} | "
              f"β: {current_beta:.3f} | "
              f"CE: {avg_ce:.2f}")
        
        # Early stopping logic
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

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    
    # Load best model
    print(f"Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path))

    return tracker, classifier_head


# ==========================================
# VISUALIZATION
# ==========================================

def log_supervised_loss_graphs(tracker: SupervisedLossTracker, latent_dim: int, 
                                beta: float, gamma: float, save_suffix: str = ""):
    """Plot and save loss curves including classification loss."""
    epochs = range(1, len(tracker.history["loss"]) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Total loss
    axes[0, 0].plot(epochs, tracker.history["loss"], 'b-')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"Total Loss (ld={latent_dim}, β={beta}, γ={gamma})")
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, tracker.history["reconstruction_loss"], 'g-')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Reconstruction Loss (BCE)")
    axes[0, 1].grid(True)
    
    # KL divergence
    axes[0, 2].plot(epochs, tracker.history["kl_loss"], 'r-')
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("KL Divergence")
    axes[0, 2].grid(True)
    
    # Classification loss
    axes[1, 0].plot(epochs, tracker.history["classification_loss"], 'm-')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Classification Loss (CE)")
    axes[1, 0].grid(True)
    
    # Validation loss
    axes[1, 1].plot(epochs, tracker.history["val_loss"], 'c-')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Validation Loss")
    axes[1, 1].grid(True)
    
    # Validation accuracy
    axes[1, 2].plot(epochs, tracker.history["val_accuracy"], 'orange')
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Accuracy")
    axes[1, 2].set_title("Validation Accuracy")
    axes[1, 2].grid(True)
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    save_path = RESULTS_DIR / f"loss_curves_{save_suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to: {save_path}")


def plot_confusion_matrix(model: ConvVAE, classifier_head: LatentClassifierHead,
                          dataloader: DataLoader, save_suffix: str, device: str = 'cuda'):
    """Generate and save confusion matrix for classifier predictions."""
    model.eval()
    classifier_head.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, _, scores in dataloader:
            data = data.to(device)
            scores = scores.to(device)
            
            mu = model.encode(data)
            logits = classifier_head(mu)
            
            valid_mask = scores >= 0
            if valid_mask.sum() > 0:
                preds = logits[valid_mask].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(scores[valid_mask].cpu().numpy())
    
    if not all_preds:
        print("No valid predictions for confusion matrix")
        return
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Score 0', 'Score 1', 'Score 2', 'Score 3'],
                yticklabels=['Score 0', 'Score 1', 'Score 2', 'Score 3'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {save_suffix}')
    
    save_path = RESULTS_DIR / f"confusion_matrix_{save_suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['Score 0', 'Score 1', 'Score 2', 'Score 3'],
                               zero_division=0))


def plot_tsne_by_score(model: ConvVAE, dataloader: DataLoader, 
                       save_suffix: str, device: str = 'cuda'):
    """Plot t-SNE visualization colored by true severity score."""
    model.eval()
    
    latent_vectors = []
    scores_list = []
    
    with torch.no_grad():
        for data, _, scores in dataloader:
            data = data.to(device)
            mu = model.encode(data)
            latent_vectors.append(mu.cpu().numpy())
            scores_list.extend(scores.numpy())
    
    X_latent = np.concatenate(latent_vectors)
    scores_array = np.array(scores_list)
    
    # Filter to valid scores only
    valid_mask = scores_array >= 0
    X_valid = X_latent[valid_mask]
    scores_valid = scores_array[valid_mask]
    
    print(f"Running t-SNE on {len(X_valid)} samples with valid scores...")
    
    # Subsample if too large
    max_samples = 2000
    if len(X_valid) > max_samples:
        idx = np.random.choice(len(X_valid), max_samples, replace=False)
        X_valid = X_valid[idx]
        scores_valid = scores_valid[idx]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_valid)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                         c=scores_valid, cmap='RdYlGn_r', alpha=0.6, s=10)
    plt.colorbar(scatter, ax=ax, label='Severity Score')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_title(f't-SNE Colored by True Score - {save_suffix}')
    
    save_path = RESULTS_DIR / f"tsne_by_score_{save_suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to: {save_path}")


def log_reconstruction(model: ConvVAE, dataloader: DataLoader, 
                       save_suffix: str, device: str = 'cuda'):
    """Log reconstruction comparison."""
    model.eval()
    with torch.no_grad():
        sample_data, _, _ = next(iter(dataloader))
        sample_data = sample_data.to(device)[:8]
        recon, _, _ = model(sample_data)
        
        comparison = torch.cat([sample_data, recon])
        grid = torchvision.utils.make_grid(comparison.cpu(), nrow=8)
        
        fig = plt.figure(figsize=(15, 5))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title(f"Top: Original | Bottom: Reconstructed ({save_suffix})")
        plt.axis('off')
        
        save_path = RESULTS_DIR / f"reconstruction_{save_suffix}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Reconstruction saved to: {save_path}")


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Frame-Level Supervised VAE Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=32, 
                        help="Latent dimension size")
    parser.add_argument("--beta", type=float, default=2.0, 
                        help="Weight for KL divergence (β-VAE)")
    parser.add_argument("--gamma", type=float, default=1.0, 
                        help="Weight for classification loss")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=60, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--annealing", type=str, default="cyclical", 
                        choices=["cyclical", "linear"],
                        help="KL annealing strategy")
    
    # Data parameters
    parser.add_argument("--crop_percent", type=float, default=0.1, 
                        help="Percentage to crop from top of images")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to image data (default: from Config)")
    parser.add_argument("--metadata_path", type=str, default=None, 
                        help="Path to metadata CSV (default: from Config)")
    
    # Options
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use inverse-frequency class weights for imbalanced data")
    parser.add_argument("--skip_viz", action="store_true",
                        help="Skip visualization generation")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Running on device: {device}")
    print(f"Config: crop={args.crop_percent}, β={args.beta}, γ={args.gamma}, "
          f"ld={args.latent_dim}, epochs={args.epochs}")
    
    # Load environment
    load_dotenv()
    
    # Paths
    data_path = args.data_path or str(Config.VAE_DATA_PATH)
    metadata_path = args.metadata_path or str(Config.METADATA_PATH)
    
    # Get dataloaders with scores
    train_loader, val_loader, class_weights = get_dataloader_with_scores(
        data_path=data_path,
        metadata_path=metadata_path,
        batch_size=args.batch_size,
        crop_percent=args.crop_percent
    )
    
    # Initialize wandb (optional)
    if HAS_WANDB:
        wandb.init(
            project="lus-medical-vae",
            group="supervised_frame_level",
            config={
                "crop_percent": args.crop_percent,
                "beta": args.beta,
                "gamma": args.gamma,
                "epochs": args.epochs,
                "latent_dim": args.latent_dim,
                "use_class_weights": args.use_class_weights,
                "level": "frame",
                "supervised": True
            },
            reinit=True
        )
    
    # Create models
    vae = ConvVAE(latent_dim=args.latent_dim, channels=CHANNELS)
    classifier_head = LatentClassifierHead(latent_dim=args.latent_dim, num_classes=4)
    
    # Model save path
    crop_suffix = f"crop{int(args.crop_percent*100)}"
    full_suffix = f"ld{args.latent_dim}_{crop_suffix}_beta{args.beta}_gamma{args.gamma}"
    model_save_path = CHECKPOINTS_DIR / f"SupervisedVAE_{full_suffix}_{args.annealing}.pth"
    
    # Train
    tracker, classifier_head = train_supervised_vae(
        vae, classifier_head, train_loader, val_loader,
        epochs=args.epochs,
        end_beta=args.beta,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        save_path=model_save_path,
        patience=args.patience,
        use_cyclical=(args.annealing == "cyclical"),
        class_weights=class_weights if args.use_class_weights else None,
        device=device
    )
    
    # Save classifier head
    classifier_path = CHECKPOINTS_DIR / f"ClassifierHead_{full_suffix}.pth"
    torch.save(classifier_head.state_dict(), classifier_path)
    print(f"Classifier head saved to: {classifier_path}")
    
    # Generate visualizations
    if not args.skip_viz:
        print("\nGenerating visualizations...")
        log_supervised_loss_graphs(tracker, args.latent_dim, args.beta, args.gamma, full_suffix)
        log_reconstruction(vae, val_loader, full_suffix, device)
        plot_confusion_matrix(vae, classifier_head, val_loader, full_suffix, device)
        plot_tsne_by_score(vae, val_loader, full_suffix, device)
    
    # Save latent features from both train and val sets
    print("\nExtracting and saving latent features...")
    latent_dir = RESULTS_DIR / "latent_features"
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    vae.eval()
    latent_vectors = []
    image_paths = []
    scores_list = []
    split_labels = []  # Track which split each sample belongs to
    
    with torch.no_grad():
        # Extract from training set
        print(f"Extracting features from training set...")
        for data, paths, scores in train_loader:
            data = data.to(device)
            mu = vae.encode(data)
            latent_vectors.append(mu.cpu().numpy())
            image_paths.extend(paths)
            scores_list.extend(scores.numpy())
            split_labels.extend(['train'] * len(paths))
        
        # Extract from validation set
        print(f"Extracting features from validation set...")
        for data, paths, scores in val_loader:
            data = data.to(device)
            mu = vae.encode(data)
            latent_vectors.append(mu.cpu().numpy())
            image_paths.extend(paths)
            scores_list.extend(scores.numpy())
            split_labels.extend(['val'] * len(paths))
    
    X_latent = np.concatenate(latent_vectors)
    np.save(latent_dir / f"latent_vectors_{full_suffix}.npy", X_latent)
    np.save(latent_dir / f"image_paths_{full_suffix}.npy", np.array(image_paths))
    np.save(latent_dir / f"scores_{full_suffix}.npy", np.array(scores_list))
    np.save(latent_dir / f"split_labels_{full_suffix}.npy", np.array(split_labels))
    
    train_count = np.sum(np.array(split_labels) == 'train')
    val_count = np.sum(np.array(split_labels) == 'val')
    print(f"Saved {X_latent.shape[0]} latent vectors of dimension {X_latent.shape[1]}")
    print(f"  Training: {train_count} samples")
    print(f"  Validation: {val_count} samples")
    
    # Log to wandb
    if HAS_WANDB and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"supervised_latent_vectors_{full_suffix}",
            type="latent_features",
            description=f"Supervised frame-level latent vectors (ld={args.latent_dim}, β={args.beta}, γ={args.gamma})"
        )
        artifact.add_dir(str(latent_dir))
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print(f"\nDone! Results saved to: {RESULTS_DIR}")
    print(f"Model checkpoint: {model_save_path}")
    print(f"Classifier head: {classifier_path}")


if __name__ == "__main__":
    main()
