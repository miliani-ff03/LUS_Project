"""
Latent Dimension Sweep Script.

Sweeps latent dimensions from 5 to 40, training the VAE 3 times per latent dim.
Produces two plots:
  1. Loss (total, reconstruction, KL) vs latent dimension
  2. Classifier accuracy + PCA components for 99% variance vs latent dimension

Usage:
    # Full sweep (default: latent_dim 5-40, 3 runs each, 60 epochs)
    python -m frame_level.latent_dim_sweep

    # Quick test
    python -m frame_level.latent_dim_sweep --ld_start 5 --ld_end 10 --n_runs 1 --epochs 5

    # Custom beta
    python -m frame_level.latent_dim_sweep --beta 1.0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import ConvVAE, LatentScoreClassifier
from shared.config import Config
from shared.utils import find_score

# Import from sibling modules
from frame_level.train_vae import (
    get_dataloader, train_vae, extract_latent_features,
    CHANNELS, CHECKPOINTS_DIR
)
from frame_level.latent_classifier import (
    PrecomputedLatentDataset, train_classifier
)

# ==========================================
# CONFIGURATION
# ==========================================

MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["WANDB_MODE"] = "offline"


# ==========================================
# PCA ANALYSIS
# ==========================================

def compute_pca_99_components(X_latent):
    """Compute number of PCA components needed to explain 99% of variance."""
    n_components = min(X_latent.shape)
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X_latent)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_99 = int(np.argmax(cumsum >= 0.99) + 1)
    return n_99


# ==========================================
# CLASSIFIER EVALUATION (simplified for sweep)
# ==========================================

def evaluate_classifier_accuracy(model, test_loader, device='cuda'):
    """Return test accuracy only (no plots/reports)."""
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

    return accuracy_score(all_labels, all_preds)


# ==========================================
# SINGLE RUN
# ==========================================

def run_single_experiment(latent_dim, train_loader, val_loader, metadata_df,
                          beta, epochs, learning_rate, crop_percent, device,
                          run_idx, classifier_epochs=50):
    """
    Train a VAE with given latent_dim, extract features, train classifier, do PCA.
    
    Returns:
        dict with keys: total_loss, recon_loss, kl_loss, test_accuracy, pca_99_components
    """
    print(f"\n{'='*60}")
    print(f"LATENT DIM = {latent_dim}, RUN {run_idx + 1}")
    print(f"{'='*60}")

    # Create VAE
    vae = ConvVAE(latent_dim=latent_dim, channels=CHANNELS)

    # Model save path (temporary for this sweep)
    suffix = f"crop{int(crop_percent * 100)}"
    full_suffix = f"ld{latent_dim}_{suffix}_beta{beta}_run{run_idx}"
    model_save_path = CHECKPOINTS_DIR / f"sweep_VAE_{full_suffix}.pth"

    # Train VAE
    tracker = train_vae(
        vae, train_loader, val_loader,
        epochs=epochs,
        end_beta=beta,
        learning_rate=learning_rate,
        save_path=model_save_path,
        patience=10,
        use_cyclical=True,
        device=device
    )

    # Get final epoch losses from tracker
    total_loss = tracker.history["loss"][-1]
    recon_loss = tracker.history["reconstruction_loss"][-1]
    kl_loss = tracker.history["kl_loss"][-1]

    print(f"Final losses - Total: {total_loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}")

    # Extract latent features
    X_latent, image_paths = extract_latent_features(vae, train_loader, device)
    print(f"Extracted {X_latent.shape[0]} latent vectors of dimension {X_latent.shape[1]}")

    # PCA analysis
    pca_99 = compute_pca_99_components(X_latent)
    print(f"PCA components for 99% variance: {pca_99}")

    # Prepare classifier data
    scores = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        score = find_score(str(path), metadata_df)
        if not np.isnan(score):
            scores.append(int(score))
            valid_indices.append(i)

    latent_valid = X_latent[valid_indices]
    print(f"Valid samples with scores: {len(scores)}")

    if len(scores) < 10:
        print("WARNING: Too few valid samples for classifier training. Skipping classifier.")
        test_accuracy = float('nan')
    else:
        dataset = PrecomputedLatentDataset(latent_valid, scores)

        # Split data 70/15/15
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            dataset, [train_size, val_size, test_size]
        )

        cls_train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        cls_val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        cls_test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # Train classifier
        classifier = LatentScoreClassifier(
            latent_dim=latent_dim,
            hidden_dims=[64, 32],
            output_dim=4,
            dropout=0.3
        )

        classifier, _ = train_classifier(
            classifier, cls_train_loader, cls_val_loader,
            epochs=classifier_epochs,
            learning_rate=1e-3,
            device=device
        )

        # Evaluate
        test_accuracy = evaluate_classifier_accuracy(classifier, cls_test_loader, device)
        print(f"Classifier test accuracy: {test_accuracy:.4f}")

    # Clean up temporary checkpoint to save disk space
    if model_save_path.exists():
        os.remove(model_save_path)

    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'test_accuracy': test_accuracy,
        'pca_99_components': pca_99,
    }


# ==========================================
# PLOTTING
# ==========================================

def plot_losses(results_df, save_path):
    """Plot total, reconstruction, and KL loss vs latent dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ld = results_df['latent_dim'].values

    # Total loss
    ax.plot(ld, results_df['total_loss_mean'], 'o-', color='#2196F3', label='Total Loss', linewidth=2)
    ax.fill_between(ld,
                    results_df['total_loss_mean'] - results_df['total_loss_std'],
                    results_df['total_loss_mean'] + results_df['total_loss_std'],
                    alpha=0.2, color='#2196F3')

    # Reconstruction loss
    ax.plot(ld, results_df['recon_loss_mean'], 's-', color='#4CAF50', label='Reconstruction Loss', linewidth=2)
    ax.fill_between(ld,
                    results_df['recon_loss_mean'] - results_df['recon_loss_std'],
                    results_df['recon_loss_mean'] + results_df['recon_loss_std'],
                    alpha=0.2, color='#4CAF50')

    # KL loss
    ax.plot(ld, results_df['kl_loss_mean'], '^-', color='#F44336', label='KL Loss', linewidth=2)
    ax.fill_between(ld,
                    results_df['kl_loss_mean'] - results_df['kl_loss_std'],
                    results_df['kl_loss_mean'] + results_df['kl_loss_std'],
                    alpha=0.2, color='#F44336')

    ax.set_xlabel('Number of Latent Features', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('VAE Loss Components vs Latent Dimension', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def plot_accuracy_and_pca(results_df, save_path):
    """Plot classifier accuracy (left y) and PCA 99% components (right y) vs latent dim."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ld = results_df['latent_dim'].values

    # Classifier accuracy (left axis)
    color_acc = '#2196F3'
    ax1.plot(ld, results_df['accuracy_mean'], 'o-', color=color_acc,
             label='Classifier Accuracy', linewidth=2, markersize=6)
    ax1.fill_between(ld,
                     results_df['accuracy_mean'] - results_df['accuracy_std'],
                     results_df['accuracy_mean'] + results_df['accuracy_std'],
                     alpha=0.2, color=color_acc)
    ax1.set_xlabel('Number of Latent Features', fontsize=13)
    ax1.set_ylabel('Classifier Accuracy', fontsize=13, color=color_acc)
    ax1.tick_params(axis='y', labelcolor=color_acc)

    # PCA components (right axis)
    ax2 = ax1.twinx()
    color_pca = '#F44336'
    ax2.plot(ld, results_df['pca_99_mean'], 's-', color=color_pca,
             label='PCA Components (99% var)', linewidth=2, markersize=6)
    ax2.fill_between(ld,
                     results_df['pca_99_mean'] - results_df['pca_99_std'],
                     results_df['pca_99_mean'] + results_df['pca_99_std'],
                     alpha=0.2, color=color_pca)
    ax2.set_ylabel('PCA Components for 99% Variance', fontsize=13, color=color_pca)
    ax2.tick_params(axis='y', labelcolor=color_pca)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='center right')

    ax1.set_title('Classifier Accuracy & PCA Effective Dimensionality vs Latent Features',
                  fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy/PCA plot saved to: {save_path}")


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Latent Dimension Sweep")
    parser.add_argument("--ld_start", type=int, default=5, help="Start latent dim (default: 5)")
    parser.add_argument("--ld_end", type=int, default=40, help="End latent dim inclusive (default: 40)")
    parser.add_argument("--ld_step", type=int, default=1, help="Step size for latent dim (default: 1)")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs per latent dim (default: 3)")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta for KL divergence (default: 2.0)")
    parser.add_argument("--epochs", type=int, default=60, help="VAE training epochs (default: 60)")
    parser.add_argument("--classifier_epochs", type=int, default=50, help="Classifier training epochs (default: 50)")
    parser.add_argument("--crop_percent", type=float, default=0.1, help="Crop percent (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to image data")
    parser.add_argument("--output_prefix", type=str, default="latent_dim_sweep",
                        help="Prefix for output files (default: latent_dim_sweep)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Build latent dim list
    latent_dims = list(range(args.ld_start, args.ld_end + 1, args.ld_step))
    total_experiments = len(latent_dims) * args.n_runs
    print(f"\nSweep configuration:")
    print(f"  Latent dims: {latent_dims[0]} to {latent_dims[-1]} (step {args.ld_step})")
    print(f"  Runs per dim: {args.n_runs}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Beta: {args.beta}, Epochs: {args.epochs}")

    # Load data once (shared across all experiments)
    data_path = args.data_path or str(Config.VAE_DATA_PATH)
    train_loader, val_loader = get_dataloader(
        data_path=data_path,
        batch_size=args.batch_size,
        crop_percent=args.crop_percent
    )

    # Load metadata once
    metadata_df = pd.read_csv(Config.METADATA_PATH)
    metadata_df['video_id'] = metadata_df['File Path'].apply(lambda p: Path(str(p)).stem)

    # Storage for all raw results
    all_results = []
    csv_path = RESULTS_DIR / f"{args.output_prefix}_results.csv"

    start_time = time.time()
    experiment_count = 0

    for ld in latent_dims:
        for run_idx in range(args.n_runs):
            experiment_count += 1
            elapsed = time.time() - start_time
            if experiment_count > 1:
                avg_time = elapsed / (experiment_count - 1)
                remaining = avg_time * (total_experiments - experiment_count + 1)
                print(f"\n[{experiment_count}/{total_experiments}] "
                      f"Elapsed: {elapsed/60:.1f}min, Est remaining: {remaining/60:.1f}min")

            # Set different seed for each run but reproducible
            seed = 42 + run_idx
            torch.manual_seed(seed)
            np.random.seed(seed)

            result = run_single_experiment(
                latent_dim=ld,
                train_loader=train_loader,
                val_loader=val_loader,
                metadata_df=metadata_df,
                beta=args.beta,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                crop_percent=args.crop_percent,
                device=device,
                run_idx=run_idx,
                classifier_epochs=args.classifier_epochs
            )

            result['latent_dim'] = ld
            result['run_idx'] = run_idx
            result['beta'] = args.beta
            all_results.append(result)

            # Save intermediate results after every experiment
            df_raw = pd.DataFrame(all_results)
            df_raw.to_csv(csv_path, index=False)
            print(f"Intermediate results saved to: {csv_path}")

    # ==========================================
    # AGGREGATE RESULTS
    # ==========================================

    df_raw = pd.DataFrame(all_results)

    # Compute mean and std per latent dim
    agg = df_raw.groupby('latent_dim').agg(
        total_loss_mean=('total_loss', 'mean'),
        total_loss_std=('total_loss', 'std'),
        recon_loss_mean=('recon_loss', 'mean'),
        recon_loss_std=('recon_loss', 'std'),
        kl_loss_mean=('kl_loss', 'mean'),
        kl_loss_std=('kl_loss', 'std'),
        accuracy_mean=('test_accuracy', 'mean'),
        accuracy_std=('test_accuracy', 'std'),
        pca_99_mean=('pca_99_components', 'mean'),
        pca_99_std=('pca_99_components', 'std'),
    ).reset_index()

    # Fill NaN std (happens when n_runs=1)
    agg = agg.fillna(0)

    agg_csv_path = RESULTS_DIR / f"{args.output_prefix}_aggregated.csv"
    agg.to_csv(agg_csv_path, index=False)
    print(f"\nAggregated results saved to: {agg_csv_path}")

    # ==========================================
    # GENERATE PLOTS
    # ==========================================

    plot_losses(agg, RESULTS_DIR / f"{args.output_prefix}_losses.png")
    plot_accuracy_and_pca(agg, RESULTS_DIR / f"{args.output_prefix}_accuracy_pca.png")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results: {csv_path}")
    print(f"Aggregated: {agg_csv_path}")
    print(f"Plots: {RESULTS_DIR}/{args.output_prefix}_losses.png")
    print(f"       {RESULTS_DIR}/{args.output_prefix}_accuracy_pca.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
