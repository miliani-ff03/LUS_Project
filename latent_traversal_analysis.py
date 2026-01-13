"""
Enhanced Latent Traversal Analysis for VAE Interpretability.

This script generates comprehensive latent space traversal visualizations:
1. Multiple random samples as baselines
2. Mean latent vector as baseline (average behavior)
3. Both high-variance AND low-variance dimensions
4. Automatic saving for multiple beta values

Usage:
    python latent_traversal_analysis.py
    
    Or with specific beta:
    python latent_traversal_analysis.py --beta 2.0

Output:
    Saves PNG files to results/feature_traversal/
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from Medical_VAE_Clustering import ConvVAE, DEVICE


# ================= CONFIGURATION =================
LATENT_DIM = 32
CROP_PERCENT = 10  # Matches your saved models

# Available beta values (based on your saved models and latent features)
AVAILABLE_BETAS = [1.0, 2.0, 5.0]

# Number of random samples to analyze
NUM_RANDOM_SAMPLES = 5

# Number of top/bottom variance dimensions to show
NUM_FEATURES_PER_GROUP = 5

# Traversal range (in standard deviations)
TRAVERSAL_RANGE = (-3, 3)
TRAVERSAL_STEPS = 7

# Output directory
OUTPUT_DIR = Path("results/feature_traversal")
# =================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Latent traversal analysis")
    parser.add_argument(
        "--beta", "-b",
        type=float,
        default=None,
        help="Beta value to analyze. If not specified, analyzes all available betas."
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=NUM_RANDOM_SAMPLES,
        help=f"Number of random samples to analyze (default: {NUM_RANDOM_SAMPLES})"
    )
    return parser.parse_args()


def load_model_and_latents(beta: float):
    """Load VAE model and corresponding latent vectors for a given beta."""
    # beta=0.1 uses a different naming convention
    if beta == 0.1:
        model_path = f"Best_VAE_beta_0.1_crop{CROP_PERCENT}.pth"
    else:
        model_path = f"Best_VAE_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta}_cyclical.pth"
    latent_path = f"results/latent_features/latent_vectors_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta}.npy"
    
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None, None, None
    if not os.path.exists(latent_path):
        print(f"  Latents not found: {latent_path}")
        return None, None, None
    
    # Load model
    vae = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(model_path, map_location=DEVICE))
    vae.eval()
    
    # Load latent vectors
    X_latent = np.load(latent_path)
    
    # Compute variance for feature ranking
    variances = np.var(X_latent, axis=0)
    
    return vae, X_latent, variances


def decode_latent(vae, z: torch.Tensor) -> np.ndarray:
    """Decode a latent vector to an image."""
    with torch.no_grad():
        z_input = vae.decoder_input(z.unsqueeze(0))
        z_matrix = z_input.view(1, 256, 4, 4)
        reconstructed = vae.decoder(z_matrix)
    return reconstructed.cpu().squeeze().numpy()


def generate_traversal_grid(
    vae,
    base_z: np.ndarray,
    feature_indices: list,
    feature_labels: list,
    title: str,
    save_path: Path
):
    """
    Generate and save a traversal grid for given features.
    
    Args:
        vae: Trained VAE model
        base_z: Baseline latent vector (numpy array)
        feature_indices: List of latent dimension indices to traverse
        feature_labels: Labels for each row (e.g., "Dim 5 (high var)")
        title: Plot title
        save_path: Where to save the figure
    """
    sweep_values = np.linspace(TRAVERSAL_RANGE[0], TRAVERSAL_RANGE[1], TRAVERSAL_STEPS)
    base_tensor = torch.tensor(base_z, dtype=torch.float32).to(DEVICE)
    
    fig, axes = plt.subplots(
        len(feature_indices), 
        TRAVERSAL_STEPS, 
        figsize=(TRAVERSAL_STEPS * 2.5, len(feature_indices) * 2.5)
    )
    
    # Handle single row case
    if len(feature_indices) == 1:
        axes = axes.reshape(1, -1)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
    for row_idx, (feat_idx, feat_label) in enumerate(zip(feature_indices, feature_labels)):
        axes[row_idx, 0].set_ylabel(feat_label, rotation=0, size='medium', labelpad=60, va='center')
        
        for col_idx, val in enumerate(sweep_values):
            modified_z = base_tensor.clone()
            modified_z[feat_idx] = val
            
            reconstructed_image = decode_latent(vae, modified_z)
            
            ax = axes[row_idx, col_idx]
            
            # Handle multi-channel images
            if reconstructed_image.ndim == 3:
                # Convert CHW to HWC for display
                img_display = np.transpose(reconstructed_image, (1, 2, 0))
                # Clip to valid range
                img_display = np.clip(img_display, 0, 1)
                ax.imshow(img_display)
            else:
                ax.imshow(reconstructed_image, cmap='gray')
            
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f"z={val:.1f}", fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # Create output directory if needed
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def analyze_beta(beta: float, num_samples: int):
    """Run full traversal analysis for a single beta value."""
    print(f"\n{'='*60}")
    print(f"Analyzing β = {beta}")
    print(f"{'='*60}")
    
    vae, X_latent, variances = load_model_and_latents(beta)
    if vae is None:
        return
    
    n_samples = X_latent.shape[0]
    print(f"  Loaded {n_samples} latent vectors")
    
    # Get high and low variance dimension indices
    sorted_by_var = np.argsort(variances)
    high_var_indices = sorted_by_var[-NUM_FEATURES_PER_GROUP:][::-1]  # Highest first
    low_var_indices = sorted_by_var[:NUM_FEATURES_PER_GROUP]  # Lowest first
    
    print(f"  High variance dims: {high_var_indices.tolist()}")
    print(f"  Low variance dims:  {low_var_indices.tolist()}")
    
    # Prepare feature labels with variance info
    high_var_labels = [f"Dim {i}\n(var={variances[i]:.2f})" for i in high_var_indices]
    low_var_labels = [f"Dim {i}\n(var={variances[i]:.3f})" for i in low_var_indices]
    
    beta_str = f"beta{beta}"
    
    # === 1. Mean latent baseline ===
    print("\n  [1/3] Generating MEAN baseline traversal...")
    mean_latent = X_latent.mean(axis=0)
    
    generate_traversal_grid(
        vae, mean_latent,
        high_var_indices.tolist(),
        high_var_labels,
        f"Latent Traversal: HIGH Variance Dims (β={beta}, Mean Baseline)",
        OUTPUT_DIR / f"{beta_str}_mean_high_variance.png"
    )
    
    generate_traversal_grid(
        vae, mean_latent,
        low_var_indices.tolist(),
        low_var_labels,
        f"Latent Traversal: LOW Variance Dims (β={beta}, Mean Baseline)",
        OUTPUT_DIR / f"{beta_str}_mean_low_variance.png"
    )
    
    # === 2. Random samples ===
    print(f"\n  [2/3] Generating traversals for {num_samples} random samples...")
    np.random.seed(42)  # Reproducibility
    sample_indices = np.random.choice(n_samples, size=min(num_samples, n_samples), replace=False)
    
    for sample_idx in sample_indices:
        sample_z = X_latent[sample_idx]
        
        generate_traversal_grid(
            vae, sample_z,
            high_var_indices.tolist(),
            high_var_labels,
            f"Latent Traversal: HIGH Var Dims (β={beta}, Sample #{sample_idx})",
            OUTPUT_DIR / f"{beta_str}_sample{sample_idx}_high_variance.png"
        )
    
    # === 3. All dimensions for mean baseline ===
    print("\n  [3/3] Generating ALL dimensions traversal for mean baseline...")
    all_indices = list(range(LATENT_DIM))
    all_labels = [f"Dim {i}" for i in all_indices]
    
    generate_traversal_grid(
        vae, mean_latent,
        all_indices,
        all_labels,
        f"Full Latent Traversal: All {LATENT_DIM} Dims (β={beta}, Mean Baseline)",
        OUTPUT_DIR / f"{beta_str}_mean_all_dimensions.png"
    )
    
    print(f"\n  ✓ Completed analysis for β={beta}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LATENT TRAVERSAL ANALYSIS")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Traversal range: {TRAVERSAL_RANGE[0]}σ to {TRAVERSAL_RANGE[1]}σ")
    print(f"Random samples per beta: {args.samples}")
    
    if args.beta is not None:
        # Single beta analysis
        analyze_beta(args.beta, args.samples)
    else:
        # Analyze all available betas
        print(f"\nAnalyzing all available betas: {AVAILABLE_BETAS}")
        for beta in AVAILABLE_BETAS:
            analyze_beta(beta, args.samples)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
