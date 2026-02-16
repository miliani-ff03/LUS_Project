import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




latent_features_path = "/cosma5/data/durham/dc-fras4/ultrasound/vae_results/latent_features"

# open the latent features and image paths
def load_model_and_latents(beta: float, LATENT_DIM: int, CROP_PERCENT: int, DEVICE):
    latent_vectors_path = os.path.join(
        latent_features_path,
        f"latent_vectors_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta}.npy"
    )
    image_paths_path = os.path.join(
        latent_features_path,
        f"image_paths_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta}.npy"
    )
    
    # Load latent vectors and image paths
    X_latent = np.load(latent_vectors_path)
    image_paths = np.load(image_paths_path)
    
    return X_latent, image_paths

def compute_variances(X_latent: np.ndarray) -> np.ndarray:
    # find variance explained by each feature

    return np.var(X_latent, axis=0)  # Shape: (latent_dim,)

def perform_pca_on_latents(X_latent: np.ndarray, n_components=None) -> tuple:
    """Perform PCA and return transformed data, variance explained, and PCA object."""
    if n_components is None:
        n_components = min(X_latent.shape)
    pca = PCA(n_components=n_components, svd_solver='full')
    X_pca = pca.fit_transform(X_latent)
    variance_explained = pca.explained_variance_ratio_
    return X_pca, variance_explained, pca



# plot variance vs feature index in order to visualise which features have the highest variance
def plot_variance_vs_feature(variances: np.ndarray, top_k: int = 10):
    # Sort variances in descending order
    sorted_indices = np.argsort(variances)[::-1]
    sorted_variances = variances[sorted_indices]
    
    # Create positions for sorted bars
    positions = np.arange(len(sorted_variances))
    
    plt.figure(figsize=(12, 6))
    # Plot all features sorted by variance
    plt.bar(positions, sorted_variances, color='steelblue')
    
    plt.xlabel("Rank (sorted by variance)")
    plt.ylabel("Variance")
    plt.title("Latent Features Sorted by Variance (Highest to Lowest)")
    
    # Optional: Show original feature indices on top
    ax = plt.gca()
    ax.set_xticks(positions)  # Show every other label to avoid crowding
    ax.set_xticklabels([f"F{sorted_indices[i]}" for i in range(0, len(sorted_indices), 1)])
    
    plt.legend()
    plt.tight_layout()
    plt.show()    

def plot_cumulative_variance(variances: np.ndarray):
    # Sort variances in descending order (lowest to highest)
    sorted_variances = np.sort(variances)[::-1]
    cumulative_variance = np.cumsum(sorted_variances)

    
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance, marker='o', linestyle='-')    
    plt.xlabel("Number of Features (sorted by variance)")
    plt.ylabel("Cumulative Variance")
    plt.title("Cumulative Variance Explained by Latent Features (Highest to Lowest)")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def calculate_effective_dimensionality(variance_explained: np.ndarray) -> dict:
    """Calculate various metrics for effective dimensionality."""
    cumsum_variance = np.cumsum(variance_explained)
    
    metrics = {
        'n_components_90': np.argmax(cumsum_variance >= 0.90) + 1,
        'n_components_95': np.argmax(cumsum_variance >= 0.95) + 1,
        'n_components_99': np.argmax(cumsum_variance >= 0.99) + 1,
        # Participation ratio (effective dimensionality)
        'participation_ratio': (np.sum(variance_explained) ** 2) / np.sum(variance_explained ** 2),
        # Shannon entropy based dimensionality
        'entropy_dim': np.exp(-np.sum(variance_explained * np.log(variance_explained + 1e-10))),
    }
    
    return metrics

def print_dimensionality_analysis(variance_explained: np.ndarray, beta: float):
    """Print comprehensive dimensionality analysis."""
    metrics = calculate_effective_dimensionality(variance_explained)
    
    print(f"\n{'='*60}")
    print(f"Dimensionality Analysis for β={beta}")
    print(f"{'='*60}")
    print(f"Total latent dimensions: {len(variance_explained)}")
    print(f"\nComponents needed for variance thresholds:")
    print(f"  90% variance: {metrics['n_components_90']} components")
    print(f"  95% variance: {metrics['n_components_95']} components")
    print(f"  99% variance: {metrics['n_components_99']} components")
    print(f"\nEffective dimensionality metrics:")
    print(f"  Participation ratio: {metrics['participation_ratio']:.2f}")
    print(f"  Entropy-based dim: {metrics['entropy_dim']:.2f}")
    print(f"\nCompression potential:")
    compression_90 = (1 - metrics['n_components_90'] / len(variance_explained)) * 100
    compression_95 = (1 - metrics['n_components_95'] / len(variance_explained)) * 100
    print(f"  {compression_90:.1f}% reduction possible (90% variance retained)")
    print(f"  {compression_95:.1f}% reduction possible (95% variance retained)")
    print(f"{'='*60}\n")

def plot_variance_explained(variance_explained: np.ndarray, beta: float = None):
    """Plot both individual and cumulative variance explained with threshold lines."""
    cumsum_variance = np.cumsum(variance_explained) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Individual variance explained
    ax1.bar(range(1, len(variance_explained) + 1), variance_explained * 100, 
            color='steelblue', alpha=0.7)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Individual Variance Explained by Each PC")
    ax1.grid(alpha=0.3)
    
    # Cumulative variance explained
    ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 
             marker='o', linewidth=2, markersize=4)
    
    # Add threshold lines
    thresholds = [90, 95, 99]
    colors = ['red', 'orange', 'green']
    for threshold, color in zip(thresholds, colors):
        n_components = np.argmax(cumsum_variance >= threshold) + 1
        ax2.axhline(y=threshold, color=color, linestyle='--', alpha=0.6, 
                   label=f'{threshold}% ({n_components} PCs)')
        ax2.axvline(x=n_components, color=color, linestyle='--', alpha=0.3)
    
    ax2.set_xlabel("Number of Principal Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    title = "Cumulative Variance Explained"
    if beta is not None:
        title += f" (β={beta})"
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.show()

def compare_betas(betas: list, LATENT_DIM: int, CROP_PERCENT: int, DEVICE: str):
    """Compare dimensionality across different beta values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    results = {}
    
    for beta in betas:
        X_latent, _ = load_model_and_latents(beta, LATENT_DIM, CROP_PERCENT, DEVICE)
        _, variance_explained, _ = perform_pca_on_latents(X_latent)
        cumsum_variance = np.cumsum(variance_explained) * 100
        
        ax.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 
                marker='o', linewidth=2, label=f'β={beta}', markersize=4)
        
        metrics = calculate_effective_dimensionality(variance_explained)
        results[beta] = metrics
    
    # Add threshold lines
    for threshold in [90, 95]:
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.3)
        ax.text(len(cumsum_variance) * 0.5, threshold + 1, f'{threshold}%', 
                fontsize=9, color='gray')
    
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("Comparison of Dimensionality Across Different β Values")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\n" + "="*80)
    print("Comparison of Effective Dimensionality Across β Values")
    print("="*80)
    print(f"{'β':<8} {'90% var':<12} {'95% var':<12} {'Part. Ratio':<15} {'Entropy Dim':<15}")
    print("-"*80)
    for beta, metrics in results.items():
        print(f"{beta:<8.1f} {metrics['n_components_90']:<12} "
              f"{metrics['n_components_95']:<12} "
              f"{metrics['participation_ratio']:<15.2f} "
              f"{metrics['entropy_dim']:<15.2f}")
    print("="*80 + "\n")
    
    return results


# 
def main():
    LATENT_DIM = 23
    CROP_PERCENT = 10
    DEVICE = 'cpu'
    beta = 2.0
    
    X_latent, image_paths = load_model_and_latents(beta, LATENT_DIM, CROP_PERCENT, DEVICE)
    variances = compute_variances(X_latent)
    plot_cumulative_variance(variances)


if __name__ == "__main__":
    main()

