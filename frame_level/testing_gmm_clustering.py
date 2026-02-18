import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import argparse
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import find_score

LATENT_FEATURES_PATH = "/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/latent_features/latent_vectors_ld23_crop10_beta2.0_gamma1.0.npy"
IMAGE_PATHS_PATH = "/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/latent_features/image_paths_ld23_crop10_beta2.0_gamma1.0.npy"
METADATA_PATH = "/cosma/home/durham/dc-fras4/code/data_preprocessing/data_tables/all_data.csv"

def load_latent_features_and_image_paths(latent_features_path=LATENT_FEATURES_PATH, image_paths_path=IMAGE_PATHS_PATH):
    latent_vectors = np.load(latent_features_path) 
    image_paths = np.load(image_paths_path, allow_pickle=True) 
    return latent_vectors, image_paths

def perform_gmm_clustering_and_visualize(latent_vectors, image_paths, df_meta, n_clusters=4, latent_dim=23):
    """Runs GMM clustering, computes t-SNE, and prepares data for visualization."""
    print(f"Performing GMM clustering with {n_clusters} clusters...")
    
    # Perform GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(latent_vectors)
    
    # Compute t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(latent_vectors)
    
    # Extract true scores from image paths
    print("Extracting true scores...")
    scores = []
    for img_path in image_paths:
        score = find_score(img_path, df_meta)
        scores.append(score if score is not None and not np.isnan(score) else -1)
    scores = np.array(scores)
    
    # Prepare results dictionary
    cluster_results = {
        'tsne_coords': tsne_coords,
        'tsne_cluster_labels': cluster_labels,
        'tsne_scores': scores
    }
    
    return cluster_results
    
def log_clustering_plots(cluster_results: dict, save_dir="results/gmm_plots"):
    """Generate and save clustering visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    tsne_coords = cluster_results['tsne_coords']
    cluster_labels = cluster_results['tsne_cluster_labels']
    true_scores = cluster_results['tsne_scores']
    
    # Plot 1 & 2: t-SNE colored by cluster and true score
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                c=cluster_labels, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    axes[0].set_title('t-SNE by GMM Cluster')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    scatter2 = axes[1].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                c=true_scores, cmap='RdYlGn_r', alpha=0.6, s=10)
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    axes[1].set_title('t-SNE by True Score')
    plt.colorbar(scatter2, ax=axes[1], label='Score')
    
    plt.tight_layout()
    tsne_path = os.path.join(save_dir, 'tsne_gmm_clusters.png')
    plt.savefig(tsne_path, dpi=150)
    print(f"Saved t-SNE plot to: {tsne_path}")
    plt.close()
    
    # Plot 3: Cluster vs Score heatmap
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_scores.astype(int), cluster_labels, 
                          labels=list(range(max(cluster_labels) + 1)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    n_clusters = len(np.unique(cluster_labels))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                yticklabels=[f'Score {i}' for i in range(4)])
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Score')
    ax.set_title('Cluster vs Score Distribution')
    
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, 'cluster_score_heatmap.png')
    plt.savefig(heatmap_path, dpi=150)
    print(f"Saved heatmap to: {heatmap_path}")
    plt.close()


def parse_args():
    """Parse arguments with Jupyter/IPython protection."""
    parser = argparse.ArgumentParser(description='Test GMM clustering on latent features')
    parser.add_argument('--n-clusters', type=int, default=4, help='Number of GMM clusters')
    parser.add_argument('--latent-features', type=str, default=LATENT_FEATURES_PATH,
                        help='Path to latent features .npy file')
    parser.add_argument('--image-paths', type=str, default=IMAGE_PATHS_PATH,
                        help='Path to image paths .npy file')
    
    # Jupyter/IPython protection
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        return parser.parse_args([])  # Return defaults in interactive mode
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load data
    print("Loading latent features and image paths...")
    latent_vectors, image_paths = load_latent_features_and_image_paths(
        args.latent_features, args.image_paths
    )
    print(f"Loaded {len(latent_vectors)} latent vectors with dimension {latent_vectors.shape[1]}")
    
    # Load metadata
    print("Loading metadata...")
    df_meta = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata with {len(df_meta)} entries")
    
    # Perform clustering and prepare visualization data
    cluster_results = perform_gmm_clustering_and_visualize(
        latent_vectors, image_paths, df_meta, n_clusters=args.n_clusters
    )
    
    # Display plots
    print("Displaying plots...")
    log_clustering_plots(cluster_results)
