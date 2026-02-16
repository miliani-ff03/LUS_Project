import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import argparse
import sys

import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap



LATENT_FEATURES_PATH = "/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/latent_features/latent_vectors_ld23_crop10_beta2.0_gamma1.0.npy"
IMAGE_PATHS_PATH = "/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/latent_features/image_paths_ld23_crop10_beta2.0_gamma1.0.npy"

RESULTS_DIR = Path("/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/clustering_plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_latent_features_and_image_paths(latent_features_path=LATENT_FEATURES_PATH, image_paths_path=IMAGE_PATHS_PATH):
    latent_vectors = np.load(latent_features_path) 
    image_paths = np.load(image_paths_path, allow_pickle=True) 
    return latent_vectors, image_paths

latent_vectors, image_paths = load_latent_features_and_image_paths()

def perform_kmean_clustering_and_log(latent_vectors = latent_vectors, image_paths = image_paths, latent_dim=32, beta=2.0, crop_suffix="crop10"):
    """Runs feature extraction, KMeans, t-SNE, and saves plots."""
    print(f"Generating clustering plots for LD={latent_dim}...")
    
    
    for N_CLUSTERS in [2, 3, 4]:
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(latent_vectors)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(latent_vectors)
        
        cluster_df = pd.DataFrame({
            'image_path': image_paths,
            'cluster_label': clusters,
            'tsne_x': X_embedded[:, 0],
            'tsne_y': X_embedded[:, 1],
            '_data_level': 'frame'  # Indicates frame-level data
        })

        table_path = RESULTS_DIR / f"cluster_table_ld{latent_dim}_beta{beta}_{crop_suffix}_k{N_CLUSTERS}.json"
        cluster_df.to_json(table_path, orient='records', indent=2)
        print(f"Cluster table saved to: {table_path}")
        
        # Also save as CSV for easy inspection
        csv_path = RESULTS_DIR / f"cluster_table_ld{latent_dim}_beta{beta}_{crop_suffix}_k{N_CLUSTERS}.csv"
        cluster_df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
        
        
        limit = min(len(X_embedded), 1000)
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_embedded[:limit, 0], 
            X_embedded[:limit, 1], 
            c=clusters[:limit], 
            cmap='tab10', 
            alpha=0.6
        )
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f"Frame Clustering LD={latent_dim}, Beta={beta}, K={N_CLUSTERS}")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.tight_layout()
        
        plot_path = RESULTS_DIR / f"clustering_ld{latent_dim}_beta{beta}_{crop_suffix}_k{N_CLUSTERS}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Clustering plot saved to: {plot_path}")

def perform_hdbscan_clustering_and_log(latent_vectors = latent_vectors, image_paths = image_paths, latent_dim=32, beta=2.0, crop_suffix="crop10"): 
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_method='eom',
        allow_single_cluster=True,
        metric='euclidean',
        algorithm='best',
        leaf_size=30
    )                       
    cluster_labels = clusterer.fit_predict(latent_vectors)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_vectors)
    
    cluster_df = pd.DataFrame({'image_path': image_paths, 'cluster_label': cluster_labels, 'umap_x': embedding[:, 0], 'umap_y': embedding[:, 1], 'probability': clusterer.probabilities_, '_data_level': 'frame'})
    
    table_path = RESULTS_DIR / f"hdbscan_table_ld{latent_dim}_beta{beta}_{crop_suffix}.json"
    cluster_df.to_json(table_path, orient='records', indent=2)
    print(f"HDBSCAN table saved to: {table_path}")
    
    csv_path = RESULTS_DIR / f"hdbscan_table_ld{latent_dim}_beta{beta}_{crop_suffix}.csv"
    cluster_df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    # Visualization 1: UMAP with cluster colors
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Cluster labels
    scatter1 = axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap='Spectral',
        alpha=0.6,
        s=10
    )
    axes[0].set_title(f'HDBSCAN Clusters (LD={latent_dim}, Beta={beta})\n{n_clusters} clusters, {n_noise} noise points')
    axes[0].set_xlabel('UMAP Dim 1')
    axes[0].set_ylabel('UMAP Dim 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
    
    # Right plot: Membership probabilities
    scatter2 = axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=clusterer.probabilities_,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    axes[1].set_title('Cluster Membership Probability')
    axes[1].set_xlabel('UMAP Dim 1')
    axes[1].set_ylabel('UMAP Dim 2')
    plt.colorbar(scatter2, ax=axes[1], label='Probability')

    plt.tight_layout()
    plot_path = RESULTS_DIR / f"hdbscan_ld{latent_dim}_beta{beta}_{crop_suffix}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"HDBSCAN plot saved to: {plot_path}")
    
    # Visualization 2: Condensed tree (shows cluster hierarchy)
    fig, ax = plt.subplots(figsize=(12, 8))
    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette='Spectral', axis=ax)
    plt.title('HDBSCAN Condensed Tree')
    plt.tight_layout()
    tree_path = RESULTS_DIR / f"hdbscan_tree_ld{latent_dim}_beta{beta}_{crop_suffix}.png"
    plt.savefig(tree_path, dpi=150)
    plt.close()
    print(f"Condensed tree saved to: {tree_path}")
    
    return cluster_df, clusterer
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run clustering on latent features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both methods
  python clustering.py --method both
  
  # Run only K-Means
  python clustering.py --method kmeans
  
  # Run only HDBSCAN
  python clustering.py --method hdbscan
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['kmeans', 'hdbscan', 'both'],
        default='both',
        help='Clustering method to run (default: both)'
    )
    
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=32,
        help='Latent dimension (default: 32)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=2.0,
        help='Beta value (default: 2.0)'
    )
    
    parser.add_argument(
        '--crop-suffix',
        type=str,
        default='crop10',
        help='Crop suffix for file naming (default: crop10)'
    )
    
    # Handle Jupyter/IPython environments
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        return parser.parse_args([])
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Running clustering with method: {args.method}")
    print(f"Parameters: latent_dim={args.latent_dim}, beta={args.beta}, crop_suffix={args.crop_suffix}")
    
    if args.method in ['kmeans', 'both']:
        perform_kmean_clustering_and_log(
            latent_vectors=latent_vectors,
            image_paths=image_paths,
            latent_dim=args.latent_dim,
            beta=args.beta,
            crop_suffix=args.crop_suffix
        )
    
    if args.method in ['hdbscan', 'both']:
        perform_hdbscan_clustering_and_log(
            latent_vectors=latent_vectors,
            image_paths=image_paths,
            latent_dim=args.latent_dim,
            beta=args.beta,
            crop_suffix=args.crop_suffix
        )
    
    print("Clustering complete!")
