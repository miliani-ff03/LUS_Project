"""
Clustering Metrics Evaluation.

Computes clustering quality metrics comparing VAE cluster assignments
against LUS scores.

Metrics computed:
- Silhouette Score (internal cluster quality)
- Davies-Bouldin Index (cluster separation)
- Calinski-Harabasz Index (cluster density)
- Mutual Information Score (cluster-score agreement)
- Adjusted Rand Index (cluster-score agreement)

Usage:
    python clustering_metrics.py
    
    Or with custom paths:
    python clustering_metrics.py --cluster-table path/to/export.csv
"""

import argparse

from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    mutual_info_score, 
    adjusted_rand_score
)

from utils import load_cluster_data, find_score


# ================= DEFAULT CONFIGURATION =================
DEFAULT_CLUSTER_TABLE = "/cosma/home/durham/dc-fras4/code/wandb/offline-run-20260121_145342-itgqznl8/files/media/table/video_cluster_labels_k4_2_972c7c613c8e85a4f63d.table.json"
# =========================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute clustering quality metrics")
    parser.add_argument(
        "--cluster-table", "-c",
        default=DEFAULT_CLUSTER_TABLE,
        help="Path to WandB exported cluster table (CSV or JSON)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data using shared utility
    cluster_df, df_metadata = load_cluster_data(args.cluster_table)
    
    # Detect data level
    data_level = cluster_df.get('_data_level', 'frame').iloc[0] if '_data_level' in cluster_df.columns else 'frame'
    print(f"Data level: {data_level.upper()}")
    
    # Map identifiers to scores
    print(f"Mapping {'video IDs' if data_level == 'video' else 'image paths'} to scores...")
    cluster_df['Score'] = cluster_df['image_path'].apply(lambda path: find_score(path, df_metadata))
    
    # Drop rows with NaN scores
    valid_clusters = cluster_df.dropna(subset=['Score'])
    print(f"Valid samples with scores: {len(valid_clusters)} / {len(cluster_df)}")
    
    # Extract necessary columns
    X = valid_clusters[['tsne_x', 'tsne_y']].values
    labels = valid_clusters['cluster_label'].values
    scores = valid_clusters['Score'].values
    
    print("\nCalculating clustering metrics...")
    print("=" * 50)
    
    # Internal clustering metrics
    silhouette_avg = silhouette_score(X, labels)
    davies_bouldin_avg = davies_bouldin_score(X, labels)
    calinski_harabasz_avg = calinski_harabasz_score(X, labels)
    
    # External clustering metrics (comparing to ground truth scores)
    mi_score = mutual_info_score(labels, scores)
    ari_score = adjusted_rand_score(labels, scores)
    
    print(f"Silhouette Score:        {silhouette_avg:.4f}  (higher is better, range [-1, 1])")
    print(f"Davies-Bouldin Index:    {davies_bouldin_avg:.4f}  (lower is better)")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_avg:.4f}  (higher is better)")
    print(f"Mutual Information:      {mi_score:.4f}  (higher means clusters correlate with scores)")
    print(f"Adjusted Rand Index:     {ari_score:.4f}  (1.0 = perfect agreement with scores)")
    print("=" * 50)


if __name__ == "__main__":
    main()
