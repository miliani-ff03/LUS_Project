"""
Frame-Level Clustering Metrics Evaluation.

Computes clustering quality metrics comparing VAE cluster assignments
against LUS scores at the frame level.

Usage:
    python -m frame_level.clustering_metrics
    python frame_level/clustering_metrics.py --cluster-table path/to/table.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    mutual_info_score,
    adjusted_rand_score
)

from shared.utils import load_cluster_data, find_score

# ==========================================
# CONFIGURATION
# ==========================================

MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default cluster table path (update as needed)
DEFAULT_CLUSTER_TABLE = str(MODULE_DIR.parent / "wandb" / "latest-run" / "files" / "cluster_table.json")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute frame-level clustering quality metrics")
    parser.add_argument(
        "--cluster-table", "-c",
        default=DEFAULT_CLUSTER_TABLE,
        help="Path to WandB exported cluster table (CSV or JSON)"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(RESULTS_DIR / "clustering_metrics.csv"),
        help="Output path for metrics CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data using shared utility
    cluster_df, df_metadata = load_cluster_data(args.cluster_table)
    
    # Detect data level
    data_level = cluster_df.get('_data_level', 'frame').iloc[0] if '_data_level' in cluster_df.columns else 'frame'
    print(f"Data level: {data_level.upper()}")
    
    if data_level != 'frame':
        print("Warning: This script is designed for frame-level data.")
    
    # Map identifiers to scores
    print("Mapping image paths to scores...")
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
    
    # Save to CSV
    import pandas as pd
    metrics_df = pd.DataFrame([{
        'level': 'frame',
        'silhouette_score': silhouette_avg,
        'davies_bouldin_index': davies_bouldin_avg,
        'calinski_harabasz_index': calinski_harabasz_avg,
        'mutual_information': mi_score,
        'adjusted_rand_index': ari_score,
        'n_samples': len(valid_clusters),
        'cluster_table': args.cluster_table
    }])
    metrics_df.to_csv(args.output, index=False)
    print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
