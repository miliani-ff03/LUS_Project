"""
Offline Sweep Launcher for Supervised VAE Training.

Runs a grid search over hyperparameters and logs all metrics,
plots, and clustering results to WandB for later analysis.

Usage:
    nohup python offline_launcher_for_sweep.py &> sweep_log.txt &
"""

import subprocess
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime

import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    davies_bouldin_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup wandb offline mode
os.environ["WANDB_MODE"] = "offline"

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not available, metrics will only be saved locally")

# ==========================================
# SWEEP CONFIGURATION
# ==========================================

LATENT_DIMS = [10, 20, 30]
BETAS = [1.0, 2.0, 5.0]
GAMMAS = [1.0, 2.0, 5.0, 10.0]
LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.01, 0.1]

# Fixed parameters
EPOCHS = 60
BATCH_SIZE = 64
CROP_PERCENT = 0.1
ANNEALING = "cyclical"
N_CLUSTERS = 4  # Match number of severity scores

# Paths
MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results" / "sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_NAME = "train_vae_with_classifier.py"


# ==========================================
# CLUSTERING & METRICS
# ==========================================

def compute_clustering_metrics(latent_vectors: np.ndarray, true_scores: np.ndarray, 
                                n_clusters: int = 4) -> dict:
    """
    Compute clustering metrics on latent space.
    
    Returns:
        Dictionary with clustering metrics and assignments
    """
    # Filter valid scores
    valid_mask = true_scores >= 0
    X_valid = latent_vectors[valid_mask]
    scores_valid = true_scores[valid_mask]
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_valid)
    
    # Internal metrics (cluster quality)
    db_score = davies_bouldin_score(X_valid, cluster_labels)
    silhouette = silhouette_score(X_valid, cluster_labels)
    calinski = calinski_harabasz_score(X_valid, cluster_labels)
    
    # External metrics (agreement with true scores)
    ari_score = adjusted_rand_score(scores_valid, cluster_labels)

    cm=confusion_matrix(scores_valid.astype(int), cluster_labels, labels=list(range(n_clusters)))
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize matches

    # Create mapping from cluster to score
    cluster_to_score = {col: row for row, col in zip(row_ind, col_ind)}
    
    # Map cluster labels to predicted scores
    predicted_scores = np.array([cluster_to_score[c] for c in cluster_labels])
    
    # Compute accuracy after Hungarian matching
    hungarian_accuracy = np.mean(predicted_scores == scores_valid.astype(int))
    
    # Also compute per-class accuracy after matching
    per_class_acc = {}
    for score in range(n_clusters):
        mask = scores_valid.astype(int) == score
        if mask.sum() > 0:
            per_class_acc[f'score_{score}_acc'] = np.mean(predicted_scores[mask] == score)

    
    # t-SNE for visualization
    print("Computing t-SNE...")
    max_samples = 3000
    if len(X_valid) > max_samples:
        idx = np.random.choice(len(X_valid), max_samples, replace=False)
        X_tsne = X_valid[idx]
        labels_tsne = cluster_labels[idx]
        scores_tsne = scores_valid[idx]
    else:
        X_tsne = X_valid
        labels_tsne = cluster_labels
        scores_tsne = scores_valid
        idx = np.arange(len(X_valid))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_tsne)
    
    return {
        'davies_bouldin': db_score,
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'adjusted_rand_index': ari_score,
        'hungarian_accuracy': hungarian_accuracy,
        'cluster_to_score_mapping': cluster_to_score,
        'per_class_accuracy': per_class_acc,
        'cluster_labels': cluster_labels,
        'predicted_scores': predicted_scores,
        'tsne_coords': X_embedded,
        'tsne_cluster_labels': labels_tsne,
        'tsne_scores': scores_tsne,
        'tsne_indices': idx,
        'kmeans_centers': kmeans.cluster_centers_
    }


def create_cluster_table(image_paths: np.ndarray, latent_vectors: np.ndarray,
                         true_scores: np.ndarray, cluster_results: dict,
                         tsne_indices: np.ndarray) -> pd.DataFrame:
    """Create a cluster table for WandB logging."""
    # Filter to valid samples used in clustering
    valid_mask = true_scores >= 0
    valid_paths = image_paths[valid_mask]
    valid_scores = true_scores[valid_mask]
    
    # Get t-SNE subset
    tsne_paths = valid_paths[tsne_indices]
    tsne_scores = valid_scores[tsne_indices]
    
    df = pd.DataFrame({
        'image_path': tsne_paths,
        'true_score': tsne_scores,
        'cluster_label': cluster_results['tsne_cluster_labels'],
        'tsne_x': cluster_results['tsne_coords'][:, 0],
        'tsne_y': cluster_results['tsne_coords'][:, 1]
    })
    
    return df


def log_clustering_plots(cluster_results: dict, save_dir: Path, suffix: str):
    """Generate and save clustering visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    tsne_coords = cluster_results['tsne_coords']
    cluster_labels = cluster_results['tsne_cluster_labels']
    true_scores = cluster_results['tsne_scores']
    
    # Plot 1: t-SNE colored by cluster
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                c=cluster_labels, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    axes[0].set_title('t-SNE by K-Means Cluster')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Plot 2: t-SNE colored by true score
    scatter2 = axes[1].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                c=true_scores, cmap='RdYlGn_r', alpha=0.6, s=10)
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    axes[1].set_title('t-SNE by True Score')
    plt.colorbar(scatter2, ax=axes[1], label='Score')
    
    plt.tight_layout()
    tsne_path = save_dir / f"tsne_clustering_{suffix}.png"
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    
    # Plot 3: Cluster vs Score heatmap
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_scores.astype(int), cluster_labels, 
                          labels=list(range(4)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'Cluster {i}' for i in range(4)],
                yticklabels=[f'Score {i}' for i in range(4)])
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Score')
    ax.set_title(f'Cluster vs Score Distribution\n{suffix}')
    
    heatmap_path = save_dir / f"cluster_score_heatmap_{suffix}.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    
    return tsne_path, heatmap_path


# ==========================================
# SINGLE RUN EXECUTION
# ==========================================

def run_single_experiment(latent_dim: int, beta: float, gamma: float, 
                          learning_rate: float, run_idx: int, total_runs: int) -> dict:
    """
    Run a single training experiment and compute all metrics.
    
    Returns:
        Dictionary with all results and metrics
    """
    suffix = f"ld{latent_dim}_beta{beta}_gamma{gamma}_lr{learning_rate}"
    print(f"\n{'='*60}")
    print(f"Run {run_idx}/{total_runs}: {suffix}")
    print(f"{'='*60}")
    
    # Create run-specific results directory
    run_dir = RESULTS_DIR / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb run
    if HAS_WANDB:
        wandb.init(
            project="lus-medical-vae",
            group="sweep_supervised",
            name=suffix,
            config={
                "latent_dim": latent_dim,
                "beta": beta,
                "gamma": gamma,
                "learning_rate": learning_rate,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "crop_percent": CROP_PERCENT,
                "annealing": ANNEALING,
                "n_clusters": N_CLUSTERS
            },
            dir=str(run_dir),
            reinit=True
        )
    
    # Run the training script
    cmd = [
        sys.executable, SCRIPT_NAME,
        "--latent_dim", str(latent_dim),
        "--beta", str(beta),
        "--gamma", str(gamma),
        "--learning_rate", str(learning_rate),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--crop_percent", str(CROP_PERCENT),
        "--annealing", ANNEALING,
        "--use_class_weights",
        "--skip_viz"  # We'll generate our own plots
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"stderr: {e.stderr}")
        if HAS_WANDB:
            wandb.finish(exit_code=1)
        return {"status": "failed", "error": str(e)}
    
    # Load saved latent features
    latent_suffix = f"ld{latent_dim}_crop{int(CROP_PERCENT*100)}_beta{beta}_gamma{gamma}"
    latent_dir = MODULE_DIR / "results" / "supervised" / "latent_features"
    
    try:
        latent_vectors = np.load(latent_dir / f"latent_vectors_{latent_suffix}.npy")
        image_paths = np.load(latent_dir / f"image_paths_{latent_suffix}.npy", allow_pickle=True)
        scores = np.load(latent_dir / f"scores_{latent_suffix}.npy")
        split_labels = np.load(latent_dir / f"split_labels_{latent_suffix}.npy", allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Could not load latent features: {e}")
        if HAS_WANDB:
            wandb.finish(exit_code=1)
        return {"status": "failed", "error": f"Missing latent features: {e}"}
    
    # Load best metrics from training
    model_name = f"SupervisedVAE_ld{latent_dim}_crop{int(CROP_PERCENT*100)}_beta{beta}_gamma{gamma}_{ANNEALING}"
    metrics_file = MODULE_DIR / "checkpoints" / f"best_metrics_{model_name}.json"
    
    best_val_acc = None
    best_val_loss = None
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            train_metrics = json.load(f)
            best_val_acc = train_metrics.get('best_val_accuracy')
            best_val_loss = train_metrics.get('best_val_loss')
            print(f"Loaded training metrics: val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
    else:
        print(f"Warning: Could not find metrics file: {metrics_file}")
    
    # Compute clustering metrics
    print("\nComputing clustering metrics...")
    cluster_results = compute_clustering_metrics(latent_vectors, scores, N_CLUSTERS)
    
    if cluster_results is None:
        if HAS_WANDB:
            wandb.finish(exit_code=1)
        return {"status": "failed", "error": "Clustering failed"}
    
    # Generate clustering plots
    print("Generating clustering plots...")
    tsne_path, heatmap_path = log_clustering_plots(cluster_results, run_dir, suffix)
    
    # Create cluster table
    cluster_df = create_cluster_table(
        image_paths, latent_vectors, scores, 
        cluster_results, cluster_results['tsne_indices']
    )
    cluster_table_path = run_dir / f"cluster_table_{suffix}.csv"
    cluster_df.to_csv(cluster_table_path, index=False)
    
    # Also save as JSON for compatibility
    cluster_json_path = run_dir / f"cluster_table_{suffix}.json"
    cluster_df.to_json(cluster_json_path, orient='records')
    
    # Extract best metrics from training (parse from stdout or load from saved files)
    # Load the loss tracker if saved, otherwise estimate from final values
    results_supervised = MODULE_DIR / "results" / "supervised"
    
    # Log everything to wandb
    if HAS_WANDB:
        # Log scalar metrics
        log_dict = {
            "clustering/davies_bouldin": cluster_results['davies_bouldin'],
            "clustering/silhouette": cluster_results['silhouette'],
            "clustering/calinski_harabasz": cluster_results['calinski_harabasz'],
            "clustering/adjusted_rand_index": cluster_results['adjusted_rand_index'],
            "clustering/hungarian_accuracy": cluster_results['hungarian_accuracy'],

        }

        # Add training metrics if available
        if best_val_acc is not None:
            log_dict["training/best_val_accuracy"] = best_val_acc
        if best_val_loss is not None:
            log_dict["training/best_val_loss"] = best_val_loss
        
        wandb.log(log_dict)

        for k, v in cluster_results['per_class_accuracy'].items():
            wandb.log({f"clustering/{k}": v})
        
        # Log images
        wandb.log({
            "plots/tsne_clustering": wandb.Image(str(tsne_path)),
            "plots/cluster_score_heatmap": wandb.Image(str(heatmap_path)),
        })
        
        # Log any existing plots from the training script
        for plot_name in ["confusion_matrix", "tsne_by_score", "reconstruction", "loss_curves"]:
            plot_path = results_supervised / f"{plot_name}_{latent_suffix}.png"
            if plot_path.exists():
                wandb.log({f"plots/{plot_name}": wandb.Image(str(plot_path))})
        
        # Log cluster table as wandb Table
        cluster_table = wandb.Table(dataframe=cluster_df)
        wandb.log({"cluster_table": cluster_table})
        
        # Save artifacts
        artifact = wandb.Artifact(
            name=f"run_results_{suffix}",
            type="results"
        )
        artifact.add_file(str(cluster_table_path))
        artifact.add_file(str(tsne_path))
        artifact.add_file(str(heatmap_path))
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    # Compile results
    run_results = {
        "status": "success",
        "config": {
            "latent_dim": latent_dim,
            "beta": beta,
            "gamma": gamma,
            "learning_rate": learning_rate
        },
        "metrics": {
            "davies_bouldin": cluster_results['davies_bouldin'],
            "silhouette": cluster_results['silhouette'],
            "calinski_harabasz": cluster_results['calinski_harabasz'],
            "adjusted_rand_index": cluster_results['adjusted_rand_index'],
            "hungarian_accuracy": cluster_results['hungarian_accuracy'],
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            **cluster_results['per_class_accuracy']
        },
        "cluster_to_score_mapping": cluster_results['cluster_to_score_mapping'],
        "paths": {
            "cluster_table": str(cluster_table_path),
            "tsne_plot": str(tsne_path),
            "heatmap_plot": str(heatmap_path)
        }
    }
    
    # Save run results
    with open(run_dir / "run_results.json", 'w') as f:
        json.dump(run_results, f, indent=2)
    
    return run_results


# ==========================================
# MAIN SWEEP
# ==========================================

def main():
    print("=" * 60)
    print("OFFLINE SWEEP LAUNCHER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Generate all combinations
    combinations = list(product(LATENT_DIMS, BETAS, GAMMAS, LEARNING_RATES))
    total_runs = len(combinations)
    
    print(f"\nSweep Configuration:")
    print(f"  Latent dims: {LATENT_DIMS}")
    print(f"  Betas: {BETAS}")
    print(f"  Gammas: {GAMMAS}")
    print(f"  Learning rates: {LEARNING_RATES}")
    print(f"  Total runs: {total_runs}")
    print(f"  Results dir: {RESULTS_DIR}")
    
    # Track all results
    all_results = []
    
    for i, (latent_dim, beta, gamma, lr) in enumerate(combinations, 1):
        result = run_single_experiment(
            latent_dim=latent_dim,
            beta=beta,
            gamma=gamma,
            learning_rate=lr,
            run_idx=i,
            total_runs=total_runs
        )
        all_results.append(result)
        
        # Save intermediate summary
        summary_path = RESULTS_DIR / "sweep_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Generate final summary
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    
    # Create summary DataFrame
    summary_rows = []
    for r in all_results:
        if r["status"] == "success":
            row = {**r["config"], **r["metrics"]}
            summary_rows.append(row)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(RESULTS_DIR / "sweep_metrics_summary.csv", index=False)

        print("\nTop 5 by Validation Accuracy:")
        print(summary_df.nlargest(5, 'best_val_accuracy')[
            ['latent_dim', 'beta', 'gamma', 'learning_rate', 'best_val_accuracy', 'best_val_loss', 'adjusted_rand_index']
        ].to_string(index=False))
        
        print("\nTop 5 by Validation Loss (lower is better):")
        print(summary_df.nsmallest(5, 'best_val_loss')[
            ['latent_dim', 'beta', 'gamma', 'learning_rate', 'best_val_loss', 'best_val_accuracy', 'adjusted_rand_index']
        ].to_string(index=False))
        
        print("\nTop 5 by Adjusted Rand Index:")
        print(summary_df.nlargest(5, 'adjusted_rand_index')[
            ['latent_dim', 'beta', 'gamma', 'learning_rate', 'adjusted_rand_index', 'davies_bouldin']
        ].to_string(index=False))
        
        print("\nTop 5 by Davies-Bouldin (lower is better):")
        print(summary_df.nsmallest(5, 'davies_bouldin')[
            ['latent_dim', 'beta', 'gamma', 'learning_rate', 'davies_bouldin', 'adjusted_rand_index']
        ].to_string(index=False))
    
    failed = sum(1 for r in all_results if r["status"] == "failed")
    print(f"\nCompleted: {total_runs - failed}/{total_runs} runs")
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nTo upload to WandB, run:")
    print(f"  wandb sync --include-offline {RESULTS_DIR}/*/wandb/offline-run-*")


if __name__ == "__main__":
    main()