"""
Latent Dimension to LUS Score Correlation Analysis.

This script analyzes whether specific latent dimensions correlate with
clinical LUS scores (0-3), helping determine if the VAE has learned
clinically meaningful representations.

Outputs:
1. Correlation heatmap: Which dimensions correlate with scores?
2. Box plots: Distribution of each dimension by LUS score
3. Feature importance: Ranking of dimensions by predictive power

Usage:
    python latent_score_correlation.py
    python latent_score_correlation.py --beta 2.0
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import find_score
from config import Config


# ================= CONFIGURATION =================
LATENT_DIM = 32
CROP_PERCENT = 10
AVAILABLE_BETAS = [1.0, 2.0, 5.0]
OUTPUT_DIR = Path("results/score_correlation")
# =================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Latent-score correlation analysis")
    parser.add_argument("--beta", "-b", type=float, default=None,
                       help="Beta value to analyze. If not specified, analyzes all.")
    return parser.parse_args()


def load_latent_data(beta: float) -> Optional[tuple]:
    """Load latent vectors, image paths, and metadata."""
    latent_path = f"results/latent_features/latent_vectors_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta}.npy"
    paths_path = f"results/latent_features/image_paths_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta}.npy"
    
    if not os.path.exists(latent_path) or not os.path.exists(paths_path):
        print(f"  Missing files for beta={beta}")
        return None
    
    X_latent = np.load(latent_path)
    image_paths = np.load(paths_path, allow_pickle=True)
    
    # Load metadata for score matching
    df_metadata = pd.read_csv(Config.METADATA_PATH)
    
    return X_latent, image_paths, df_metadata


def map_scores_to_latents(X_latent: np.ndarray, image_paths: np.ndarray, 
                          df_metadata: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with latent dimensions and their corresponding scores."""
    print("  Mapping image paths to LUS scores...")
    
    data = []
    for i, path in enumerate(image_paths):
        score = find_score(str(path), df_metadata)
        if not np.isnan(score):
            row = {'score': int(score), 'path': path}
            for dim in range(X_latent.shape[1]):
                row[f'z_{dim}'] = X_latent[i, dim]
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"  Successfully matched {len(df)} samples with scores")
    print(f"  Score distribution: {df['score'].value_counts().sort_index().to_dict()}")
    
    return df


def compute_correlations(df: pd.DataFrame, latent_dim: int) -> pd.DataFrame:
    """Compute Spearman correlation between each latent dimension and LUS score."""
    correlations = []
    
    for dim in range(latent_dim):
        col = f'z_{dim}'
        corr, pval = stats.spearmanr(df[col], df['score'])
        correlations.append({
            'dimension': dim,
            'correlation': corr,
            'p_value': pval,
            'abs_correlation': abs(corr),
            'significant': pval < 0.05
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    return corr_df


def plot_correlation_heatmap(corr_df: pd.DataFrame, beta: float, output_dir: Path):
    """Plot heatmap of all dimension correlations with scores."""
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Reshape correlations for heatmap (1 row, 32 columns)
    corr_values = corr_df.sort_values('dimension')['correlation'].values.reshape(1, -1)
    
    sns.heatmap(
        corr_values,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=0.5,
        xticklabels=[f'z_{i}' for i in range(len(corr_values[0]))],
        yticklabels=['Corr'],
        ax=ax,
        cbar_kws={'label': 'Spearman Correlation'}
    )
    
    ax.set_title(f'Latent Dimension ↔ LUS Score Correlation (β={beta})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Latent Dimension')
    
    plt.tight_layout()
    save_path = output_dir / f'beta{beta}_correlation_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_top_dimensions_by_score(df: pd.DataFrame, corr_df: pd.DataFrame, 
                                  beta: float, output_dir: Path, top_n: int = 6):
    """Plot box plots of top correlated dimensions grouped by LUS score."""
    top_dims = corr_df.head(top_n)['dimension'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, dim in enumerate(top_dims):
        col = f'z_{dim}'
        corr_val = corr_df[corr_df['dimension'] == dim]['correlation'].values[0]
        pval = corr_df[corr_df['dimension'] == dim]['p_value'].values[0]
        
        ax = axes[idx]
        
        # Box plot
        df.boxplot(column=col, by='score', ax=ax, grid=False)
        
        # Add individual points with jitter for better visualization
        scores = df['score'].unique()
        for score in sorted(scores):
            score_data = df[df['score'] == score][col]
            jitter = np.random.normal(0, 0.08, len(score_data))
            ax.scatter(score + 1 + jitter, score_data, alpha=0.3, s=5, color='steelblue')
        
        ax.set_title(f'Dimension z_{dim}\nρ={corr_val:.3f} (p={pval:.2e})', fontsize=11)
        ax.set_xlabel('LUS Score')
        ax.set_ylabel(f'z_{dim} value')
        
        # Color the title based on correlation direction
        if corr_val > 0.1:
            ax.title.set_color('darkred')
        elif corr_val < -0.1:
            ax.title.set_color('darkblue')
    
    plt.suptitle(f'Top {top_n} Correlated Latent Dimensions (β={beta})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / f'beta{beta}_top_dimensions_boxplot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_score_distribution_in_latent_space(df: pd.DataFrame, corr_df: pd.DataFrame,
                                             beta: float, output_dir: Path):
    """Plot 2D scatter of top 2 correlated dimensions colored by score."""
    top_2 = corr_df.head(2)['dimension'].tolist()
    
    if len(top_2) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        df[f'z_{top_2[0]}'],
        df[f'z_{top_2[1]}'],
        c=df['score'],
        cmap='RdYlGn_r',
        alpha=0.5,
        s=10
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('LUS Score', fontsize=12)
    
    corr1 = corr_df[corr_df['dimension'] == top_2[0]]['correlation'].values[0]
    corr2 = corr_df[corr_df['dimension'] == top_2[1]]['correlation'].values[0]
    
    ax.set_xlabel(f'z_{top_2[0]} (ρ={corr1:.3f})', fontsize=12)
    ax.set_ylabel(f'z_{top_2[1]} (ρ={corr2:.3f})', fontsize=12)
    ax.set_title(f'LUS Score Distribution in Top 2 Latent Dimensions (β={beta})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / f'beta{beta}_2d_score_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_linear_separability(df: pd.DataFrame, latent_dim: int) -> dict:
    """
    Test how well a linear classifier can predict LUS score from latent features.
    High accuracy = latent space captures score-relevant information.
    """
    X = df[[f'z_{i}' for i in range(latent_dim)]].values
    y = df['score'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    
    # Get accuracy and per-dimension importance
    accuracy = clf.score(X_scaled, y)
    
    # Per-dimension importance (mean absolute coefficient across classes)
    importance = np.mean(np.abs(clf.coef_), axis=0)
    
    return {
        'accuracy': accuracy,
        'importance': importance,
        'coef': clf.coef_
    }


def plot_feature_importance(linear_results: dict, corr_df: pd.DataFrame,
                            beta: float, output_dir: Path):
    """Plot feature importance from both correlation and linear model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correlation-based importance
    ax1 = axes[0]
    dims = corr_df.sort_values('dimension')['dimension'].values
    corrs = corr_df.sort_values('dimension')['abs_correlation'].values
    colors = ['darkred' if c > 0.1 else 'steelblue' for c in corrs]
    ax1.bar(dims, corrs, color=colors, alpha=0.7)
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Significance threshold')
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('|Spearman Correlation|')
    ax1.set_title('Correlation-based Importance')
    ax1.legend()
    
    # Linear model importance
    ax2 = axes[1]
    importance = linear_results['importance']
    colors = ['darkred' if imp > np.mean(importance) else 'steelblue' for imp in importance]
    ax2.bar(range(len(importance)), importance, color=colors, alpha=0.7)
    ax2.axhline(y=np.mean(importance), color='red', linestyle='--', alpha=0.5, label='Mean importance')
    ax2.set_xlabel('Latent Dimension')
    ax2.set_ylabel('Linear Model Coefficient')
    ax2.set_title(f'Linear Separability (Accuracy={linear_results["accuracy"]:.1%})')
    ax2.legend()
    
    plt.suptitle(f'Latent Dimension Importance for LUS Score Prediction (β={beta})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'beta{beta}_feature_importance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_summary_report(corr_df: pd.DataFrame, linear_results: dict,
                            beta: float, n_samples: int, output_dir: Path):
    """Generate a text summary of the analysis."""
    report_lines = [
        f"=" * 60,
        f"LATENT-SCORE CORRELATION REPORT (β={beta})",
        f"=" * 60,
        f"",
        f"Samples analyzed: {n_samples}",
        f"Linear classifier accuracy: {linear_results['accuracy']:.1%}",
        f"",
        f"TOP 10 CORRELATED DIMENSIONS:",
        f"-" * 40,
    ]
    
    for _, row in corr_df.head(10).iterrows():
        sig = "**" if row['significant'] else ""
        direction = "↑" if row['correlation'] > 0 else "↓"
        report_lines.append(
            f"  z_{int(row['dimension']):2d}: ρ={row['correlation']:+.3f} {direction} (p={row['p_value']:.2e}) {sig}"
        )
    
    report_lines.extend([
        "",
        "INTERPRETATION:",
        "-" * 40,
    ])
    
    # Interpretation
    best_corr = corr_df.iloc[0]['correlation']
    if abs(best_corr) > 0.3:
        report_lines.append("  ✅ STRONG: Some dimensions strongly correlate with LUS scores")
    elif abs(best_corr) > 0.15:
        report_lines.append("  🟡 MODERATE: Some dimensions moderately correlate with LUS scores")
    else:
        report_lines.append("  🔴 WEAK: No dimensions strongly correlate with LUS scores")
    
    if linear_results['accuracy'] > 0.6:
        report_lines.append("  ✅ GOOD: Latent space is linearly separable by score")
    elif linear_results['accuracy'] > 0.4:
        report_lines.append("  🟡 MODERATE: Partial linear separability by score")
    else:
        report_lines.append("  🔴 POOR: Latent space is not well-separated by score")
    
    report_lines.append(f"\n{'=' * 60}")
    
    report = "\n".join(report_lines)
    
    # Print to console
    print(report)
    
    # Save to file
    save_path = output_dir / f'beta{beta}_report.txt'
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {save_path}")


def analyze_beta(beta: float, output_dir: Path):
    """Run full correlation analysis for a single beta value."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing β = {beta}")
    print(f"{'=' * 60}")
    
    data = load_latent_data(beta)
    if data is None:
        return
    
    X_latent, image_paths, df_metadata = data
    
    # Map scores to latents
    df = map_scores_to_latents(X_latent, image_paths, df_metadata)
    
    if len(df) < 100:
        print(f"  ⚠️ Too few samples with scores ({len(df)}), skipping")
        return
    
    # Compute correlations
    print("\n  Computing correlations...")
    corr_df = compute_correlations(df, LATENT_DIM)
    
    # Compute linear separability
    print("  Computing linear separability...")
    linear_results = compute_linear_separability(df, LATENT_DIM)
    
    # Generate plots
    print("\n  Generating visualizations...")
    plot_correlation_heatmap(corr_df, beta, output_dir)
    plot_top_dimensions_by_score(df, corr_df, beta, output_dir)
    plot_score_distribution_in_latent_space(df, corr_df, beta, output_dir)
    plot_feature_importance(linear_results, corr_df, beta, output_dir)
    
    # Generate report
    print("\n")
    generate_summary_report(corr_df, linear_results, beta, len(df), output_dir)


def main():
    args = parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LATENT-SCORE CORRELATION ANALYSIS")
    print("=" * 60)
    
    if args.beta is not None:
        analyze_beta(args.beta, OUTPUT_DIR)
    else:
        for beta in AVAILABLE_BETAS:
            analyze_beta(beta, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
