"""
Frame-Level Analysis Package for Medical VAE Clustering.

This package contains scripts for training and analyzing VAE models
at the frame level (individual images, not grouped by video).

Modules:
    - train_vae: Train VAE model on individual frames
    - latent_classifier: Train classifier on frame-level latent embeddings
    - clustering_metrics: Compute clustering quality metrics
    - cluster_analysis: Generate interactive t-SNE visualizations
"""

from pathlib import Path

# Package-level constants
PACKAGE_DIR = Path(__file__).parent
RESULTS_DIR = PACKAGE_DIR / "results"
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    'PACKAGE_DIR',
    'RESULTS_DIR', 
    'CHECKPOINTS_DIR',
]
