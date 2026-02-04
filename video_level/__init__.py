"""
Video-Level Analysis Package for Medical VAE Clustering.

This package contains scripts for training and analyzing VAE models
at the video level (frames grouped by video ID with aggregation).

Modules:
    - train_vae: Train VAE model with video-level aggregation
    - latent_classifier: Train classifier on video-level latent embeddings
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
