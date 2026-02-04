"""
Shared components for Medical VAE Clustering project.

This module contains common utilities, models, and configuration
used by both frame_level and video_level analysis packages.
"""

from .config import Config, ProjectConfig
from .utils import (
    load_cluster_data,
    find_score,
    get_hospital_from_path
)
from .models import (
    ConvVAE,
    KLAnnealer,
    CyclicalAnnealer,
    EarlyStopping,
    LossTracker,
    TransformerVideoAggregator,
    GatedAttention,
    LatentScoreClassifier,
    vae_loss_function,
    aggregate_frame_latents
)

__all__ = [
    # Config
    'Config',
    'ProjectConfig',
    # Utils
    'load_cluster_data',
    'find_score',
    'get_hospital_from_path',
    # Models
    'ConvVAE',
    'KLAnnealer',
    'CyclicalAnnealer',
    'EarlyStopping',
    'LossTracker',
    'TransformerVideoAggregator',
    'GatedAttention',
    'LatentScoreClassifier',
    'vae_loss_function',
    'aggregate_frame_latents',
]
