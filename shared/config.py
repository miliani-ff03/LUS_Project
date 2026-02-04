"""
Centralized configuration for the Medical VAE Clustering project.

Usage:
    from shared.config import Config
    
    # Access paths
    data_path = Config.DATA_DIR / "for_vae" / "all_images"
    metadata = Config.METADATA_PATH
"""

from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProjectConfig:
    """Project-wide configuration with sensible defaults for COSMA environment."""
    
    # Base directories
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Data paths (COSMA5 storage)
    DATA_DIR: Path = Path("/cosma5/data/durham/dc-fras4/ultrasound/output_frames")
    VAE_DATA_PATH: Path = DATA_DIR / "for_vae" / "all_images"
    
    # Metadata
    METADATA_PATH: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_preprocessing" / "data_tables" / "all_data.csv")
    
    # Output directories (level-specific results folders are in frame_level/ and video_level/)
    RESULTS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    
    # Model defaults
    DEFAULT_LATENT_DIM: int = 32
    DEFAULT_BATCH_SIZE: int = 64
    DEFAULT_EPOCHS: int = 60
    DEFAULT_BETA: float = 1.0
    DEFAULT_CROP_PERCENT: float = 0.10
    
    # Image settings
    IMAGE_SIZE: int = 64
    IMAGE_CHANNELS: int = 3
    
    # Hospital identifiers used in data
    HOSPITALS: tuple = ("JCUH", "MFT", "UHW")
    
    def __post_init__(self):
        """Ensure output directories exist."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "ProjectConfig":
        """
        Create config with optional environment variable overrides.
        
        Environment variables:
            LUS_DATA_DIR: Override data directory
            LUS_METADATA_PATH: Override metadata file path
        """
        config = cls()
        
        if data_dir := os.environ.get("LUS_DATA_DIR"):
            config.DATA_DIR = Path(data_dir)
            config.VAE_DATA_PATH = config.DATA_DIR / "for_vae" / "all_images"
        
        if metadata_path := os.environ.get("LUS_METADATA_PATH"):
            config.METADATA_PATH = Path(metadata_path)
        
        return config


# Global config instance - import this
Config = ProjectConfig()
