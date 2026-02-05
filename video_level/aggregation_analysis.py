"""
Aggregation Analysis - Visualize Frame Importance

Analyzes which frames the transformer aggregator focuses on when creating
video-level embeddings. Uses attention weights or gradient-based attribution.

Usage:
    python video_level/aggregation_analysis.py --model_path path/to/aggregator.pth --video_idx 0
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import ConvVAE, TransformerVideoAggregator, LatentScoreClassifier
from shared.config import Config
from shared.utils import find_score


# ==========================================
# CONFIGURATION
# ==========================================

MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default checkpoint paths
VAE_CHECKPOINTS_DIR = MODULE_DIR.parent / "VAE_checkpoints"

IMAGE_SIZE = 64
CHANNELS = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==========================================
# MODEL LOADING
# ==========================================

def load_models(aggregator_path: str, latent_dim: int, vae_path: str = None, 
                beta: float = 2.0, device: str = DEVICE):
    """
    Load VAE and Transformer Aggregator from checkpoints.
    
    Args:
        aggregator_path: Path to the aggregator checkpoint
        latent_dim: Dimension of the latent space
        vae_path: Path to VAE checkpoint. If None, auto-detect from aggregator filename
        beta: Beta value used in VAE training (for auto-detection)
        device: torch device
        
    Returns:
        vae: Loaded ConvVAE model
        aggregator: Loaded TransformerVideoAggregator model
        classifier: Loaded classifier (if exists in checkpoint)
    """
    print(f"Loading models...")
    
    # Load aggregator checkpoint
    agg_checkpoint = torch.load(aggregator_path, map_location=device)
    
    # Check if it's a combined checkpoint or raw state_dict
    is_combined = 'vae_state_dict' in agg_checkpoint or 'aggregator_state_dict' in agg_checkpoint
    
    # Initialize and load VAE
    vae = ConvVAE(latent_dim=latent_dim, channels=CHANNELS).to(device)
    
    if is_combined and 'vae_state_dict' in agg_checkpoint:
        vae.load_state_dict(agg_checkpoint['vae_state_dict'])
        print(f"  ✓ Loaded VAE from combined checkpoint")
    else:
        # Auto-detect VAE path if not provided
        if vae_path is None:
            vae_path = VAE_CHECKPOINTS_DIR / f"Best_VAE_ld{latent_dim}_crop10_beta{beta}_cyclical.pth"
        
        if Path(vae_path).exists():
            vae_checkpoint = torch.load(vae_path, map_location=device)
            vae.load_state_dict(vae_checkpoint)
            print(f"  ✓ Loaded VAE from: {vae_path}")
        else:
            print(f"  ⚠ VAE checkpoint not found: {vae_path}")
            print(f"    Available: {list(VAE_CHECKPOINTS_DIR.glob('*.pth'))}")
    
    # Infer n_heads from checkpoint to handle non-divisible latent dims (e.g., 23)
    # in_proj_weight shape is [3*embed_dim, embed_dim], we need n_heads that divides embed_dim
    n_heads = 4  # default
    if 'transformer.layers.0.self_attn.in_proj_weight' in agg_checkpoint:
        # Find largest factor of latent_dim that is <= 4
        for h in [4, 2, 1]:
            if latent_dim % h == 0:
                n_heads = h
                break
        print(f"  Using n_heads={n_heads} for latent_dim={latent_dim}")
    
    # Initialize and load Transformer Aggregator
    aggregator = TransformerVideoAggregator(latent_dim=latent_dim, n_heads=n_heads).to(device)
    
    if is_combined and 'aggregator_state_dict' in agg_checkpoint:
        aggregator.load_state_dict(agg_checkpoint['aggregator_state_dict'])
        print(f"  ✓ Loaded Aggregator from combined checkpoint")
    else:
        # Raw state_dict (just the aggregator weights)
        aggregator.load_state_dict(agg_checkpoint)
        print(f"  ✓ Loaded Aggregator from: {aggregator_path}")
    
    # Try to load classifier if present
    classifier = None
    if is_combined and 'classifier_state_dict' in agg_checkpoint:
        classifier = LatentScoreClassifier(latent_dim=latent_dim, num_classes=4).to(device)
        classifier.load_state_dict(agg_checkpoint['classifier_state_dict'])
        classifier.eval()
        print("  ✓ Loaded Classifier weights")
    
    vae.eval()
    aggregator.eval()
    return vae, aggregator, classifier


def get_frame_transform(crop_percent: float = 0.1):
    """Get the transform used during training."""
    crop_pixels = int(IMAGE_SIZE * crop_percent)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # CRITICAL: grayscale, not RGB
        transforms.Resize((IMAGE_SIZE + crop_pixels, IMAGE_SIZE)),
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


# ==========================================
# VIDEO DATA LOADING
# ==========================================

def load_video_data(data_dir: str, metadata_path: str, frames_per_video: int = 10):
    """
    Load video frame paths grouped by video ID with their scores.
    
    Returns:
        List of (video_id, frame_paths, score) tuples
    """
    all_images = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
    metadata_df = pd.read_csv(metadata_path)
    
    video_frames = defaultdict(list)
    for img_path in all_images:
        video_id = _extract_video_id(img_path)
        video_frames[video_id].append(img_path)
    
    samples = []
    for video_id, frames in video_frames.items():
        frames.sort(key=lambda x: _extract_frame_number(x))
        
        if len(frames) != frames_per_video:
            continue
        
        score = find_score(frames[0], metadata_df)
        if not np.isnan(score):
            samples.append((video_id, frames[:frames_per_video], int(score)))
    
    print(f"Found {len(samples)} complete videos with valid scores")
    return samples


def _extract_video_id(path: str) -> str:
    """Extract video ID prefixed with hospital."""
    filename = os.path.basename(path)
    hospital = None
    for h in ['JCUH', 'MFT', 'UHW']:
        if h in path:
            hospital = h
            break
    
    match = re.match(r"(.+?)_selected_frame_\d+\.png", filename, re.IGNORECASE)
    if match:
        base_id = match.group(1)
    else:
        base_id = os.path.basename(os.path.dirname(path))
    
    if hospital:
        return f"{hospital}_{base_id}"
    return base_id


def _extract_frame_number(path: str) -> int:
    """Extract frame number from filename for sorting."""
    filename = os.path.basename(path)
    match = re.search(r'_selected_frame_(\d+)\.png', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbers = re.findall(r'(\d+)', filename)
    if numbers:
        return int(numbers[-1])
    return 0


# ==========================================
# ATTENTION ANALYSIS
# ==========================================

def compute_attention_weights(vae, aggregator, frames_tensor, device=DEVICE):
    """
    Compute attention weights from transformer using a hook.
    
    For TransformerEncoder, we hook into the self-attention layers to
    extract attention weights for the CLS token attending to frame tokens.
    """
    attn_weights_list = []
    
    def hook_fn(module, input, output):
        """Hook to capture attention weights from MultiheadAttention."""
        # MultiheadAttention returns (attn_output, attn_weights) when need_weights=True
        # But TransformerEncoderLayer doesn't expose weights by default
        # We'll use gradient-based method as primary approach
        pass
    
    # For now, use gradient-based importance as it's more reliable
    return compute_gradient_importance(vae, aggregator, frames_tensor, device)


def compute_gradient_importance(vae, aggregator, frames_tensor, device=DEVICE):
    """
    Compute frame importance using gradient-based attribution.
    
    This measures how much each frame contributes to the final video embedding
    by computing the gradient of the embedding norm with respect to each frame.
    """
    frames_tensor = frames_tensor.clone().requires_grad_(True).to(device)
    
    # Encode frames through VAE
    latents = vae.encode(frames_tensor)  # (10, latent_dim)
    latents = latents.unsqueeze(0)  # (1, 10, latent_dim)
    
    # Aggregate through transformer
    video_embedding = aggregator(latents)  # (1, latent_dim)
    
    # Compute gradient of embedding norm w.r.t. input frames
    loss = video_embedding.norm()
    loss.backward()
    
    # Frame importance = gradient magnitude per frame
    importance = frames_tensor.grad.abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
    
    # Normalize to [0, 1]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    return importance


def compute_latent_space_importance(vae, aggregator, frames_tensor, device=DEVICE):
    """
    Alternative: Compute importance based on latent space gradients.
    
    This is more efficient as it works in the lower-dimensional latent space.
    """
    with torch.no_grad():
        latents = vae.encode(frames_tensor.to(device))  # (10, latent_dim)
    
    latents = latents.clone().requires_grad_(True)
    latents_batch = latents.unsqueeze(0)  # (1, 10, latent_dim)
    
    video_embedding = aggregator(latents_batch)
    loss = video_embedding.norm()
    loss.backward()
    
    # Importance based on latent gradient magnitude
    importance = latents.grad.abs().mean(dim=1).detach().cpu().numpy()
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    return importance


# ==========================================
# VISUALIZATION
# ==========================================

def visualize_importance(vae, aggregator, video_frames_paths, video_id, true_score, 
                        classifier=None, device=DEVICE, save_path=None):
    """
    Visualize frame importance based on gradient attribution.
    
    Args:
        vae: Loaded ConvVAE model
        aggregator: Loaded TransformerVideoAggregator model
        video_frames_paths: List of paths to video frames
        video_id: ID of the video
        true_score: Ground truth severity score
        classifier: Optional classifier for prediction
        device: torch device
        save_path: Path to save the figure (optional)
    """
    vae.eval()
    aggregator.eval()
    
    transform = get_frame_transform()
    
    # 1. Load and preprocess frames
    frames_list = []
    for p in video_frames_paths:
        img = Image.open(p).convert('L')  # Grayscale!
        img_tensor = transform(img)
        frames_list.append(img_tensor)
    
    frames_tensor = torch.stack(frames_list)  # (10, 1, 64, 64)
    
    # 2. Compute importance weights
    importance = compute_gradient_importance(vae, aggregator, frames_tensor, device)
    
    # 3. Get prediction if classifier available
    pred_score = None
    if classifier is not None:
        classifier.eval()
        with torch.no_grad():
            latents = vae.encode(frames_tensor.to(device))
            video_emb = aggregator(latents.unsqueeze(0))
            pred_logits = classifier(video_emb)
            pred_score = torch.argmax(pred_logits, dim=1).item()
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    title = f"Video: {video_id} | True Score: {true_score}"
    if pred_score is not None:
        title += f" | Predicted: {pred_score}"
    plt.suptitle(title, fontsize=16)

    for i in range(min(10, len(video_frames_paths))):
        ax = axes[i // 5, i % 5]
        
        # Load original image for display
        img = plt.imread(video_frames_paths[i])
        ax.imshow(img, cmap='gray')
        
        # Color the border based on importance (red = high)
        frame_importance = importance[i] if i < len(importance) else 0
        color = plt.cm.Reds(frame_importance)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
            
        ax.set_title(f"Frame {i+1} | Imp: {frame_importance:.3f}")
        ax.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    return importance


def visualize_multiple_videos(vae, aggregator, samples, n_videos=5, classifier=None, 
                              device=DEVICE, output_dir=None):
    """Visualize importance for multiple videos."""
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(n_videos, len(samples))):
        video_id, frame_paths, true_score = samples[i]
        
        save_path = None
        if output_dir:
            safe_id = video_id.replace('/', '_')
            save_path = output_dir / f"attention_analysis_{safe_id}.png"
        
        print(f"\nAnalyzing video {i+1}/{n_videos}: {video_id}")
        visualize_importance(vae, aggregator, frame_paths, video_id, true_score,
                           classifier=classifier, device=device, save_path=save_path)


# ==========================================
# MAIN
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze transformer aggregator attention")
    parser.add_argument('--model_path', type=str, 
                       default=str(MODULE_DIR / "Transformer_Aggregator_ld23_crop10_beta2.0_transformer.pth"),
                       help="Path to transformer aggregator checkpoint")
    parser.add_argument('--vae_path', type=str, default=None,
                       help="Path to VAE checkpoint (auto-detected if not provided)")
    parser.add_argument('--latent_dim', type=int, default=23,
                       help="Latent dimension (must match checkpoint)")
    parser.add_argument('--beta', type=float, default=2.0,
                       help="Beta value used in VAE training (for auto-detecting VAE checkpoint)")
    parser.add_argument('--video_idx', type=int, default=0,
                       help="Index of video to analyze (or -1 for multiple)")
    parser.add_argument('--n_videos', type=int, default=5,
                       help="Number of videos to analyze if video_idx=-1")
    parser.add_argument('--data_dir', type=str, default=str(Config.VAE_DATA_PATH),
                       help="Directory containing frame images")
    parser.add_argument('--metadata_path', type=str, default=str(Config.METADATA_PATH),
                       help="Path to metadata CSV")
    parser.add_argument('--output_dir', type=str, default=str(RESULTS_DIR / "attention_analysis"),
                       help="Directory to save output figures")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Loading model from: {args.model_path}")
    
    # Load models
    vae, aggregator, classifier = load_models(
        aggregator_path=args.model_path, 
        latent_dim=args.latent_dim,
        vae_path=args.vae_path,
        beta=args.beta,
        device=DEVICE
    )
    
    # Load video data
    print(f"\nLoading video data from: {args.data_dir}")
    samples = load_video_data(args.data_dir, args.metadata_path)
    
    if len(samples) == 0:
        print("No valid videos found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.video_idx >= 0:
        # Analyze single video
        if args.video_idx >= len(samples):
            print(f"Video index {args.video_idx} out of range (max: {len(samples)-1})")
            return
        
        video_id, frame_paths, true_score = samples[args.video_idx]
        safe_id = video_id.replace('/', '_')
        save_path = output_dir / f"attention_analysis_{safe_id}.png"
        
        print(f"\nAnalyzing video: {video_id}")
        importance = visualize_importance(vae, aggregator, frame_paths, video_id, true_score,
                                         classifier=classifier, device=DEVICE, save_path=save_path)
        
        print(f"\nFrame importance scores:")
        for i, imp in enumerate(importance):
            print(f"  Frame {i+1}: {imp:.4f}")
    else:
        # Analyze multiple videos
        visualize_multiple_videos(vae, aggregator, samples, n_videos=args.n_videos,
                                 classifier=classifier, device=DEVICE, output_dir=output_dir)


if __name__ == "__main__":
    main()