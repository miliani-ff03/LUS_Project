# Lung Ultrasound (LUS) VAE Clustering Project - AI Agent Instructions

## Project Overview
Medical imaging analysis pipeline using **Variational Autoencoders (VAE)** to learn latent representations of lung ultrasound videos, then classify them by severity scores (0-3). Runs on **COSMA HPC cluster** (Durham University).

## Environment Setup
**CRITICAL**: Always use the `LUS` conda environment to avoid numpy/pandas binary compatibility issues:
```bash
conda activate LUS
```
All Python commands must be run with this environment activated. The base conda environment has incompatible package versions that will cause import errors.

## Architecture & Key Components

### 1. Data Pipeline
- **Source**: 3 hospitals (JCUH, MFT, UHW) → frames stored at `/cosma5/data/durham/dc-fras4/ultrasound/output_frames/`
- **Metadata**: `data_preprocessing/data_tables/all_data.csv` maps videos to severity scores
- **Video ID Format**: `{HOSPITAL}_{base_video_id}` (e.g., `JCUH_27_LU_4_RPB`)
- **Frame Naming**: `{video_id}_selected_frame_NNNN.png` (10 frames per video)

### 2. Core Models (`Medical_VAE_Video_Clustering.py`)

**ConvVAE Architecture**:
- Input: 64×64 grayscale images (CHANNELS=1, crop top 10% by default)
- Encoder: 4 conv layers → latent space (default: 32 dims)
- Decoder: 4 transposed conv layers → reconstruction
- Key parameter: `beta` (KL divergence weight, cyclical annealing by default)

**Video Aggregation Methods** (frame embeddings → video embedding):
- `mean`: Average pooling (fastest)
- `max`: Max pooling
- `concat`: Concatenate all frames (320 dims for 10 frames)
- `transformer`: `TransformerVideoAggregator` with CLS token (like BERT)

### 3. Latent Score Classifier (`latent_classifier.py`)
- MLP: `latent_dim → [64, 32] → 4 classes` (scores 0-3)
- Input: Video-level embeddings from VAE
- Output: Predicted severity scores

## Critical Workflows

### Training VAE
```bash
python Medical_VAE_Video_Clustering.py \
  --latent_dim 32 --beta 2.0 --crop_percent 0.1 \
  --aggregation transformer --epochs 60
```
**Outputs**: `Best_VAE_ld{ld}_crop{crop}_beta{beta}_{annealing}.pth`

### Training Classifier
```bash
python latent_classifier.py \
  --vae-checkpoint Best_VAE_ld32_crop10_beta2.0_cyclical.pth \
  --latent-dim 32 --aggregation mean \
  --use-class-weights  # For imbalanced data
```
**Outputs**: `results/classifier/best_classifier.pth`, confusion matrix, training curves

### Interactive Cluster Analysis
```bash
python cluster_analysis.py \
  --cluster-table wandb/.../*.table.json \
  --output results/interactive_tsne.html
```

## Project-Specific Conventions

### 1. Module-Level Argparse Protection
**Problem**: `Medical_VAE_Video_Clustering.py` runs argparse at module level, breaking imports.
**Solution**: Always wrap in `parse_args()` function called only in `if __name__ == "__main__"`:
```python
def parse_args():
    parser = argparse.ArgumentParser(...)
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        return parser.parse_args([])  # Jupyter/IPython
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    LATENT_DIM = args.latent_dim  # Update globals
```

### 2. Video ID Handling (`utils.py::find_score()`)
Match metadata to frames via **multiple fallback strategies**:
1. Direct match: `video_id` in metadata
2. Strip hospital prefix: `JCUH_27_LU_4_RPB` → `27_LU_4_RPB`
3. Extract from path: `image_path.split('_selected_frame')[0]`

### 3. Channel Count Consistency
**CRITICAL**: VAE trained on grayscale (CHANNELS=1), but dataset loads RGB then converts:
```python
# WRONG - causes shape mismatch
image = Image.open(path).convert('RGB')

# CORRECT - load grayscale directly
image = Image.open(path).convert('L')
```

### 4. Checkpoint Naming Convention
Pattern: `{ModelName}_ld{latent_dim}_crop{int(crop_percent*100)}_beta{beta}_{annealing}.pth`
Example: `Best_VAE_ld32_crop10_beta2.0_cyclical.pth`

### 5. WandB Offline Mode
```python
os.environ["WANDB_MODE"] = "offline"  # All scripts use offline logging
```
Upload later: `python upload_to_wandb.py`

## File Organization
```
code/
├── config.py              # Centralized paths, COSMA-specific defaults
├── utils.py               # Shared: load_cluster_data(), find_score()
├── Medical_VAE_Video_Clustering.py  # Main VAE training
├── latent_classifier.py   # Score prediction from latent space
├── cluster_analysis.py    # Interactive t-SNE with Plotly
├── data_preprocessing/    # Jupyter notebooks for data cleaning
│   └── data_tables/       # all_data.csv (metadata)
└── results/               # Generated plots, models, embeddings
```

## Common Pitfalls
1. **Import conflicts**: Don't import from `Medical_VAE_Video_Clustering.py` without checking argparse protection
2. **GPU memory**: Use `batch_size=16` for video data (10 frames × batch)
3. **Score distribution**: Heavily imbalanced (33% score 0, 44% score 1, 22% score 2, <1% score 3) → use `--use-class-weights`
4. **Transformer requires checkpoint**: `--aggregation transformer` needs `--transformer-checkpoint` argument

## Testing New Features
1. Check `config.py` for paths/defaults before hardcoding
2. Use `utils.find_score()` for metadata lookups (handles all edge cases)
3. Test with small subset: `--epochs 5 --batch-size 8`
4. Validate shapes: VAE expects `(batch, 1, 64, 64)` not `(batch, 3, 64, 64)`
