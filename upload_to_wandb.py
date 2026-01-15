import wandb
import os
from pathlib import Path
# Initialize wandb
wandb.init(
    project="Medical-VAE-Clustering",
    name="archive-results-2026-01-15",
    job_type="archive"
)
# Upload feature traversal plots as artifacts
artifact = wandb.Artifact("feature-traversal-plots", type="results")
artifact.add_dir("results/feature_traversal")
wandb.log_artifact(artifact)
# Upload individual plots
wandb.log({
    "combined_loss": wandb.Image("combined_loss_plot.png"),
    "latent_dim_vs_loss": wandb.Image("latent_dim_vs_loss.png"),
})

wandb.finish()
print("Upload complete! Files can now be deleted locally.")