import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mutual_info_score, adjusted_rand_score

from Medical_VAE_Clustering import ConvVAE, DEVICE

# ================= CONFIGURATION =================
LATENT_DIM = 32
CROP_PERCENT = 10 
BETA = 2.0 # Matches your saved models

latent_features_path = "/cosma5/data/durham/dc-fras4/ultrasound/vae_results/latent_features"


# download latent features
def load_model_and_latents(beta: float, LATENT_DIM: int, CROP_PERCENT: int):
    # Format beta to handle floats properly (2.0 -> "2.0", 2 -> "2.0")
    beta_str = f"{beta:.1f}"
    
    latent_vectors_path = os.path.join(
        latent_features_path,
        f"latent_vectors_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta_str}.npy"
    )
    image_paths_path = os.path.join(
        latent_features_path,
        f"image_paths_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta_str}.npy"
    )
    
    # Load latent vectors and image paths
    X_latent = np.load(latent_vectors_path)
    image_paths = np.load(image_paths_path)

    if beta == 0.1:
        model_path = f"Best_VAE_beta_0.1_crop{CROP_PERCENT}.pth"
    else:
        model_path = f"Best_VAE_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta_str}_cyclical.pth"
    
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None, None, None
    
    # Load model
    vae = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(model_path, map_location=DEVICE))
    vae.eval()

    variances = np.var(X_latent, axis=0)
    
    return vae, X_latent, variances

# performe k-means clustering
def perform_kmeans(X_latent: np.ndarray, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_latent)
    return cluster_labels, kmeans.cluster_centers_

# calculate clustering metrics
def calculate_clustering_metrics(X_latent: np.ndarray, cluster_labels: np.ndarray):

    silhouette_avg = silhouette_score(X_latent, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(X_latent, cluster_labels)
    calinski_harabasz_avg = calinski_harabasz_score(X_latent, cluster_labels)

    return silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg    


# Load the model and latents once
print("Loading model and latent features...")
vae, X_latent, variances = load_model_and_latents(beta=BETA, LATENT_DIM=LATENT_DIM, CROP_PERCENT=CROP_PERCENT)

if vae is None:
    print("Failed to load model and latents.")
    exit(1)

# Store results
results = []

# Try clustering for k from 2 to 32
for k in range(2, 33):
    print(f"\nClustering with k={k}")
    
    cluster_labels, cluster_centers = perform_kmeans(X_latent, n_clusters=k)
    silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg = calculate_clustering_metrics(X_latent, cluster_labels)
    
    # Store the results
    results.append({
        'k': k,
        'silhouette_score': silhouette_avg,
        'davies_bouldin_score': davies_bouldin_avg,
        'calinski_harabasz_score': calinski_harabasz_avg
    })
    
    print(f"  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Davies-Bouldin Score: {davies_bouldin_avg:.4f}")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz_avg:.4f}")

# Create DataFrame and save to CSV
df_results = pd.DataFrame(results)
beta_str = f"{BETA:.1f}"
output_filename = f"clustering_metrics_ld{LATENT_DIM}_crop{CROP_PERCENT}_beta{beta_str}.csv"
df_results.to_csv(output_filename, index=False)

print(f"\n{'='*60}")
print(f"Results saved to: {output_filename}")
print(f"{'='*60}")
print("\nSummary:")
print(df_results.to_string(index=False))

    