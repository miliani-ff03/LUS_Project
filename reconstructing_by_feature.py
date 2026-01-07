from Medical_VAE_Clustering import ConvVAE, DEVICE, LATENT_DIM
import torch
import numpy as np
import matplotlib.pyplot as plt

X_latent = np.load("results/latent_features/latent_vectors_ld32_crop10_beta2.0.npy")
image_paths = np.load("results/latent_features/image_paths_ld32_crop10_beta2.0.npy")

variances = np.var(X_latent, axis=0)  # Shape: (latent_dim,)

# Get indices of top 5 features with highest variance
top_k = 5
top_k_indices = np.argsort(variances)[-top_k:][::-1]
print(f"Top {top_k} latent feature indices with highest variance: {top_k_indices}")


# Load the trained model
vae = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load("Best_VAE_beta_2_crop10.pth", map_location=DEVICE))
vae.eval()

# Select a single sample to reconstruct (e.g., sample 0)
sample_idx = 20
base_z_numpy = X_latent[sample_idx]  # Shape: (latent_dim,)
base_z = torch.tensor(base_z_numpy, dtype=torch.float32).to(DEVICE)

grid_size = 7
sweep_values = np.linspace(-3, 3, grid_size)

fig, axes = plt.subplots(len(top_k_indices), grid_size, figsize=(15, 8))
plt.subplots_adjust(wspace=0.1, hspace=0.5)

for row_idx, feature_idx in enumerate(top_k_indices):
    axes[row_idx, 0].set_ylabel(f"Feature {feature_idx}", rotation=0, size='large', labelpad=40)

    for col_idx, val in enumerate(sweep_values):
        modified_z = base_z.clone()
        modified_z[feature_idx] = val

        with torch.no_grad():
            z_input = vae.decoder_input(modified_z.unsqueeze(0))
            z_matrix = z_input.view(1, 256, 4, 4)
            reconstructed = vae.decoder(z_matrix)

        reconstructed_image = reconstructed.cpu().squeeze().numpy()

        ax = axes[row_idx, col_idx]
        ax.imshow(reconstructed_image, cmap='gray')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(f"{val:.2f}", fontsize=10)
plt.suptitle(f"Latent feature traversal (sample {sample_idx})", fontsize=16)
plt.show()