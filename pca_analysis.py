from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load latent feature vectors
X_latent = np.load("results/latent_features/latent_vectors_ld32_crop0_beta1.0.npy")
image_paths = np.load("results/latent_features/image_paths_ld32_crop0_beta1.0.npy")


# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_latent)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 2)
plt.scatter(principal_components[:, 0], principal_components[:, 1], cmap='viridis', s=5)
plt.colorbar()
plt.title('PCA Visualization of Latent Space')

plt.show()