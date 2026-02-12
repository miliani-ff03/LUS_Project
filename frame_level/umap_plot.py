# plot latent features using UMAP
import matplotlib.pyplot as plt
import umap
import numpy as np



latent_vectors = "/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/latent_features/latent_vectors_ld32_crop10_beta2.0_gamma1.0.npy"
image_paths = "/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/latent_features/image_paths_ld32_crop10_beta2.0_gamma1.0.npy"
scores = "/cosma/home/durham/dc-fras4/code/frame_level/results/supervised/latent_features/scores_ld32_crop10_beta2.0_gamma1.0.npy"


def plot_umap(latent_vectors, image_paths, scores, n_neighbors=15, min_dist=0.1, metric='euclidean'):

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42) 
    embedding = reducer.fit_transform(latent_vectors) 
    plt.scatter(embedding[:, 0], embedding[:, 1], c=scores, cmap='viridis', s=10) 
    plt.colorbar(label='Score') 
    plt.title('UMAP Projection of Latent Features') 
    plt.xlabel('UMAP Dimension 1') 
    plt.ylabel('UMAP Dimension 2') 
    plt.grid(True) 
    plt.tight_layout() 
    plt.show()  

if __name__ == "__main__":
    latent_vectors = np.load(latent_vectors) 
    image_paths = np.load(image_paths, allow_pickle=True) 
    scores = np.load(scores) 
    plot_umap(latent_vectors, image_paths, scores)