# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, FashionMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
import glob
from PIL import Image
from dotenv import load_dotenv
import wandb
import os
import scipy.stats as stats
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_MODE"] = "offline"

# wandb sync wandb/offline-run-<timestamp> to sync runs when back online

# %%
# ==========================================
# 1. CONFIGURATION
# ==========================================
# Image settings
IMAGE_SIZE = 64
CHANNELS = 1  # Use 1 for grayscale (X-ray/MRI), 3 for RGB
LATENT_DIM = 32 # Size of the compressed feature vector

# Training settings
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
BETA = 5
# cude, mps, cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE

# %%
# ==========================================
# 2. VAE MODEL ARCHITECTURE
# ==========================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        
        # ENCODER: Compresses image -> hidden representation
        self.encoder = nn.Sequential(
            # Input: (B, C, 64, 64)
            nn.Conv2d(CHANNELS, 32, kernel_size=4, stride=2, padding=1), # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (256, 4, 4)
            nn.ReLU(),
        )
        
        # LATENT SPACE: Mean and Log-Variance
        # Flatten size: 256 channels * 4 * 4 pixels = 4096
        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # DECODER: Expands hidden representation -> reconstructed image
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, CHANNELS, kernel_size=4, stride=2, padding=1), # -> (C, 64, 64)
            nn.Sigmoid() # Pixel values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        """
        The "Magic" of VAE: Adds noise during training to force continuity,
        but allows backpropagation to work.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        # Latent sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z_input = self.decoder_input(z)
        z_matrix = z_input.view(z_input.size(0), 256, 4, 4)
        reconstruction = self.decoder(z_matrix)
        
        return reconstruction, mu, logvar

# ==========================================
# 3. LOSS FUNCTION
# ==========================================

class LossTracker:
    def __init__(self):
        self.history = {"loss": [], "reconstruction_loss": [], "kl_loss": []}

    def add(self, total, recon, kl):
        self.history["loss"].append(total)
        self.history["reconstruction_loss"].append(recon)
        self.history["kl_loss"].append(kl)

def vae_loss_function(recon_x, x, mu, logvar, beta=10):
    """
    Loss = Reconstruction Error + KL Divergence
    """
    # # 1. How well did we reconstruct the image? (Binary Cross Entropy is good for normalized images)
    # BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # # 2. How far is our latent distribution from a standard normal distribution?
    # # This acts as a regularizer so the latent space doesn't "explode"
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    
    # possible kl fix normalisation
    batch_size = x.size(0)
    # BCE: sum over pixels per image, then mean over batch
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size

    # KLD: sum over latent dims per image, then mean over batch
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = BCE + beta*KLD
    return total_loss, BCE, KLD

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
class FlatImageDataset(Dataset):
    """Dataset for images in a flat directory (no class subfolders)"""
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Return dummy label
    
class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder"""
    def __getitem__(self, index):
        # Original ImageFolder returns (image, class_index)
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        image = original_tuple[0]
        
        # Retrieve the path from the internal list of samples
        # self.samples contains tuples of (path, class_index)
        image_path = self.samples[index][0]
        
        return image, image_path

def get_dataloader(use_real_data=True, data_path='/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae/all_images', batch_size=BATCH_SIZE):
    """
    Returns the dataloader.
    Toggle 'use_real_data' to True and provide 'data_path' for your medical images.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda img: transforms.functional.crop(img, top=int(img.size[1] * 0.25), left=0, height=int(img.size[1] * 0.75), width=img.size[0])), 
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=CHANNELS),
        transforms.ToTensor(),
    ])

    if use_real_data and data_path:
        print(f"Loading real data from {data_path}...")
        try:
            # CHANGED: Use the custom class that returns paths
            dataset = ImageFolderWithPaths(root=data_path, transform=transform)
        except (FileNotFoundError, RuntimeError):
            print("ImageFolder failed - trying flat directory structure...")
            dataset = FlatImageDataset(root_dir=data_path, transform=transform)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset

def train_vae(model, dataloader, epochs, beta, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    tracker = LossTracker()
    num_samples = len(dataloader.dataset)
    
    for epoch in range(epochs):
        total_loss = 0
        total_bce = 0
        total_kld = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        avg_loss = total_loss / num_samples
        avg_bce = total_bce / num_samples
        avg_kld = total_kld / num_samples
        tracker.add(avg_loss, avg_bce, avg_kld)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.2f} | Recon Loss: {avg_bce:.2f} | KL Loss: {avg_kld:.2f}")
            
    return tracker

def extract_latent_features(model, dataloader):
    """
    Passes all data through encoder to get the latent vectors (mu).
    """
    model.eval()
    latent_vectors = []
    image_paths = [] # Kept just for verification, not used in clustering
    
    with torch.no_grad():
        for data, paths in dataloader:
            data = data.to(DEVICE)
            _, mu, _ = model(data)
            latent_vectors.append(mu.cpu().numpy())
            image_paths.extend(paths)
            
    return np.concatenate(latent_vectors), image_paths

# log loss graphs of total, reconstruction, kl divergence
def log_loss_graphs(tracker, latent_dim):
    """
    Logs loss curves to WandB.
    """
    epochs = range(1, len(tracker.history["loss"]) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, tracker.history["loss"], label="Total Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Total Loss (Latent Dim={latent_dim})")
    plt.grid(True)
    
    # Reconstruction Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, tracker.history["reconstruction_loss"], label="Reconstruction Loss", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Reconstruction Loss (Latent Dim={latent_dim})")
    plt.grid(True)
    
    # KL Divergence
    plt.subplot(1, 3, 3)
    plt.plot(epochs, tracker.history["kl_loss"], label="KL Divergence", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"KL Divergence (Latent Dim={latent_dim})")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save and log to WandB
    os.makedirs("results/loss_plots", exist_ok=True)
    plot_path = f"results/loss_plots/loss_curves_ld{latent_dim}.png"
    plt.savefig(plot_path)
    plt.close()
    
    wandb.log({
        "plots/loss_curves": wandb.Image(plot_path),
        "latent_dim": latent_dim
    })



def plot_latent_distributions(latent_vectors, latent_dim, epoch=None):
    """
    Plots the distribution of each latent feature dimension.
    latent_vectors: shape (N_samples, latent_dim)
    """
    # Calculate grid size (approx square)
    n_cols = 6  
    n_rows = (latent_dim + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(latent_dim):
        ax = axes[i]
        feature_data = latent_vectors[:, i]
        
        # Calculate stats
        mu = np.mean(feature_data)
        std = np.std(feature_data)
        
        # Plot Histogram
        # 'density=True' normalizes it so we can overlay the Gaussian
        count, bins, ignored = ax.hist(feature_data, bins=30, density=True, 
                                     alpha=0.6, color='gray', edgecolor='none')
        
        # Overlay Gaussian Curve
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=1)
        
        # Formatting
        ax.set_title(f"Feat {i+1}\n({mu:.3f}, {std:.3f})", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Optional: Check for collapse (if std is very close to 1 and mean close to 0)
        if 0.9 < std < 1.1 and -0.1 < mu < 0.1:
            ax.set_facecolor('#eaffea') # Light green for "standard normal" (potentially unused)
    
    # Hide unused subplots
    for j in range(latent_dim, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    # Save and Log
    os.makedirs("results/latent_dist", exist_ok=True)
    filename = f"results/latent_dist/dist_ld{latent_dim}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Latent distribution plot saved to {filename}")
    
    # If using wandb
    if wandb.run is not None:
        wandb.log({f"plots/latent_distributions_ld{latent_dim}": wandb.Image(filename)})


# clustering function

def perform_clustering_and_log(model, dataloader, latent_dim):
        """
        Runs feature extraction, KMeans, t-SNE, and logs plots to WandB.
        """
        print(f"Generating plots for Latent Dim {latent_dim}...")
        
        # 1. Extract Features
        X_latent, image_paths = extract_latent_features(model, dataloader)
        
        # 2. KMeans Clustering
        N_CLUSTERS = 4
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_latent)

        # 3. t-SNE Visualization
        tsne = TSNE(n_components=2, random_state=42)
        # Limit to 1000 points for speed, or remove slicing for full dataset
        limit = min(len(X_latent), 1000) 
        X_embedded = tsne.fit_transform(X_latent[:limit])
        clusters_plot = clusters[:limit]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_plot, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f"Clustering in Latent Space (LD={latent_dim})")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.tight_layout()
        
        # Save locally first
        os.makedirs("results/plots", exist_ok=True)
        plot_path = f"results/plots/cluster_ld{latent_dim}.png"
        plt.savefig(plot_path)
        plt.close()

        table_data = list(zip(
            image_paths[:limit], 
            clusters[:limit], 
            X_embedded[:, 0], 
            X_embedded[:, 1]
        ))


        # 4. Log to WandB
        # This works in offline mode too (it queues the image file for later sync)
        wandb.log({
            "plots/clustering": wandb.Image(plot_path),
            "cluster_labels": wandb.Table(
                    data=table_data, 
                    columns=["image_path", "cluster_label", "tsne_x", "tsne_y"]
                )
        })
        
        print("Clustering plots logged.")

def log_reconstruction(model, dataloader, latent_dim):
   
    print(f"Generating reconstruction plots for Latent Dim {latent_dim}...")
    
    model.eval()
    with torch.no_grad():
        # Get a single batch of data
        sample_data, _ = next(iter(dataloader))
        
        # Take just the first 8 images
        sample_data = sample_data.to(DEVICE)[:8]
        
        # Pass through VAE
        recon, _, _ = model(sample_data)
        
        # Create a grid: Top row = Original, Bottom row = Reconstructed
        comparison = torch.cat([sample_data, recon])
        
        # Make grid expects (B, C, H, W)
        grid = torchvision.utils.make_grid(comparison.cpu(), nrow=8)
        
        plt.figure(figsize=(15, 5))
        # permute needed because matplotlib expects (H, W, C) but torch gives (C, H, W)
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f"Top: Original | Bottom: Reconstructed (LD={latent_dim})")
        plt.axis('off')
        
        # Save locally first (crucial for offline mode)
        os.makedirs("results/reconstructions", exist_ok=True)
        save_path = f"results/reconstructions/recon_ld{latent_dim}.png"
        plt.savefig(save_path)
        plt.close() # Close memory
        
        # Log to WandB
        wandb.log({
            "plots/reconstruction": wandb.Image(save_path),
            "latent_dim": latent_dim
        })
        print("Reconstruction plots logged.")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize Weights & Biases
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY:
        try:
            wandb.login(key=WANDB_API_KEY)
        except Exception as e:
            print(f"W&B login failed: {e}")
    else:
        print("WANDB_API_KEY not found in environment. Proceeding without explicit login.")

    print(f"Running on device: {DEVICE}")
    
    # --- STEP 1: DATA ---
    # TO USE YOUR DATA: Set use_real_data=True and provide path
    # Folder structure must be: /path/to/data/images_subfolder/img1.png
    DATA_PATH = os.getenv("DATA_PATH", "/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae/all_images")
    dataloader, dataset = get_dataloader(use_real_data=True, data_path=DATA_PATH)
    
    # --- STEP 2: TRAIN MODEL ---

    latent_dims = [64] 
    summary = []

    for ld in latent_dims:
        print(f"\nTraining VAE with LATENT_DIM={ld}...")
        
        # RE-INIT WANDB FOR EACH LATENT DIM
        run = wandb.init(
            project="lus-medical-vae",
            group="manual_latent_sweep",
            config={
                "image_size": IMAGE_SIZE,
                "channels": CHANNELS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "beta": BETA,
                "latent_dim": ld,
                "device": str(DEVICE),
            },
            reinit=True
        )

        vae = ConvVAE(latent_dim=ld).to(DEVICE)
        # Use config values
        current_beta = wandb.config.beta
        current_lr = wandb.config.learning_rate
        tracker = train_vae(vae, dataloader, epochs=EPOCHS, beta=current_beta, learning_rate=current_lr)
        # Log per-epoch losses
        for epoch_idx, (loss, recon, kl) in enumerate(zip(tracker.history["loss"], tracker.history["reconstruction_loss"], tracker.history["kl_loss"])):
            wandb.log({
                "epoch": epoch_idx + 1,
                "loss/total": loss,
                "loss/reconstruction": recon,
                "loss/kl": kl,
                "latent_dim": ld,
            })

    # Save loss arrays for this latent dim
        out_dir = os.path.join("results", "loss_sweep")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"loss_total_ld{ld}.npy"), np.array(tracker.history["loss"]))
        np.save(os.path.join(out_dir, f"loss_recon_ld{ld}.npy"), np.array(tracker.history["reconstruction_loss"]))
        np.save(os.path.join(out_dir, f"loss_kl_ld{ld}.npy"), np.array(tracker.history["kl_loss"]))

       
    # cluster and log
        perform_clustering_and_log(vae, dataloader, ld)
        log_reconstruction(vae, dataloader, ld)
        log_loss_graphs(tracker, ld)

        # Keep last-epoch metrics for a compact summary
        summary.append({
            "latent_dim": ld,
            "final_loss": tracker.history["loss"][-1],
            "final_recon": tracker.history["reconstruction_loss"][-1],
            "final_kl": tracker.history["kl_loss"][-1],
        })

        
    # Summary plot across latent dims (final epoch)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    
    xs = [s["latent_dim"] for s in summary]
    
    # Left plot: Total and Reconstruction loss
    ax1.plot(xs, [s["final_loss"] for s in summary], "o-", label="Total Loss", linewidth=2, markersize=8)
    ax1.set_xlabel("LATENT_DIM")
    ax1.set_ylabel("Final Epoch Loss")
    ax1.set_title("Total Loss vs Latent Dimension")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(xs, [s["final_recon"] for s in summary], "s-", label="Reconstruction Loss", linewidth=2, markersize=8)
    ax2.set_xlabel("LATENT_DIM")
    ax2.set_ylabel("Final Epoch Loss")
    ax2.set_title("Reconstruction Loss vs Latent Dimension")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    
    # Right plot: KL Divergence on separate scale
    ax3.plot(xs, [s["final_kl"] for s in summary], "^-", color='red', label="KL Divergence", linewidth=2, markersize=8)
    ax3.set_xlabel("LATENT_DIM")
    ax3.set_ylabel("KL Divergence")
    ax3.set_title("KL Divergence vs Latent Dimension")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    final_summary_path = "results/latent_dim_vs_loss.png"
    plt.savefig(final_summary_path)
    plt.close()
    # wandb.log({"plots/latent_dim_vs_loss": wandb.Image(final_summary_path)})
    # wandb_run.finish()

    # %%
    # --- STEP 3: UNSUPERVISED CLUSTERING ---
    # print("\nExtracting features for clustering...")
    # # We extract the 'mu' vector which represents the image content
    # X_latent, y_true = extract_latent_features(vae, dataloader)
    
    # print("Running K-Means Clustering...")
    # # We assume 10 clusters (classes) for FashionMNIST. 
    # # CHANGE THIS to 2 (e.g., Healthy vs Sick) for your medical data
    # N_CLUSTERS = 4 
    # kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    # clusters = kmeans.fit_predict(X_latent)

    # cluster_centers = []
    # for k in range(N_CLUSTERS):
    # # Select all latent vectors belonging to cluster k
    #     cluster_points = X_latent[clusters == k]
    
    #     # Calculate the mean (centroid) of these vectors
    #     centroid = np.mean(cluster_points, axis=0)
        
    #     cluster_centers.append(centroid)

    # # Convert the list of centers back to a numpy array
    # cluster_centers = np.array(cluster_centers)

    # # put cluster centre latent vectors into decoder to visualise typical images
    # vae.eval()
    # with torch.no_grad():
    #     cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32).to(DEVICE)
    #     z_input = vae.decoder_input(cluster_centers_tensor)
    #     z_matrix = z_input.view(z_input.size(0), 256, 4, 4)
    #     reconstructions = vae.decoder(z_matrix)

    # Plot the cluster center reconstructions
    # plt.figure(figsize=(12, 6))
    # for i in range(N_CLUSTERS):
    #     plt.subplot(1, N_CLUSTERS, i + 1)
    #     img = reconstructions[i].cpu().squeeze().numpy()
    #     plt.imshow(img, cmap='gray')
    #     plt.title(f"Cluster {i} Center")
    #     plt.axis('off')
    # plt.suptitle("Cluster Center Reconstructions")
    # plt.show()
    # # save into results
    # os.makedirs("results/cluster_centers", exist_ok=True)
    # for i in range(N_CLUSTERS):
    #     img = reconstructions[i].cpu().squeeze().numpy()
    #     plt.imsave(f"results/cluster_centers/cluster_{i}_center.png", img, cmap='gray')
        
    
# %%
    
    # # --- STEP 4: VISUALIZATION ---
    # print("Visualizing Latent Space with t-SNE (this might take a moment)...")
    # # Reduce 32-dim latent space to 2-dim for plotting
    # tsne = TSNE(n_components=2, random_state=42)
    # X_embedded = tsne.fit_transform(X_latent[:1000]) # Only plot first 1000 for speed
    # clusters_plot = clusters[:1000]
    
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_plot, cmap='tab10', alpha=0.6)
    # plt.colorbar(scatter, label='Cluster ID')
    # plt.title("Unsupervised Clustering of Medical Images (Latent Space)")
    # plt.xlabel("t-SNE Dim 1")
    # plt.ylabel("t-SNE Dim 2")
    # plt.show()

    # # --- STEP 5: SHOW RECONSTRUCTION ---
    # # Verify the VAE actually learned shapes
    # vae.eval()
    # with torch.no_grad():
    #     sample_data, _ = next(iter(dataloader))
    #     sample_data = sample_data.to(DEVICE)[:8]
    #     recon, _, _ = vae(sample_data)
        
    #     # Create a grid: Top row original, Bottom row reconstruction
    #     comparison = torch.cat([sample_data, recon])
    #     grid = torchvision.utils.make_grid(comparison.cpu(), nrow=8)
        
    #     plt.figure(figsize=(15, 5))
    #     plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    #     plt.title("Top: Original | Bottom: Reconstructed")
    #     plt.axis('off')
    #     plt.show()

    # print("Done! The variable 'clusters' now contains your unsupervised labels.")


# %%

