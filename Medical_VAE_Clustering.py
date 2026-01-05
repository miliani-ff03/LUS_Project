# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
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
import argparse
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_MODE"] = "offline"

# wandb sync wandb/offline-run-<timestamp> to sync runs when back online

# %%
# ==========================================
# 1. CONFIGURATION
# ==========================================
parser = argparse.ArgumentParser(description="VAE Training with Crop Sweep")

parser.add_argument("--crop_percent", type=float, default=0.25, help="Percentage to crop from top (0.0 to 1.0)")
parser.add_argument("--beta", type=float, default=5.0, help="Beta parameter for KL Divergence")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension size")

if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
    # Use default values in notebook
    args = parser.parse_args([])
else:
    # Parse command line args normally
    args = parser.parse_args()

# Image settings
IMAGE_SIZE = 64
CHANNELS = 1  # Use 1 for grayscale (X-ray/MRI), 3 for RGB
LATENT_DIM = args.latent_dim # Size of the compressed feature vector

# Training settings
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = args.epochs
BETA = args.beta
CROP_PERCENT = args.crop_percent  # e.g., 0.25 means crop top 25% of image
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
class KLAnnealer:
    """Linearly anneals beta from start_beta to end_beta over warmup_epochs."""
    def __init__(self, total_epochs, start_beta=0.0, end_beta=1.0):
        self.total_epochs = total_epochs
        self.warmup_epochs = max(1, total_epochs // 10) 
        self.start_beta = start_beta
        self.end_beta = end_beta
    
    def get_beta(self, epoch):
        if epoch < self.warmup_epochs:
            return self.start_beta + (self.end_beta - self.start_beta) * (epoch / self.warmup_epochs)
        else:
            return self.end_beta
        
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, delta=0.01, path='best_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        
        if self.best_score is None or val_loss< self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
        
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class LossTracker:
    def __init__(self):
        self.history = {"loss": [], "reconstruction_loss": [], "kl_loss": [], "validation_loss": []}

    def add(self, total, recon, kl, val):
        self.history["loss"].append(total)
        self.history["reconstruction_loss"].append(recon)
        self.history["kl_loss"].append(kl)
        self.history["validation_loss"].append(val)

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
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

def get_dataloader(use_real_data=True, data_path='/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae/all_images', batch_size=BATCH_SIZE, crop_percent=0.25, val_split=0.2):
    """
    Returns the dataloader.
    Toggle 'use_real_data' to True and provide 'data_path' for your medical images.
    """
    transform_ops = [ transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)) ]

    if crop_percent > 0:
        transform_ops.append(
            transforms.Lambda(lambda img: transforms.functional.crop(
                img, 
                top=int(img.size[1] * crop_percent), 
                left=0, 
                height=int(img.size[1] * (1 - crop_percent)), 
                width=img.size[0]
            ))
        )
    transform_ops.extend([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.Grayscale(num_output_channels=CHANNELS), transforms.ToTensor()])
    
    transform = transforms.Compose(
        transform_ops)

    if use_real_data and data_path:
        print(f"Loading real data from {data_path} with crop_percent={crop_percent}")
        try:
            # CHANGED: Use the custom class that returns paths
            full_dataset = ImageFolderWithPaths(root=data_path, transform=transform)
        except (FileNotFoundError, RuntimeError):
            print("ImageFolder failed - trying flat directory structure...")
            full_dataset = FlatImageDataset(root_dir=data_path, transform=transform)
        
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Data Split: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader

def train_vae(model, train_loader, val_loader, epochs, end_beta, learning_rate, save_path="Best_VAE.pth", patience=10):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tracker = LossTracker()
    kl_annealer = KLAnnealer(total_epochs=epochs, start_beta=0.0, end_beta=end_beta)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0

        current_beta = kl_annealer.get_beta(epoch)

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=current_beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(DEVICE)
                recon_val, mu_val, logvar_val = model(val_data)
                v_loss, _, _ = vae_loss_function(recon_val, val_data, mu_val, logvar_val, beta=current_beta)
                val_loss += v_loss.item()

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        tracker.add(avg_train_loss, avg_bce, avg_kld, avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f} | Beta: {current_beta:.4f}")
        
        # --- EARLY STOPPING CHECK ---
        if epoch >= kl_annealer.warmup_epochs:
            early_stopping(avg_val_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        else:
            if avg_val_loss < early_stopping.val_loss_min - early_stopping.delta:
                early_stopping.save_checkpoint(avg_val_loss, model)

    # Load the best model (saved by EarlyStopping)
    print(f"Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path))

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
def log_loss_graphs(tracker, latent_dim, beta, crop_suffix=""):
    """
    Logs loss curves to WandB.
    """
    epochs = range(1, len(tracker.history["loss"]) + 1)
    
    fig = plt.figure(figsize=(12, 4))
    
    # Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, tracker.history["loss"], label="Total Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Total Loss (ld={latent_dim}, beta={beta}, crop={crop_suffix})")
    plt.grid(True)
    
    # Reconstruction Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, tracker.history["reconstruction_loss"], label="Reconstruction Loss", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Reconstruction Loss (ld={latent_dim}, beta={beta}, crop={crop_suffix})")
    plt.grid(True)
    
    # KL Divergence
    plt.subplot(1, 3, 3)
    plt.plot(epochs, tracker.history["kl_loss"], label="KL Divergence", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"KL Divergence (ld={latent_dim}, beta={beta}, crop={crop_suffix})")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save and log to WandB
    # os.makedirs("results/loss_plots", exist_ok=True)
    # plot_path = f"results/loss_plots/loss_curves_ld{latent_dim}.png"
    # plt.savefig(plot_path)
    # plt.close()
    if wandb.run is not None:
        wandb.log({
            "plots/loss_curves": wandb.Image(fig),
            "latent_dim": latent_dim,
            "beta": beta,
            "crop_percent": crop_suffix
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
    # os.makedirs("results/latent_dist", exist_ok=True)
    # filename = f"results/latent_dist/dist_ld{latent_dim}.png"
    # plt.savefig(filename)
    # plt.close()
    
    # print(f"Latent distribution plot saved to {filename}")
    
    # If using wandb
    if wandb.run is not None:
        wandb.log({f"plots/latent_distributions_ld{latent_dim}": wandb.Image(fig)})

    plt.close(fig)

# clustering function

def perform_clustering_and_log(model, dataloader, latent_dim, beta, crop_suffix=""):
        """
        Runs feature extraction, KMeans, t-SNE, and logs plots to WandB.
        """
        print(f"Generating plots for Latent Dim {latent_dim}...")
        
        # 1. Extract Features
        X_latent, image_paths = extract_latent_features(model, dataloader)
        
        # 2. KMeans Clustering
       
        for N_CLUSTERS in [2,3,4]:
            kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(X_latent)

            # 3. t-SNE Visualization
            tsne = TSNE(n_components=2, random_state=42)
            # Limit to 1000 points for speed, or remove slicing for full dataset
            limit = min(len(X_latent), 1000) 
            X_embedded = tsne.fit_transform(X_latent[:limit])
            clusters_plot = clusters[:limit]
            
            fig = plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_plot, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, label='Cluster ID')
            plt.title(f"Clustering LD={latent_dim}, Beta={beta}, Crop={crop_suffix}, K={N_CLUSTERS}")
            plt.xlabel("t-SNE Dim 1")
            plt.ylabel("t-SNE Dim 2")
            plt.tight_layout()
            
            # Save locally first
            # os.makedirs("results/plots", exist_ok=True)
            # plot_path = f"results/plots/ld{latent_dim}_crop{crop_suffix}_beta{beta}_k{N_CLUSTERS}.png"
            # plt.savefig(plot_path)
            # plt.close()

            table_data = list(zip(
                image_paths[:limit], 
                clusters[:limit], 
                X_embedded[:, 0], 
                X_embedded[:, 1]
            ))


            # 4. Log to WandB
            # This works in offline mode too (it queues the image file for later sync)
            wandb.log({
                f"plots/clustering_k{N_CLUSTERS}": wandb.Image(fig),
                f"cluster_labels_k{N_CLUSTERS}": wandb.Table(
                        data=table_data, 
                        columns=["image_path", "cluster_label", "tsne_x", "tsne_y"]
                    )
            })
            
        plt.close(fig)

def log_reconstruction(model, dataloader, latent_dim, beta, crop_suffix=""):
   
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
        
        fig = plt.figure(figsize=(15, 5))
        # permute needed because matplotlib expects (H, W, C) but torch gives (C, H, W)
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f"Top: Original | Bottom: Reconstructed (LD={latent_dim}, Beta={beta}, Crop={crop_suffix})")
        plt.axis('off')
        
        # Save locally first (crucial for offline mode)
        # os.makedirs("results/reconstructions", exist_ok=True)
        # save_path = f"results/reconstructions/recon_ld{latent_dim}_crop{crop_suffix}_beta{beta}.png"
        # plt.savefig(save_path)
        # plt.close() # Close memory
        
        # Log to WandB
        wandb.log({
            "plots/reconstruction": wandb.Image(fig),
            "latent_dim": latent_dim,
            "beta": beta,
            "crop_suffix": crop_suffix
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
    print(f"Using crop_percent={CROP_PERCENT}, beta={BETA}, epochs={EPOCHS}, latent_dim={LATENT_DIM}")
    # --- STEP 1: DATA ---
    # TO USE YOUR DATA: Set use_real_data=True and provide path
    # Folder structure must be: /path/to/data/images_subfolder/img1.png
    DATA_PATH = os.getenv("DATA_PATH", "/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae/all_images")
    train_loader, val_loader = get_dataloader(use_real_data=True, data_path=DATA_PATH, crop_percent=CROP_PERCENT, val_split=0.2)
    
    # --- STEP 2: TRAIN MODEL ---
    run = wandb.init(
        project="lus-medical-vae",
        group="crop_sweep_offline",
        config={"crop_percent": CROP_PERCENT,
            "end_beta": BETA,
            "epochs": EPOCHS,
            "latent_dim": LATENT_DIM          
        },
        reinit=True
    )

    vae = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
    suffix = f"crop{int(CROP_PERCENT*100)}"
    full_suffix = f"ld{LATENT_DIM}_crop{int(CROP_PERCENT*100)}_beta{BETA}"

    model_save_name = f"Best_VAE_{suffix}.pth"

    tracker = train_vae(vae, train_loader, val_loader, epochs=EPOCHS, end_beta=BETA, learning_rate=LEARNING_RATE)
    
    # Save and log
    perform_clustering_and_log(vae, train_loader, LATENT_DIM, BETA, crop_suffix=suffix)
    log_reconstruction(vae, val_loader, LATENT_DIM, BETA, crop_suffix=suffix)
    log_loss_graphs(tracker, LATENT_DIM, BETA, crop_suffix=suffix)
    print("Extracting features for distribution plot...")
    X_latent, image_paths = extract_latent_features(vae, train_loader)
    plot_latent_distributions(X_latent, LATENT_DIM, epoch=EPOCHS)

    os.makedirs("results/latent_features", exist_ok=True)
    np.save(f"results/latent_features/latent_vectors_{full_suffix}.npy", X_latent)
    np.save(f"results/latent_features/image_paths_{full_suffix}.npy", np.array(image_paths))
    print(f"Saved {X_latent.shape[0]} latent vectors of dimension {X_latent.shape[1]}")

    artifact = wandb.Artifact(
    name=f"latent_vectors_{full_suffix}",
    type="latent_features",
    description=f"Latent vectors for ld={LATENT_DIM}, crop={CROP_PERCENT}, beta={BETA}"
    )
    artifact.add_file(f"results/latent_features/latent_vectors_{full_suffix}.npy")
    artifact.add_file(f"results/latent_features/image_paths_{full_suffix}.npy")
    wandb.log_artifact(artifact)
    
    wandb.finish()
    

