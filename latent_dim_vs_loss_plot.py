from Medical_VAE_Clustering import ConvVAE, get_dataloader, train_vae, vae_loss_function, DEVICE, LossTracker, KLAnnealer, EarlyStopping
import torch
import numpy as np
import matplotlib.pyplot as plt

latent_dims = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


results = {"latent_dims": latent_dims, "final_recon_loss": [], "final_kl_loss": [], "final_val_loss": [], "number_of_epochs": []}

EPOCHS = 100
BETA = 1.0
CROP_PERCENT = 0.0
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

train_loader, val_loader= get_dataloader(use_real_data=True, data_path="/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae", batch_size=BATCH_SIZE, crop_percent=CROP_PERCENT, val_split=0.2)

for ld in latent_dims:
    print(f"\nTraining VAE with latent dimension: {ld}")

    temp_save_path = f"temp_vae_ld_{ld}.pth"
    vae = ConvVAE(latent_dim=ld).to(DEVICE)
    tracker = train_vae(vae, train_loader, val_loader, epochs=EPOCHS, end_beta=BETA, learning_rate=LEARNING_RATE, save_path=temp_save_path, patience=10)


    final_recon = tracker.history["reconstruction_loss"][-1]
    final_kl = tracker.history["kl_loss"][-1]
    final_val = tracker.history["validation_loss"][-1]
    num_epochs = len(tracker.history["validation_loss"])
    
    results["final_recon_loss"].append(final_recon)
    results["final_kl_loss"].append(final_kl)
    results["final_val_loss"].append(final_val)
    results["number_of_epochs"].append(num_epochs)

    print(f"Final Reconstruction Loss: {final_recon:.4f}")
    print(f"Final KL Loss: {final_kl:.4f}")
    print(f"Final Validation Loss: {final_val:.4f}")
    print(f"Number of Epochs Trained: {num_epochs}")

# Plotting results

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Reconstruction Loss vs Latent Dimension
axes[0].plot(results["latent_dims"], results["final_recon_loss"], marker='o')
axes[0].set_title("Reconstruction Loss vs Latent Dimension")
axes[0].set_xlabel("Latent Dimension")
axes[0].set_ylabel("Reconstruction Loss")
axes[0].grid(True)

# KL Loss vs Latent Dimension
axes[1].plot(results["latent_dims"], results["final_kl_loss"], marker='o', color='orange')
axes[1].set_title("KL Loss vs Latent Dimension")
axes[1].set_xlabel("Latent Dimension")
axes[1].set_ylabel("KL Loss")
axes[1].grid(True)

# Validation Loss vs Latent Dimension
axes[2].plot(results["latent_dims"], results["final_val_loss"], marker='o', color='green')
axes[2].set_title("Validation Loss vs Latent Dimension")
axes[2].set_xlabel("Latent Dimension")
axes[2].set_ylabel("Validation Loss")
axes[2].grid(True)  

plt.tight_layout()
plt.savefig("latent_dim_vs_loss_run2.png")
plt.show()

# save numerical results
np.savez("latent_dim_vs_loss_results_run2.npz", **results)

# to run use nohup python latent_dim_vs_loss_plot.py &> log_2 &