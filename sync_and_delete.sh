#!/bin/bash

# Sync offline wandb runs
echo "Syncing offline wandb runs..."
wandb sync --sync-all

# Check if sync was successful
if [ $? -eq 0 ]; then
    echo "Sync completed successfully!"
    
    # Delete offline wandb files
    echo "Deleting offline wandb files..."
    find . -path "*/wandb/offline-run-*" -type d
    find . -path "*/wandb/run-*-offline" -type d
    
    # Delete them
    find . -path "*/wandb/offline-run-*" -type d -exec rm -rf {} +
    find . -path "*/wandb/run-*-offline" -type d -exec rm -rf {} +
    echo "Cleanup completed!"
else
    echo "Sync failed. Offline files not deleted."
    exit 1
fi

# to run chmod +x sync_and_delete.sh and then ./sync_and_delete.sh