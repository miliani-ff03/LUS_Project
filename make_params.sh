#!/bin/bash

# Clear existing file
> params.txt

# Define your grid
LATENT_DIMS=(32 64)
BETAS=(1 5 10)
LRS=(0.001 0.0001)

# Nested loops to write every combination to a file
for ld in "${LATENT_DIMS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for lr in "${LRS[@]}"; do
        # Write space-separated values to the file
        echo "$ld $beta $lr" >> params.txt
    done
  done
done

echo "Created params.txt with $(wc -l < params.txt) combinations."