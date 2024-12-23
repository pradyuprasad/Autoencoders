# MNIST Autoencoder

A simple autoencoder implementation that compresses MNIST digits into a 3D latent space with visualization of the encoded representations and reconstruction quality metrics.

## Overview

This project implements a basic autoencoder architecture that:
- Compresses 28x28 MNIST digits into a 3-dimensional latent space
- Reconstructs the original images from these 3D representations
- Provides visualization tools for the latent space
- Measures reconstruction quality using SSIM and MSE metrics

## Structure

- `autoencoder.py`: Core autoencoder architecture implementation
- `train_autoencoder.py`: Training loop and data loading utilities
- `eval_model.py`: Model evaluation and metrics calculation
- `display_image.py`: Utilities for visualizing MNIST digits

## Results

| Model Version    | Parameters | SSIM   | MSE    | Notes                  |
|-----------------|------------|--------|---------|------------------------|
| Original        | 209,411    | 0.7192 | 0.0309 | First implementation  |
| Large (10x)     | 2,218,243  | 0.7544 | 0.0277 | 10x larger model      |


## How to run this
1. Install uv following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)
2. Run `uv sync` to set your dev environment
3. Run `uv run train_autoencoder.py` to train the model
4. Run `uv run eval_model.py` to evaluate the results
5. Run `uv run interpolation.py` to understand how well the model interpolates between a 5 and a 9
