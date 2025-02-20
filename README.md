# Regularized Autoencoder (RAE) Experiment

This repository contains the implementation of the **Regularized Autoencoder (RAE)** for various tasks such as reconstruction, pattern completion, classification, and density modeling, based on the description and methodology outlined in the referenced paper.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Features](#features)
  - [Tasks Implemented](#tasks-implemented)
  - [Expected Results](#expected-results)
- [File Structure](#file-structure)
- [References](#references)

## Overview

The Regularized Autoencoder (RAE) is a feedforward neural network designed for unsupervised learning. It consists of an encoder that maps input data to a latent representation and a decoder that reconstructs the input from the latent space. The RAE optimizes the following data log-likelihood:

\[ \psi = \sum_j \big(x[j] \log z_0[j] + (1 - x[j]) \log(1 - z_0[j])\big) \]

The key contributions of this implementation include:
- Incorporating **L2 regularization** on decoder weights.
- Using **Monte Carlo sampling** for estimating the marginal log-likelihood.
- Tasks such as **reconstruction**, **pattern completion**, **classification**, and **density modeling**.

## Setup

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- tqdm

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nardos24/Regularized-Autoencoder-Experiment
   cd Regularized-Autoencoder-Experiment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset. Ensure the MNIST dataset is saved as `.npy` files under the `data/` directory:
   ```
   data/
   ├── trainX.npy
   ├── trainY.npy
   ├── validX.npy
   ├── validY.npy
   ├── testX.npy
   ├── testY.npy
   ```

## Configuration

All configurable parameters are stored in `experiments/config.yaml`:

```yaml
# Device configuration
device: "cpu"  # Set to "cuda" for GPU

# Model parameters
latent_dim: 20  # Latent space dimensionality
lambda_reg: 0.00001  # L2 regularization coefficient
init_std: 0.1  # Standard deviation for weight initialization

# Training parameters
batch_size: 200
learning_rate: 0.1
epochs: 50

# File paths
model_path: "models/rae_model.pth"
log_dir: "logs/"

# Dataset paths
trainX: "data/trainX.npy"
trainY: "data/trainY.npy"
validX: "data/validX.npy"
validY: "data/validY.npy"
testX: "data/testX.npy"
testY: "data/testY.npy"

# Evaluation parameters
n_gmm_components: 75
n_mc_samples: 5000
chunk_size: 1000
```

## Usage

### Training
To train the RAE model, run:
```bash
python src/main.py --task train --config experiments/config.yaml
```

### Evaluation
To evaluate the model on all tasks (reconstruction, classification, pattern completion, and density modeling), run:
```bash
python src/main.py --task evaluate --config experiments/config.yaml
```

## Features

### Tasks Implemented

1. **Reconstruction**: Measures the model's ability to reconstruct the input using Binary Cross-Entropy (BCE) loss.
2. **Pattern Completion**: Evaluates the model's performance on completing masked parts of the input using Masked Mean Squared Error (M-MSE).
3. **Classification**: Uses the latent representations for a downstream classification task via logistic regression.
4. **Density Modeling**: Estimates the marginal log-likelihood using Monte Carlo sampling with a Gaussian Mixture Model (GMM).

### Expected Results

The results may vary slightly depending on the training process, but according to the referenced paper, the expected outcomes are:

| Metric                       | Expected Value         |
|------------------------------|------------------------|
| Reconstruction Loss (BCE)    | ~60                   |
| Pattern Completion (M-MSE)   | ~22.10 ± 1.44         |
| Classification Error          | ~11%                  |
| Marginal Log-Likelihood       | ~-117.01 ± 0.18       |

Our experiment results were as follows:

| Metric                       | Observed Value         |
|------------------------------|------------------------|
| Validation Loss (BCE)        | 80.1370               |
| Test Loss (BCE)              | 79.5107               |
| Classification Error          | 11.15%               |
| Pattern Completion (M-MSE)   | 18.22                 |
| Marginal Log-Likelihood       | -387.5964             |


## References

If you find this repository useful, please consider citing the original paper:

- [Synaptic Learning as Error-Correction in Biological and Artificial Neural Networks](https://www.nature.com/articles/s41467-022-29632-7)
