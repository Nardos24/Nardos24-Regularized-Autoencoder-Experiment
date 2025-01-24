import torch
import os
from tqdm import tqdm
from src.model import RegularizedAutoencoder
from src.data_loader import get_data_loaders
from src.utils import binary_cross_entropy

def train_model(config):
    """
    Train the Regularized Autoencoder (RAE) model.

    Args:
        config: Configuration dictionary containing training parameters.
    """
    # Load configurations
    device = torch.device(config["device"])
    latent_dim = config["latent_dim"]
    lambda_reg = config["lambda_reg"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    model_path = config["model_path"]
    init_std = config.get("init_std", 0.1)  # Get standard deviation from config

    # Initialize model, optimizer, and data loaders
    model = RegularizedAutoencoder(latent_dim, std=init_std).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loader, val_loader, _ = get_data_loaders(config)

    # Create directory for saving models
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            data = data.to(device).view(data.size(0), -1)  # Flatten input
            optimizer.zero_grad()

            # Forward pass
            recon_data = model(data)

            # Flatten reconstruction for BCE calculation
            recon_data_flat = recon_data.view(data.size(0), -1)
            data_flat = data.view(data.size(0), -1)

            # Compute reconstruction loss
            loss = binary_cross_entropy(data_flat, recon_data_flat)
            

            # Add L2 regularization for decoder weights
            l2_reg = sum(torch.sum(param ** 2) for param in model.decoder.parameters())
            loss += lambda_reg * l2_reg

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item() * data.size(0)

       
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Train BCE Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), model_path)

    print(f"Training complete. Model saved to {model_path}")
