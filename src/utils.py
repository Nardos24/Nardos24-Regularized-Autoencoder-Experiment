import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import torch
from sklearn.mixture import GaussianMixture
from src.model import RegularizedAutoencoder
from src.data_loader import get_data_loaders


def extract_latent_representations(model, dataloader, device):
    """
    Extract latent representations of data from the encoder.

    Args:
        model: Trained RAE model.
        dataloader: DataLoader providing data to encode.
        device: Device to run the computations on (CPU or GPU).

    Returns:
        latents: Latent representations of the dataset.
    """
    model.eval()
    latents = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device).view(data.size(0), -1)  # Flatten input
            z = model.encoder(data)  # Encode to latent space
            latents.append(z.cpu().numpy())  # Collect latent vectors

    return np.vstack(latents)


def monte_carlo_log_likelihood(gmm, model, dataloader, n_samples=5000, batch_size=1000):
    model.eval()
    device = next(model.parameters()).device

    # Sample latent vectors from GMM
    gmm_samples, _ = gmm.sample(n_samples)
    z_samples = torch.tensor(gmm_samples, dtype=torch.float32, device=device)

    # Calculate log p(z) under GMM
    log_p_z = gmm.score_samples(gmm_samples)  # Shape: [n_samples]

    # Initialize log p(x)
    log_p_x = []

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Processing Data for Log Likelihood"):
            data = data.to(device).view(data.size(0), -1)  # Flatten inputs
            batch_size_data = data.size(0)

            # Initialize log p(x) for this batch
            batch_log_p_x = torch.zeros(batch_size_data, device=device)

            # Process latent samples in chunks
            for start in range(0, n_samples, batch_size):
                current_chunk_size = min(batch_size, n_samples - start)

                # Get a chunk of latent vectors
                chunk_z_samples = z_samples[start:start + current_chunk_size]

                # Decode latent vectors
                recon_x = model.decoder(chunk_z_samples)  # Shape: [current_chunk_size, input_dim]
                recon_x = recon_x.view(current_chunk_size, -1)  # Flatten reconstruction

                # Repeat data for the chunk size
                x_repeated = data.repeat(current_chunk_size, 1)

                # Ensure `x_repeated` matches `recon_x`
                x_repeated = x_repeated[:recon_x.size(0), :]

                # Calculate log p(x|z) using BCE
                chunk_log_p_x_given_z = -torch.nn.functional.binary_cross_entropy(
                    recon_x, x_repeated, reduction='none'
                ).sum(dim=1)  # Shape: [current_chunk_size]

                # Accumulate log p(x|z) for this chunk
                batch_log_p_x += torch.logsumexp(chunk_log_p_x_given_z.view(-1), dim=0)

            # Normalize by the total number of samples
            batch_log_p_x -= torch.log(torch.tensor(n_samples, dtype=torch.float32, device=device))
            log_p_x.extend(batch_log_p_x.cpu().numpy())

    log_p_x = np.array(log_p_x)  # Shape: [n_data]

    # Combine log p(x) with log p(z)
    log_likelihood = np.mean(log_p_x)  # Final marginal log-likelihood
    return log_likelihood


def binary_cross_entropy(X, X_hat, offset=1e-7):
    """
    Compute the Binary Cross-Entropy (BCE) loss.

    Parameters:
    - X: PyTorch tensor of shape (N, D), the actual sensory input (ground truth).
    - X_hat: PyTorch tensor of shape (N, D), the predicted sensory input.

    Returns:
    - BCE: Binary Cross-Entropy loss (scalar).
    """
    # Ensure X_hat values are clipped to avoid log(0)
    X_hat = torch.clamp(X_hat, offset, 1 - offset)

    # Compute BCE loss
    bce_loss = -torch.sum(X * torch.log(X_hat) + (1 - X) * torch.log(1 - X_hat), dim=1)
    return bce_loss.mean()

def evaluate_reconstruction(model, dataloader):
    """
    Evaluate reconstruction error using Binary Cross-Entropy (BCE).

    Args:
        model: Trained RAE model.
        dataloader: DataLoader providing data to reconstruct.

    Returns:
        Average reconstruction loss over the dataset.
    """
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device  # Retrieve device from the model

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device).view(data.size(0), -1)  # Flatten input
            recon_data = model(data)  # Reconstruct data

            # Flatten reconstructed data for BCE calculation
            recon_data_flat = recon_data.view(data.size(0), -1)

            # Compute reconstruction loss
            loss = binary_cross_entropy(data, recon_data_flat)
            total_loss += loss.item() * data.size(0)

    return total_loss / len(dataloader.dataset)  # Average over dataset size

def evaluate_classification(model, loader):
    """
    Evaluate classification error using the trained RAE model and a logistic regression classifier.

    Args:
        model: Trained Regularized Autoencoder (RAE) with an encoder.
        loader: DataLoader providing batches of (data, labels).

    Returns:
        err: Classification error percentage.
    """
    model.eval()
    latents, labels = [], []
    device = next(model.parameters()).device  # Retrieve device from the model

    with torch.no_grad():
        for data, label in loader:
            # Flatten data and move to the appropriate device
            data = data.view(data.size(0), -1).to(device)
            label = label.to(device)

            # Convert one-hot encoded labels to class indices (if necessary)
            if len(label.shape) > 1:  # Check for one-hot encoded labels
                label = torch.argmax(label, dim=1)  # Convert to class indices

            # Collect latent representations and labels
            z = model.encoder(data)  # Extract latent representations
            latents.append(z.cpu().numpy())
            labels.append(label.cpu().numpy())

    # Combine latents and labels into arrays
    latents = np.vstack(latents)  # Shape: [num_samples, latent_dim]
    labels = np.hstack(labels).reshape(-1)  # Shape: [num_samples]

    # Ensure latents and labels have the same number of samples
    if len(latents) != len(labels):
        print(f"Mismatch detected: latents {len(latents)}, labels {len(labels)}")
        min_len = min(len(latents), len(labels))
        latents = latents[:min_len]
        labels = labels[:min_len]

    # Split latent data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(latents, labels, test_size=0.2, random_state=42)

    # Train logistic regression on training split
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate on test split
    predictions = clf.predict(X_test)
    err = 100 * (1 - accuracy_score(y_test, predictions))

    return err


def masked_mse(model, loader):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, _ in loader:
          
            data = data.view(data.size(0), -1)

            mask = torch.ones_like(data, dtype=torch.bool) 
            mask[:, : data.size(1) // 2] = 0  

            masked_data = data * mask.float()  

            reconstructed = model(masked_data).view(data.size(0), -1)

            mse = F.mse_loss(reconstructed[~mask], data[~mask], reduction="sum")
            total_mse += mse.item() * data.size(0)
            total_samples += data.size(0)

    avg_mse = total_mse / (total_samples * data.size(1) // 2)
    return avg_mse
