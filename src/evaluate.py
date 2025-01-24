import torch
from sklearn.mixture import GaussianMixture
from src.data_loader import get_data_loaders
from src.model import RegularizedAutoencoder
from src.utils import (
    masked_mse,
    evaluate_classification,
    monte_carlo_log_likelihood,
    evaluate_reconstruction,
    extract_latent_representations,
)


def evaluate_model(config):
    """
    Evaluate the model on various tasks, including reconstruction, classification,
    pattern completion, and density modeling using Monte Carlo log-likelihood.

    Args:
        config: Configuration dictionary with parameters.
    """
    device = torch.device(config["device"])
    model = RegularizedAutoencoder(config["latent_dim"], config["std"]).to(device)
    model.load_state_dict(torch.load(config["model_path"]))

    _, val_loader, test_loader = get_data_loaders(config)

    # Reconstruction Loss
    val_loss = evaluate_reconstruction(model, val_loader)
    test_loss = evaluate_reconstruction(model, test_loader)

    # Classification Error
    classification_error = evaluate_classification(model, test_loader)

    # Masked MSE
    masked_mse_value = masked_mse(model, test_loader)

    # Extract latent representations for density modeling
    print("Extracting latent representations for density modeling...")
    train_loader, _, _ = get_data_loaders(config)  # Ensure access to train_loader
    train_latents = extract_latent_representations(model, train_loader, device)

    # Fit GMM to latent representations
    print("Fitting GMM to latent representations...")
    gmm = GaussianMixture(n_components=config["n_gmm_components"], covariance_type="full", random_state=42)
    gmm.fit(train_latents)

    # Monte Carlo Log-Likelihood
    print("Calculating Monte Carlo Log-Likelihood...")
    log_likelihood = monte_carlo_log_likelihood(
        gmm, model, test_loader, n_samples=config["n_mc_samples"], batch_size=config["chunk_size"]
    )

    # Display Results
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Classification Error: {classification_error:.2f}%")
    print(f"Masked MSE: {masked_mse_value:.4f}")
    print(f"Monte Carlo Log-Likelihood: {log_likelihood:.4f}")
