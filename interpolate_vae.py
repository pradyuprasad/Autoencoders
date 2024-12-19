from variational_autoencoder import VariationalAutoencoder
import torch
from train_autoencoder import load_data
import einops
from display_image import plot_interpolation

def interpolate_digits(model, dataloader, digit_a, digit_b, steps=10):
    """
    Interpolate between two digits in the latent space of a VAE.

    Args:
        model: Trained VariationalAutoencoder.
        dataloader: DataLoader containing the dataset.
        digit_a (int): First digit to interpolate from.
        digit_b (int): Second digit to interpolate to.
        steps (int): Number of interpolation steps.
    """
    model.eval()
    points_a_mu = []
    points_b_mu = []

    # Extract latent means (mu) for the two digits
    with torch.no_grad():
        for images, labels in dataloader:
            mask_a = (labels == digit_a)
            mask_b = (labels == digit_b)

            if mask_a.any():
                batch_a = images[mask_a]
                _, mu, _ = model(batch_a, return_latent=True)
                points_a_mu.append(mu)

            if mask_b.any():
                batch_b = images[mask_b]
                _, mu, _ = model(batch_b, return_latent=True)
                points_b_mu.append(mu)

    # Combine all latent means (mu) into tensors
    if not points_a_mu or not points_b_mu:
        print(f"Not enough data for digits {digit_a} and {digit_b}")
        return

    points_a_mu = torch.cat(points_a_mu, dim=0)
    points_b_mu = torch.cat(points_b_mu, dim=0)

    # Compute the average latent vectors
    avg_a_mu = torch.mean(points_a_mu, dim=0)  # Shape: [latent_dim]
    avg_b_mu = torch.mean(points_b_mu, dim=0)  # Shape: [latent_dim]

    # Interpolate between the two average latent vectors
    interpolated_images = []
    with torch.no_grad():
        for i in range(steps):
            alpha = i / (steps - 1)  # Linear interpolation factor
            interpolated_latent = (1 - alpha) * avg_a_mu + alpha * avg_b_mu

            # Decode the interpolated latent vector
            decoded = model.decoder(interpolated_latent.unsqueeze(0))  # Add batch dimension
            image = einops.rearrange(decoded, "b (c h w) -> b c h w", c=1, h=28, w=28)
            interpolated_images.append(image[0])

    # Plot the interpolation results
    plot_interpolation(interpolated_images)

if __name__ == "__main__":
    # Load the trained beta-VAE model
    model = VariationalAutoencoder(final_dim=16)
    model.load_state_dict(torch.load("variational_autoencoder.pth", weights_only=True))

    # Load the MNIST dataset
    _, test_dataloader = load_data()

    # Example: Interpolate between digits 3 and 8
    interpolate_digits(model, test_dataloader, digit_a=0, digit_b=1, steps=10)
