from variational_autoencoder import VariationalAutoencoder
from autoencoder import Autoencoder
import torch
from train_autoencoder import load_data
import einops
import matplotlib.pyplot as plt
import sys

def plot_and_save_interpolation(images, filename):
    """Plot interpolation and save to file"""
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))

    for i, (ax, image) in enumerate(zip(axes, images)):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if image.shape[-1] == 1:
                image = image.squeeze()

        ax.imshow(image, cmap='gray')
        ax.set_title(f'Step {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

def interpolate_digits(model_type, digit_a, digit_b, steps=10):
    # Load appropriate model
    if model_type == 'vae':
        model = VariationalAutoencoder(final_dim=32)
        path = "variational_autoencoder.pth"
        is_vae = True
    else:
        model = Autoencoder(final_dim=32)
        path = "autoencoder.pth"
        is_vae = False

    model.load_state_dict(torch.load(path, weights_only=True))
    _, test_dataloader = load_data()
    model.eval()

    points_a = []
    points_b = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            mask_a = (labels == digit_a)
            mask_b = (labels == digit_b)

            if mask_a.any():
                batch_a = images[mask_a]
                if is_vae:
                    _, mu, _ = model(batch_a, return_latent=True)
                    points_a.append(mu)
                else:
                    x = einops.rearrange(batch_a, "b c h w -> b (c h w)")
                    encoded = model.encoder(x)
                    points_a.append(encoded)

            if mask_b.any():
                batch_b = images[mask_b]
                if is_vae:
                    _, mu, _ = model(batch_b, return_latent=True)
                    points_b.append(mu)
                else:
                    x = einops.rearrange(batch_b, "b c h w -> b (c h w)")
                    encoded = model.encoder(x)
                    points_b.append(encoded)

    points_a = torch.cat(points_a)
    points_b = torch.cat(points_b)

    avg_a = torch.mean(points_a, dim=0)
    avg_b = torch.mean(points_b, dim=0)

    interpolated_images = []
    with torch.no_grad():
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated = (1 - alpha) * avg_a + alpha * avg_b

            if is_vae:
                decoded = model.decoder(interpolated.unsqueeze(0))
            else:
                decoded = model.decoder(interpolated.unsqueeze(0))

            image = einops.rearrange(decoded, "b (c h w) -> b c h w", c=1, h=28, w=28)
            interpolated_images.append(image[0])

    filename = f"interpolation_{model_type}_{digit_a}_to_{digit_b}.png"
    plot_and_save_interpolation(interpolated_images, filename)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py [vae|ae] digit_a digit_b")
        sys.exit(1)

    model_type = sys.argv[1]
    if model_type not in ['vae', 'ae']:
        print("Model type must be 'vae' or 'ae'")
        sys.exit(1)

    try:
        digit_a = int(sys.argv[2])
        digit_b = int(sys.argv[3])
        if not (0 <= digit_a <= 9 and 0 <= digit_b <= 9):
            raise ValueError
    except ValueError:
        print("Digits must be integers between 0 and 9")
        sys.exit(1)

    interpolate_digits(model_type, digit_a, digit_b)
