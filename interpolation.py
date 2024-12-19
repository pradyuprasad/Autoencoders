from variational_autoencoder import VariationalAutoencoder
import torch
from train_autoencoder import load_data
import einops
from display_image import plot_interpolation

model = VariationalAutoencoder(final_dim=16)
model.load_state_dict(torch.load("variational_autoencoder.pth", weights_only=True))
train_dataloader, test_dataloader = load_data()
model.eval()

points_5 = []
points_9 = []

with torch.no_grad():
    for images, labels in test_dataloader:
        mask_5 = (labels == 5)
        mask_9 = (labels == 9)

        if mask_5.any():
            batch_5s = images[mask_5]
            x = einops.rearrange(batch_5s, "b c h w -> b (c h w)")
            encoded_5s = model.encoder(x)
            points_5.append(encoded_5s)

        if mask_9.any():
            batch_9s = images[mask_9]
            x = einops.rearrange(batch_9s, "b c h w -> b (c h w)")
            encoded_9s = model.encoder(x)
            points_9.append(encoded_9s)

points_5 = torch.cat(points_5)
points_9 = torch.cat(points_9)
avg_5 = einops.reduce(points_5, "b d -> d", "mean")
avg_9 = einops.reduce(points_9, "b d -> d", "mean")
diff_x = avg_9[0] - avg_5[0]
diff_y = avg_9[1] - avg_5[1]
diff_z = avg_9[2] - avg_5[2]

interpolated_images = []
with torch.no_grad():
    for i in range(10):
        point = torch.tensor([
            avg_5[0] + (i/9) * diff_x,
            avg_5[1] + (i/9) * diff_y,
            avg_5[2] + (i/9) * diff_z
        ])
        decoded = model.decoder(point.unsqueeze(0))
        image = einops.rearrange(decoded, "b (c h w) -> b c h w", c=1, h=28, w=28)
        interpolated_images.append(image[0])

plot_interpolation(interpolated_images)

