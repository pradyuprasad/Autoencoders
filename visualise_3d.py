from autoencoder import Autoencoder
import torch
from train_autoencoder import load_data
import matplotlib.pyplot as plt
import einops

model = Autoencoder()
model.load_state_dict(torch.load("autoencoder.pth", weights_only=True))
train_dataloader, test_dataloader = load_data()
model.eval()

encoded_points = []
labels_for_plot = []

with torch.no_grad():
    for images, labels in test_dataloader:
        # Get encoded representations (similar to your forward method)
        x = einops.rearrange(images, "b c h w -> b (c h w)")
        encoded = model.encoder(x)
        
        encoded_points.append(encoded)
        labels_for_plot.append(labels)

# Convert to numpy arrays
encoded_points = torch.cat(encoded_points).numpy()
labels_for_plot = torch.cat(labels_for_plot).numpy()

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(encoded_points[:, 0], 
                    encoded_points[:, 1], 
                    encoded_points[:, 2],
                    c=labels_for_plot, 
                    cmap='tab10')
plt.colorbar(scatter)
plt.title('Latent Space Visualization')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()
