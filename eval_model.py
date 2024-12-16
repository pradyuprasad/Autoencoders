from autoencoder import Autoencoder
import torch
from train_autoencoder import load_data
import matplotlib.pyplot as plt

model = Autoencoder()
model.load_state_dict(torch.load("autoencoder.pth", weights_only=True))
train_dataloader, test_dataloader = load_data()
model.eval()

with torch.no_grad():
    for i, (images, labels) in enumerate(test_dataloader):
        if i == 0:
            reconstructed = model(images)
            for j in range(len(images)):
                plt.figure(figsize=(12, 6))
                
                # Original image
                plt.subplot(1, 2, 1)
                plt.imshow(images[j].squeeze(), cmap='gray')
                plt.title('Original')
                plt.axis('off')
                
                # Reconstructed image
                plt.subplot(1, 2, 2)
                plt.imshow(reconstructed[j].squeeze(), cmap='gray')
                plt.title('Reconstructed')
                plt.axis('off')
                
                plt.show()
            break
