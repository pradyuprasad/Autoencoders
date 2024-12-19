from variational_autoencoder import VariationalAutoencoder
import torch
from train_autoencoder import load_data
from torchmetrics.image import StructuralSimilarityIndexMeasure


model = VariationalAutoencoder(final_dim=16)
model.load_state_dict(torch.load("variational_autoencoder.pth", weights_only=True))
train_dataloader, test_dataloader = load_data()
model.eval()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

total_ssim = 0
total_mse = 0
num_batches = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(test_dataloader):
        reconstructed = model(images, return_latent=False)
        ssim_score = ssim(reconstructed, images)
        mse_score = torch.nn.functional.mse_loss(reconstructed, images)

        total_ssim += ssim_score
        total_mse += mse_score
        num_batches += 1

avg_ssim = total_ssim / num_batches
avg_mse = total_mse / num_batches
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average MSE: {avg_mse:.4f}")
