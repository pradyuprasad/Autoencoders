from variational_autoencoder import VariationalAutoencoder
from autoencoder import Autoencoder
import torch
from train_autoencoder import load_data
from torchmetrics.image import StructuralSimilarityIndexMeasure
import sys

if len(sys.argv) != 2 or sys.argv[1] not in ['vae', 'ae']:
   print("Usage: python script.py [vae|ae]")
   sys.exit(1)

model_type = sys.argv[1]
if model_type == 'vae':
   model = VariationalAutoencoder(final_dim=32)
   path = "variational_autoencoder.pth"
else:
   model = Autoencoder(final_dim=32)
   path = "autoencoder.pth"

model.load_state_dict(torch.load(path, weights_only=True))
train_dataloader, test_dataloader = load_data()
model.eval()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
total_ssim = 0
total_mse = 0
num_batches = 0

with torch.no_grad():
   for i, (images, labels) in enumerate(test_dataloader):
       if model_type == 'vae':
           reconstructed = model(images)[0]  # VAE returns (reconstructed, mu, logvar)
       else:
           reconstructed = model(images)
       ssim_score = ssim(reconstructed, images)
       mse_score = torch.nn.functional.mse_loss(reconstructed, images)
       total_ssim += ssim_score
       total_mse += mse_score
       num_batches += 1

avg_ssim = total_ssim / num_batches
avg_mse = total_mse / num_batches
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average MSE: {avg_mse:.4f}")
