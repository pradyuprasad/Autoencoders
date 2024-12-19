import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from variational_autoencoder import VariationalAutoencoder
from autoencoder import Autoencoder

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    datasets = [torchvision.datasets.MNIST(root='./data', train=x, download=True, transform=ToTensor())
                for x in [True, False]]
    return [DataLoader(ds, batch_size=64, shuffle=True) for ds in datasets]

def train_model(model_type='ae', epochs=100, patience=10, min_delta=0.0001, beta=0.001):
    train_dl, test_dl = load_data()
    is_vae = model_type == 'vae'
    model = (VariationalAutoencoder() if is_vae else Autoencoder()).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    best_loss = float('inf')
    patience_count = 0
    mu_logs = []

    for epoch in range(epochs):
        model.train()
        train_recon = train_kl = num_batches = 0
        epoch_mu_logvar = []

        for img, _ in train_dl:
            img = img.to(device)
            optimizer.zero_grad()
            if is_vae:
                recon, mu, logvar = model(img)
                epoch_mu_logvar.append((mu.detach().cpu(), logvar.detach().cpu()))
                recon_loss = nn.functional.mse_loss(img, recon)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2).clamp(max=88) - logvar.exp().clamp(min=1e-8, max=1e8))
                loss = recon_loss + beta * kl_loss
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
            else:
                recon = model(img)
                loss = nn.functional.mse_loss(img, recon)
                train_recon += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            num_batches += 1

        if is_vae:
            mu_logs = epoch_mu_logvar

        model.eval()
        with torch.no_grad():
            val_loss = sum(nn.functional.mse_loss(model(img.to(device))[0] if is_vae else model(img.to(device)), img.to(device))
                          for img, _ in test_dl) / len(test_dl)

        if val_loss < best_loss - min_delta:
            best_loss, patience_count = val_loss, 0
        else:
            patience_count += 1

        avg_recon = train_recon / num_batches
        print(f"Epoch {epoch:3d} | Val Loss: {val_loss:.4f} | Train Recon: {avg_recon:.4f}" +
              (f" | Train KL: {train_kl/num_batches:.4f}" if is_vae else "") +
              f" | Patience: {patience_count}")

        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if is_vae and mu_logs:
        mus, logvars = zip(*mu_logs)
        plt.figure(figsize=(12, 4))
        for i, (data, title) in enumerate(zip([torch.cat(mus), torch.cat(logvars)], ['mu', 'logvar'])):
            plt.subplot(1, 2, i+1)
            sns.histplot(data.numpy().flatten(), bins=50)
            plt.title(f'Distribution of {title} values')
            plt.xlabel(title)
        plt.show()

    return model

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['vae', 'ae']:
        print("Usage: python train.py [vae|ae]")
        sys.exit(1)

    torch.manual_seed(42)
    model_type = sys.argv[1]
    model = train_model(model_type=model_type, beta=0.001 if model_type == 'vae' else None)
    torch.save(model.state_dict(), f"{'variational_' if model_type == 'vae' else ''}autoencoder.pth")
