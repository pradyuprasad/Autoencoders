import torchvision
from torchvision.transforms import ToTensor
from variational_autoencoder import VariationalAutoencoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader

def calculate_total_loss(actual, reconstructed, mu, logvar, beta=0.00001, current_epoch=0, warmup_epochs=10, alpha=1.0):
    recon_loss = nn.functional.mse_loss(actual, reconstructed)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    beta_t = beta * min(1.0, current_epoch / warmup_epochs)
    return alpha * recon_loss + beta_t * kl_loss

def get_validation_loss(model: VariationalAutoencoder, test_dataloader: torch.utils.data.DataLoader, current_epoch: int):
    sum_loss = 0
    model.eval()
    with torch.no_grad():
        for index, (image, label) in enumerate(test_dataloader):
            output, mu, logvar = model(image, return_latent=True)
            sum_loss += calculate_total_loss(image, output, mu, logvar, current_epoch=current_epoch)
    model.train()
    return sum_loss / len(test_dataloader)

def train_autoencoder(train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, num_epochs: int = 100, patience: int = 10, min_delta: float = 0.0001, beta: float = 0.0001) -> VariationalAutoencoder:
    model = VariationalAutoencoder(final_dim=16)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    best_loss = float('inf')
    patience_counter = 0
    print("Starting training")

    final_mus = []
    final_logvars = []

    for i in range(num_epochs):
        model.train()
        train_recon_loss = 0
        train_kl_loss = 0
        num_batches = 0
        epoch_mus = []
        epoch_logvars = []

        for index, (image, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(image)
            epoch_mus.append(mu.detach().cpu())
            epoch_logvars.append(logvar.detach().cpu())
            recon_loss = loss_fn(image, reconstructed)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = calculate_total_loss(image, reconstructed, mu, logvar, beta, current_epoch=i)
            loss.backward()
            optimizer.step()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            num_batches += 1

        final_mus = epoch_mus
        final_logvars = epoch_logvars

        avg_train_recon = train_recon_loss / num_batches
        avg_train_kl = train_kl_loss / num_batches
        val_loss = get_validation_loss(model, test_dataloader, i)
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        train_total_loss = avg_train_recon + beta * avg_train_kl  # Using same beta weighting as val
        print(f"Epoch {i:7d} | Val Loss: {val_loss:.4f} | Train Total: {train_total_loss:.4f} | Train Recon: {avg_train_recon:.4f} | Train KL: {avg_train_kl:.4f} | Patience: {patience_counter}")
        if patience_counter >= patience:
            print(f"Early stopping at epoch {i}")
            break

    # Plot the distributions
    all_mus = torch.cat(final_mus, dim=0).numpy()
    all_logvars = torch.cat(final_logvars, dim=0).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(all_mus.flatten(), bins=50)
    plt.title('Distribution of mu values')
    plt.xlabel('mu')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.histplot(all_logvars.flatten(), bins=50)
    plt.title('Distribution of logvar values')
    plt.xlabel('logvar')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    return model

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


if __name__ == "__main__":
    set_seed(42)
    train_dataloader, test_dataloader = load_data()
    model = train_autoencoder(train_dataloader, test_dataloader)
    torch.save(model.state_dict(), "variational_autoencoder.pth")
    print("saved the model")
