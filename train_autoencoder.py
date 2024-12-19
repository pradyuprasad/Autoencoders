import torchvision
from torchvision.transforms import ToTensor
from autoencoder import Autoencoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

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

def get_validation_loss(model: Autoencoder, test_dataloader: torch.utils.data.DataLoader):
    total_loss = 0
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for index, (image, label) in enumerate(test_dataloader):
            output = model(image)
            loss = loss_fn(image, output)
            total_loss += loss.item()
    model.train()
    return total_loss / len(test_dataloader)


def train_autoencoder(train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
                      num_epochs: int = 100, patience: int = 5, min_delta: float = 0.0001) -> Autoencoder:
    model = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    best_loss = float('inf')
    patience_counter = 0
    print("Starting training")
    for i in range(num_epochs):
        model.train()
        for index, (image, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            reconstructed = model(image)
            loss = loss_fn(image, reconstructed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss = get_validation_loss(model, test_dataloader)

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"At epoch {i} the validation loss is, {val_loss}, and patience is {patience_counter}")
        if patience_counter > patience:
            print(f"Early stopping at epoch {i}")
            break

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
    torch.save(model.state_dict(), "autoencoder.pth")
    print("saved the model")
