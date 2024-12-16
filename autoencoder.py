import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, final_dim=3):
        '''
        28x28 input -> dimension of final_sim -> 28x28 input
        '''
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ELU(),
                nn.Linear(128, 32),
                nn.ELU(),
                nn.Linear(32, final_dim)
                )
        self.decoder = nn.Sequential(
                nn.Linear(final_dim, 32),
                nn.ELU(),
                nn.Linear(32, 128),
                nn.ELU(),
                nn.Linear(128, 28*28),
                nn.Sigmoid()
                )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
