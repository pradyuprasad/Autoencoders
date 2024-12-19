import torch.nn as nn
import einops

class Autoencoder(nn.Module):
    def __init__(self, final_dim=3):
        '''
        28x28 input -> dimension of final_sim -> 28x28 input
        '''
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, final_dim)
                )
        self.decoder = nn.Sequential(
                nn.Linear(final_dim, 32),
                nn.ELU(),
                nn.Linear(32, 64),
                nn.ELU(),
                nn.Linear(64, 128),
                nn.ELU(),
                nn.Linear(128, 28*28),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b (c h w)")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return einops.rearrange(decoded, "b (c h w) -> b c h w", c=1, h=28, w=28)
