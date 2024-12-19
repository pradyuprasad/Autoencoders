import torch.nn as nn
import einops

class Autoencoder(nn.Module):
    def __init__(self, final_dim=3):
        '''
        28x28 input -> dimension of final_dim -> 28x28 input
        Using same architecture as VAE:
        - Encoder: 784 -> 512 -> 256 -> 128 -> final_dim
        - Decoder: final_dim -> 128 -> 256 -> 512 -> 784
        '''
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, final_dim)
                )
        self.decoder = nn.Sequential(
                nn.Linear(final_dim, 128),
                nn.ELU(),
                nn.Linear(128, 256),
                nn.ELU(),
                nn.Linear(256, 512),
                nn.ELU(),
                nn.Linear(512, 28*28),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b (c h w)")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return einops.rearrange(decoded, "b (c h w) -> b c h w", c=1, h=28, w=28)
