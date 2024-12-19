import torch.nn as nn
import torch
import einops

class VariationalAutoencoder(nn.Module):
    def __init__(self, final_dim=3):
        '''
        28x28 input -> dimension of final_sim -> 28x28 input
        '''
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, final_dim * 2)
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

    def forward(self, x, return_latent=True):
        x = einops.rearrange(x, "b c h w -> b (c h w)")
        encoded = self.encoder(x)  # Shape: [batch_size, final_dim * 2]

        mu, logvar = torch.chunk(encoded, 2, dim=1)

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        decoded = self.decoder(z)
        decoded = einops.rearrange(decoded, "b (c h w) -> b c h w", c=1, h=28, w=28)

        if return_latent:
            return decoded, mu, logvar
        else:
            return decoded
