import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 360),
            nn.ReLU(),
            nn.Linear(360, 360),
            nn.ReLU(),
            nn.Linear(360, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 360),
            nn.ReLU(),
            nn.Linear(360, 360),
            nn.ReLU(),
            nn.Linear(360, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)


class RegularizedAutoencoder(nn.Module):
    def __init__(self, latent_dim, std):
        super(RegularizedAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.initialize_weights(std)

    def initialize_weights(self, std):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
