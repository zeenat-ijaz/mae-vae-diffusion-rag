import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc_conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.reshape(-1, 64, 7, 7)
        h = F.relu(self.dec_conv1(h))
        return self.dec_conv2(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


def kl_term(mu, logvar):
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)
    return kl.mean()


def recon_term(logits, x):
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return bce.sum(dim=(1, 2, 3)).mean()


def vae_loss(logits, x, mu, logvar):
    recon = recon_term(logits, x)
    kl = kl_term(mu, logvar)
    return recon + kl, recon, kl
