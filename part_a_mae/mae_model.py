import torch
import torch.nn as nn


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = 64
        self.num_keep = 16

        self.patch_embed = nn.Linear(192, 192)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 192) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=192, nhead=3, dim_feedforward=768, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=6, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(192)

        self.dec_embed = nn.Linear(192, 128)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.dec_pos_embed = nn.Parameter(torch.randn(1, 64, 128) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2, enable_nested_tensor=False)
        self.dec_out = nn.Linear(128, 192)

    def random_masking(self, batch_size, device):
        noise = torch.rand(batch_size, self.num_patches, device=device)
        order = noise.argsort(dim=1)
        keep_idx = order[:, :self.num_keep]
        mask = torch.ones(batch_size, self.num_patches, device=device)
        mask.scatter_(1, keep_idx, 0.0)
        return keep_idx, mask

    def forward(self, patches):
        batch_size = patches.shape[0]
        keep_idx, mask = self.random_masking(batch_size, patches.device)

        x = self.patch_embed(patches) + self.pos_embed
        index = keep_idx.unsqueeze(-1).expand(-1, -1, 192)
        visible = torch.gather(x, 1, index)
        encoded = self.enc_norm(self.encoder(visible))

        d = self.dec_embed(encoded)
        full = self.mask_token.expand(batch_size, self.num_patches, 128)
        index = keep_idx.unsqueeze(-1).expand(-1, -1, 128)
        full = full.scatter(1, index, d)
        full = full + self.dec_pos_embed
        full = self.decoder(full)
        pred = self.dec_out(full)

        loss = self.masked_loss(pred, patches, mask)
        return loss, pred, mask

    def masked_loss(self, pred, patches, mask):
        per_patch = ((pred - patches) ** 2).mean(dim=-1)
        return (per_patch * mask).sum() / mask.sum()

    def features(self, patches):
        x = self.patch_embed(patches) + self.pos_embed
        encoded = self.enc_norm(self.encoder(x))
        return encoded.mean(dim=1)
