import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class ConvAutoencoder(nn.Module):
    """Deterministic conv autoencoder trained externally (LPIPS reconstruction loss,
    images in [0, 1]). Kept architecturally identical to the external training script
    so ae_<dim>.pt checkpoints load via load_state_dict without modification.
    """

    def __init__(self, latent_dim: int):
        super().__init__()

        if latent_dim % 16 != 0:
            raise ValueError("latent_dim must be divisible by 16 for CIFAR (4x4)")

        self.latent_dim = latent_dim
        self.latent_channels = latent_dim // 16

        # ------------------------------------------------------------------
        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4
        # ------------------------------------------------------------------
        self.encoder_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.down1 = nn.Sequential(ResNetBlock(128), nn.Conv2d(128, 128, 3, stride=2, padding=1))  # -> 16x16
        self.down2 = nn.Sequential(ResNetBlock(128), nn.Conv2d(128, 256, 3, stride=2, padding=1))  # -> 8x8
        self.down3 = nn.Sequential(ResNetBlock(256), nn.Conv2d(256, 256, 3, stride=2, padding=1))  # -> 4x4

        self.encoder_out = nn.Sequential(
            ResNetBlock(256),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, self.latent_channels, kernel_size=1)
        )

        # ------------------------------------------------------------------
        # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32
        # ------------------------------------------------------------------
        self.decoder_in = nn.Conv2d(self.latent_channels, 256, kernel_size=1)

        self.up1 = nn.Sequential(ResNetBlock(256),
                                 nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1))  # -> 8x8
        self.up2 = nn.Sequential(ResNetBlock(256),
                                 nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))  # -> 16x16
        self.up3 = nn.Sequential(ResNetBlock(128),
                                 nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1))  # -> 32x32

        self.decoder_out = nn.Sequential(
            ResNetBlock(128),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def encode(self, x: torch.Tensor, sample: bool = False):
        h = self.encoder_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        z = self.encoder_out(h)

        z_flat = torch.flatten(z, 1)
        return z_flat, z_flat, torch.zeros_like(z_flat)  # (z, mean, dummy logvar)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        z_spatial = latent.view(-1, self.latent_channels, 4, 4)
        h = self.decoder_in(z_spatial)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return self.decoder_out(h)

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encode(x, sample=False)
        recon_logits = self.decode(z)
        return recon_logits, mean, logvar
