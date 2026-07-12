import torch
import torch.nn as nn


class StudentDenoiser(nn.Module):
    """Time-independent, latent-conditioned one-step student: pure residual
    conv blocks over full (3, 32, 32) images, no time embedding, no
    attention, no ODE -- deliberately simpler than the teacher UNet since it
    is always evaluated once at a fixed t=0 (see
    code/cifar10/train_students.py). Learns the same conditional mapping as
    the teacher, (noise x0, AE latent) -> generated image x1, in a single
    forward pass: x1_pred = x0 + student(x0, latent).

    latent_input_dim is the raw AE latent size for this experiment (one of
    64/128/256/384/512/1024); latent_embed_dim is a fixed internal
    conditioning width, kept constant across all latent_input_dim settings so
    the six students are architecturally comparable except for the
    unavoidable input projection from the raw latent dimension.
    """

    def __init__(
        self,
        latent_input_dim: int,
        latent_embed_dim: int = 256,
        hidden_channels: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.latent_input_dim = latent_input_dim
        self.latent_embed_dim = latent_embed_dim
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks

        self.input_proj = nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1)

        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_input_dim, latent_embed_dim),
            nn.GELU(),
            nn.Linear(latent_embed_dim, hidden_channels),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(min(32, hidden_channels), hidden_channels),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(min(32, hidden_channels), hidden_channels),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            )
            for _ in range(n_blocks)
        ])

        self.output_head = nn.Sequential(
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_input_dim:
            raise ValueError(
                f"StudentDenoiser(latent_input_dim={self.latent_input_dim}) received a "
                f"latent of shape {tuple(latent.shape)}; expected (batch, {self.latent_input_dim})."
            )
        if latent.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x ({x.shape[0]}) and latent ({latent.shape[0]})."
            )

        h = self.input_proj(x)
        cond = self.latent_mlp(latent)
        h = h + cond[:, :, None, None]
        for block in self.blocks:
            h = h + block(h)
        return self.output_head(h)


def param_count(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    return f"{n / 1e3:.2f} K"


def load_student(ckpt_path: str, latent_dim: int, device: str = "cpu") -> StudentDenoiser:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if ckpt.get("conditioning") != "ae_latent":
        raise ValueError(
            f"Checkpoint '{ckpt_path}' is missing conditioning=='ae_latent' metadata, so it "
            f"is not a latent-conditioned student checkpoint -- it was likely produced by an "
            f"older, latent-free StudentDenoiser and is not compatible with this loader."
        )
    if ckpt["latent_dim"] != latent_dim:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' was distilled from latent_dim={ckpt['latent_dim']}, "
            f"but latent_dim={latent_dim} was requested."
        )

    model = StudentDenoiser(
        latent_input_dim=ckpt["latent_dim"],
        latent_embed_dim=ckpt["latent_embed_dim"],
        hidden_channels=ckpt["student_hidden_channels"],
        n_blocks=ckpt["student_n_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
