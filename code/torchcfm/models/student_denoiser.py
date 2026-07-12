import torch
import torch.nn as nn


class StudentDenoiser(nn.Module):
    """Time-independent one-step student: pure residual conv blocks over full
    (3, 32, 32) images, no time embedding, no FiLM conditioning, no attention
    -- deliberately simpler than the teacher UNet since it is always
    evaluated at a single fixed t=0 (see code/cifar10/train_students.py).
    Distills noise directly into the teacher's final generated image.
    """

    def __init__(self, hidden_channels: int = 64, n_blocks: int = 4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks

        self.input_proj = nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1)

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

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        # t is accepted only for call-site compatibility with the teacher's
        # forward(t, x, y) convention; the student never uses it (always t=0).
        h = self.input_proj(x)
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
    if ckpt["latent_dim"] != latent_dim:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' was distilled from latent_dim={ckpt['latent_dim']}, "
            f"but latent_dim={latent_dim} was requested."
        )
    model = StudentDenoiser(
        hidden_channels=ckpt["student_hidden_channels"],
        n_blocks=ckpt["student_n_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
