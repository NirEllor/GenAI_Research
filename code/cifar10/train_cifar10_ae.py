"""
Step 0 — Train ConvAutoencoder from scratch on CIFAR-10 for each latent dim.

Saves: checkpoints/<exp_name>/ae_<dim>.pt
       checkpoints/<exp_name>/config.json

Usage:
    python train_cifar10_ae.py                        # all dims sequentially
    python train_cifar10_ae.py --dim 64               # single dim
    python train_cifar10_ae.py --exp-name my_run      # custom experiment name
"""

import argparse
import lpips
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "torchcfm"))
from models.conv_autoencoder import ConvAutoencoder
from exp_config import get_paths, add_exp_arg, save_config, print_exp_summary

LATENT_DIMS  = [64, 128, 256, 384, 512, 1024]
EPOCHS       = 1000
BATCH_SIZE   = 128
LR           = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0
LPIPS_WEIGHT = 1.0
MSE_WEIGHT   = 0.0
KL_WEIGHT    = 0.0


def get_cifar10_loader(batch_size: int) -> DataLoader:
    tf = transforms.Compose([
        transforms.ToTensor(),   # → [0, 1]
    ])
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)


def train_one_dim(dim: int, device: torch.device, ckpt_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Training ConvAutoencoder  latent_dim={dim}  device={device}")
    print(f"{'='*60}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"ae_{dim}.pt"

    loader = get_cifar10_loader(BATCH_SIZE)
    model  = ConvAutoencoder(latent_dim=dim).to(device)
    opt      = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched    = CosineAnnealingLR(opt, T_max=EPOCHS)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()  # frozen VGG backbone — only AE weights train
    for p in lpips_fn.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, _ in tqdm(loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False):
            imgs = imgs.to(device)

            recon_logits, mean, logvar = model(imgs)
            recon_for_loss = torch.sigmoid(recon_logits)

            recon_lpips = F.interpolate(
                recon_for_loss,
                size=(64, 64),
                mode="bilinear",
                align_corners=False
            )

            imgs_lpips = F.interpolate(
                imgs,
                size=(64, 64),
                mode="bilinear",
                align_corners=False
            )

            lpips_loss = lpips_fn(
                recon_lpips * 2 - 1,
                imgs_lpips * 2 - 1
            ).mean()

            kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()
            pixel_loss = F.mse_loss(recon_for_loss, imgs)
            loss = MSE_WEIGHT * pixel_loss + LPIPS_WEIGHT * lpips_loss + KL_WEIGHT * kl_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        sched.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"latent_dim": dim, "state_dict": model.state_dict()}, save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}"
                f"{'  [saved]' if avg_loss == best_loss else ''}"
            )

    print(f"Done. Best loss={best_loss:.6f}  →  {save_path}")


def get_device(dim: int = None) -> torch.device:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to train (omit for all dims sequentially)")
    add_exp_arg(parser)
    args = parser.parse_args()

    paths = get_paths(args.exp_name)

    save_config(paths, section="autoencoder", extra={
        "model": "ConvAutoencoder",
        "dataset": "CIFAR-10",
        "latent_dims": LATENT_DIMS,
        "loss_type": "LPIPS+MSE+KL",
        "lpips_net": "vgg",
        "lpips_input_size": 64,
        "mse_weight": MSE_WEIGHT,
        "kl_weight": KL_WEIGHT,
        "lpips_weight": LPIPS_WEIGHT,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "deterministic_encoding": True,
    })

    dims = [args.dim] if args.dim else LATENT_DIMS

    print_exp_summary(
        paths,
        ckpt_path=paths.ckpt_dir / "ae_<dim>.pt",
    )

    for dim in dims:
        train_one_dim(dim, get_device(dim), paths.ckpt_dir)


if __name__ == "__main__":
    main()
