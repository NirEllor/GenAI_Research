# Train one-step "student" models that distill each Latent-CFM teacher
# (code/cifar10/train_cifar10_ddp_vae_cond_ic.py, via the synthetic datasets
# from generate_teacher_datasets.py) into a single forward pass.
#
# For each latent dim x dataset size (6 x 4 = 24 students by default):
#   - Loads the independent (x0, x1) pairs generated for that dim/size
#     (x0 = starting image noise at t=0, x1 = the teacher's Euler-integrated
#     final sample at t=1 -- this repo's convention, see
#     train_cifar10_ddp_vae_cond_ic.py).
#   - Trains a student UNetModelWrapper (no latent conditioning, small
#     channel/res-block count) to predict the single global velocity
#     v = x1 - x0 evaluated at x0 (t fixed at 0): one-step distillation, so a
#     trained student generates a sample as x1_pred = x0 + student(x0) in a
#     single forward pass -- no ODE integration needed at inference.
#
# Skips any student checkpoint that already exists -- safe to restart.
#
# Usage:
#   python code/cifar10/train_students.py                       # all 24 students sequentially
#   python code/cifar10/train_students.py --dim 128             # all 4 sizes for dim=128
#   python code/cifar10/train_students.py --dim 128 --size 200000  # single student

import sys

sys.path.append("./code/cifar10/")
sys.path.append("./code/torchcfm/models/unet/")

import os
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from unet_resnetVAE import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", "./code/cifar10/runs/", "base dir containing per-dim synthetic/ data (output of generate_teacher_datasets.py)")
flags.DEFINE_string("output_dir", None, "base dir to write student checkpoints; defaults to --input_dir")
flags.DEFINE_string("model", "icfm", "flow matching model subdirectory (must match generate_teacher_datasets.py)")

flags.DEFINE_integer("dim", None, "single latent dim to distil (omit for all --latent_dims)")
flags.DEFINE_integer("size", None, "single dataset size to use (omit for all --dataset_sizes)")
flags.DEFINE_list("latent_dims", ["64", "128", "256", "384", "512", "1024"], "AE latent dims to process")
flags.DEFINE_list("dataset_sizes", ["50000", "100000", "150000", "200000"], "synthetic dataset sizes to distil on")

flags.DEFINE_integer("student_num_channels", 64, "student UNet base channel count (teacher default is 128)")
flags.DEFINE_integer("student_num_res_blocks", 1, "student UNet res blocks per level (teacher default is 2)")

flags.DEFINE_integer("epochs", 500, "max training epochs")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_float("lr", 3e-4, "learning rate")
flags.DEFINE_float("weight_decay", 1e-4, "AdamW weight decay")
flags.DEFINE_float("grad_clip", 1.0, "gradient norm clipping")
flags.DEFINE_float("ema_decay", 0.9999, "EMA decay rate")
flags.DEFINE_integer("log_interval", 50, "print running loss every N steps")
flags.DEFINE_integer("early_stop_patience", 50, "stop if no improvement for this many epochs")
flags.DEFINE_float("early_stop_min_delta", 0.0, "minimum loss improvement to reset patience")

flags.DEFINE_bool("load_to_ram", True, "copy the dataset into RAM before training (off: file-backed memmap, lower RAM, slower)")
flags.DEFINE_integer("num_workers", 2, "DataLoader workers")
flags.DEFINE_bool("overwrite", False, "retrain students that already have a checkpoint")
flags.DEFINE_integer("seed", 0, "RNG seed")

use_cuda = torch.cuda.is_available()


def get_device(dim=None):
    if not use_cuda:
        return torch.device("cpu")
    if dim is not None:
        dims = [int(d) for d in FLAGS.latent_dims]
        gpu_id = dims.index(dim) % torch.cuda.device_count() if dim in dims else 0
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda")


def param_count(model):
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    return f"{n / 1e3:.2f} K"


def build_student(device):
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=FLAGS.student_num_res_blocks,
        num_channels=FLAGS.student_num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.0,
        num_latents=None,
    ).to(device)


def create_ema(model, device):
    ema = deepcopy(model).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema(ema, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class NoiseSamplePairs(Dataset):
    """(x0, x1) pairs from generate_teacher_datasets.py's per-size output:
    x0 = noise_fp16_n{size}.npy (unclipped float16), x1 = images_uint8_n{size}.npy
    (uint8, rescaled back to [-1,1] here)."""

    def __init__(self, synthetic_dir, size, load_to_ram):
        noise_path = synthetic_dir / f"noise_fp16_n{size}.npy"
        images_path = synthetic_dir / f"images_uint8_n{size}.npy"
        mmap_mode = None if load_to_ram else "r"
        self.x0 = np.load(str(noise_path), mmap_mode=mmap_mode)
        self.x1 = np.load(str(images_path), mmap_mode=mmap_mode)
        if load_to_ram:
            self.x0 = np.array(self.x0)
            self.x1 = np.array(self.x1)

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        x0 = torch.from_numpy(np.array(self.x0[idx])).to(torch.float32)
        # Inverts generate_teacher_datasets.py's (x1 * 127.5 + 128).clip(0, 255).to(uint8) encoding.
        x1 = (torch.from_numpy(np.array(self.x1[idx])).to(torch.float32) - 128.0) / 127.5
        return x0, x1


def plot_loss(history, dim, n_samples, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("One-step distillation loss")
    ax.set_title(f"Student Loss  (dim={dim}, n={n_samples:,})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(plots_dir / f"student_loss_{dim}_{n_samples}.png"), dpi=150)
    plt.close()


def train_student(dim, n_samples, device, base_dir):
    students_dir = base_dir / "students"
    students_dir.mkdir(parents=True, exist_ok=True)
    out_path = students_dir / f"student_{dim}_{n_samples}.pt"
    if out_path.exists() and not FLAGS.overwrite:
        print(f"  [skip] {out_path.name} already exists.", flush=True)
        return

    synthetic_dir = base_dir / "synthetic"
    if not (synthetic_dir / f"images_n{n_samples}.done").exists():
        print(f"  [ERROR] {synthetic_dir} has no completed images_n{n_samples} dataset "
              f"-- run generate_teacher_datasets.py first.", flush=True)
        return

    dataset = NoiseSamplePairs(synthetic_dir, n_samples, FLAGS.load_to_ram)
    loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    print(f"  Batch shape      : ({FLAGS.batch_size}, 3, 32, 32)", flush=True)
    print(f"  Steps per epoch  : {steps_per_epoch:,}", flush=True)

    student = build_student(device)
    ema_student = create_ema(student, device)
    print(f"  Student params   : {param_count(student)}", flush=True)

    optimizer = AdamW(student.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=FLAGS.epochs, eta_min=FLAGS.lr * 0.01)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in tqdm(range(1, FLAGS.epochs + 1), desc=f"    dim={dim} n={n_samples:,}"):
        student.train()
        total_loss = 0.0

        for batch_idx, (x0, x1) in enumerate(loader):
            x0 = x0.to(device)
            x1 = x1.to(device)

            t = torch.zeros(x0.size(0), device=device)
            v_target = x1 - x0
            v_pred = student(t, x0)
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), FLAGS.grad_clip)
            optimizer.step()
            update_ema(ema_student, student, FLAGS.ema_decay)

            total_loss += loss.item()

            if (batch_idx + 1) % FLAGS.log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"[dim={dim} n={n_samples} ep {epoch:03d} step {batch_idx + 1:05d}] loss={avg:.5f}", flush=True)

        scheduler.step()

        avg_loss = total_loss / steps_per_epoch
        history.append(avg_loss)

        if avg_loss < best_loss - FLAGS.early_stop_min_delta:
            best_loss = avg_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = deepcopy(ema_student.state_dict())
        else:
            epochs_without_improvement += 1

        print(
            f"    [dim={dim} n={n_samples}] epoch {epoch:03d}  loss={avg_loss:.5f}  "
            f"best={best_loss:.5f}  best_epoch={best_epoch:03d}",
            flush=True,
        )

        if epochs_without_improvement >= FLAGS.early_stop_patience:
            print(f"  Early stopping at epoch {epoch:03d}. Best epoch={best_epoch:03d}, best_loss={best_loss:.5f}", flush=True)
            break

    if best_state is not None:
        ema_student.load_state_dict(best_state)

    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": ema_student.state_dict(),
            "latent_dim": dim,
            "n_samples": n_samples,
            "student_num_channels": FLAGS.student_num_channels,
            "student_num_res_blocks": FLAGS.student_num_res_blocks,
            "loss_history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
        },
        tmp_path,
    )
    os.replace(tmp_path, out_path)
    print(f"  Saved -> {out_path} (best_epoch={best_epoch}, best_loss={best_loss:.5f})", flush=True)

    plot_loss(history, dim, n_samples, students_dir / "plots")


def main(argv):
    torch.manual_seed(FLAGS.seed)

    dims = [FLAGS.dim] if FLAGS.dim else [int(d) for d in FLAGS.latent_dims]
    sizes = [FLAGS.size] if FLAGS.size else [int(s) for s in FLAGS.dataset_sizes]

    for dim in dims:
        device = get_device(dim)
        print(f"\n{'=' * 60}\nDistilling students  latent_dim={dim}  device={device}\n{'=' * 60}", flush=True)
        base_dir = (Path(FLAGS.output_dir) if FLAGS.output_dir else Path(FLAGS.input_dir)) / f"latent_{dim}" / FLAGS.model
        for n_samples in sizes:
            train_student(dim, n_samples, device, base_dir)

    print("\nStudent distillation complete.", flush=True)


if __name__ == "__main__":
    app.run(main)
