# Train one-step "student" models that distill each Latent-CFM teacher
# (code/cifar10/train_cifar10_ddp_vae_cond_ic.py, via the single synthetic
# pool per dim from generate_teacher_datasets.py) into a single forward pass,
# conditioned on the same AE latent the teacher receives.
#
# For each latent dim x dataset size (6 x 4 = 24 students by default):
#   - Loads that dim's single (x0, x1, latent) synthetic pool once
#     (x0 = starting image noise at t=0, x1 = the teacher's Euler-integrated
#     final sample at t=1, latent = the AE latent that conditioned that
#     generation -- this repo's convention, see
#     train_cifar10_ddp_vae_cond_ic.py and generate_teacher_datasets.py) and
#     trains each dataset-size experiment on the prefix `[0:n]` of that pool,
#     so a bigger dataset-size experiment is a strict superset of a smaller
#     one rather than a disjoint resample.
#   - Trains a StudentDenoiser (code/torchcfm/models/student_denoiser.py --
#     small, time-independent residual conv net, no timestep embedding, no
#     attention) to predict the single global velocity v = x1 - x0 evaluated
#     at (x0, latent), t fixed at 0: one-step distillation, so a trained
#     student generates a sample as x1_pred = x0 + student(x0, latent) in a
#     single forward pass -- no ODE integration needed at inference.
#   - Every student trains for the same fixed --total_steps optimizer
#     updates, regardless of dataset size, so dataset size is the only
#     variable that differs between experiments (a larger pool would
#     otherwise also mean more gradient updates per "epoch").
#
# Skips any student checkpoint that already exists -- safe to restart.
#
# Usage:
#   python code/cifar10/train_students.py                       # all 24 students sequentially
#   python code/cifar10/train_students.py --dim 128             # all 4 sizes for dim=128
#   python code/cifar10/train_students.py --dim 128 --size 200000  # single student

import sys

sys.path.append("./code/cifar10/")
sys.path.append("./code/torchcfm/models/")

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

from student_denoiser import StudentDenoiser, param_count

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", "./code/cifar10/runs/", "base dir containing per-dim synthetic/ data (output of generate_teacher_datasets.py)")
flags.DEFINE_string("output_dir", None, "base dir to write student checkpoints; defaults to --input_dir")
flags.DEFINE_string("model", "icfm", "flow matching model subdirectory (must match generate_teacher_datasets.py)")

flags.DEFINE_integer("dim", None, "single latent dim to distil (omit for all --latent_dims)")
flags.DEFINE_integer("size", None, "single dataset size to use (omit for all --dataset_sizes)")
flags.DEFINE_list("latent_dims", ["64", "128", "256", "384", "512", "1024"], "AE latent dims to process")
flags.DEFINE_list("dataset_sizes", ["50000", "100000", "150000", "200000"], "synthetic dataset sizes to distil on")

flags.DEFINE_integer("student_hidden_channels", 64, "student residual-block channel width")
flags.DEFINE_integer("student_n_blocks", 4, "number of student residual blocks")
flags.DEFINE_integer("latent_embed_dim", 256, "fixed internal latent conditioning width, constant across all latent dims")

flags.DEFINE_integer("total_steps", 20000, "fixed number of optimizer steps to train every student, regardless of dataset size")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_float("lr", 3e-4, "learning rate")
flags.DEFINE_float("weight_decay", 1e-4, "AdamW weight decay")
flags.DEFINE_float("grad_clip", 1.0, "gradient norm clipping")
flags.DEFINE_float("ema_decay", 0.9999, "EMA decay rate")
flags.DEFINE_integer("log_interval", 50, "print/record running loss every N optimizer steps")

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


def build_student(dim, device):
    return StudentDenoiser(
        latent_input_dim=dim,
        latent_embed_dim=FLAGS.latent_embed_dim,
        hidden_channels=FLAGS.student_hidden_channels,
        n_blocks=FLAGS.student_n_blocks,
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


def load_synthetic_pool(synthetic_dir, latent_dim, load_to_ram):
    """Loads generate_teacher_datasets.py's single per-dim (x0, x1, latent)
    pool once. train_students.py then trains each --dataset_sizes experiment
    on a `[0:n]` prefix of this same pool (see NoiseLatentSamplePairs), so
    increasing the dataset size only adds samples rather than resampling."""
    noise_path = synthetic_dir / "noise_fp16.npy"
    images_path = synthetic_dir / "images_uint8.npy"
    latents_path = synthetic_dir / "latents_fp32.npy"
    mmap_mode = None if load_to_ram else "r"
    x0 = np.load(str(noise_path), mmap_mode=mmap_mode)
    x1 = np.load(str(images_path), mmap_mode=mmap_mode)
    latent = np.load(str(latents_path), mmap_mode=mmap_mode)
    if load_to_ram:
        x0 = np.array(x0)
        x1 = np.array(x1)
        latent = np.array(latent)

    if latent.ndim != 2 or latent.shape[1] != latent_dim:
        raise ValueError(
            f"Latent shape mismatch in '{latents_path}': expected (N, {latent_dim}), "
            f"got {tuple(latent.shape)}."
        )
    n = x0.shape[0]
    if len(x1) != n or len(latent) != n:
        raise ValueError(
            f"Sample count mismatch under '{synthetic_dir}': "
            f"x0 has {n}, x1 has {len(x1)}, latent has {len(latent)}."
        )
    return x0, x1, latent


class NoiseLatentSamplePairs(Dataset):
    """A `[0:prefix_size]` prefix of a shared per-dim (x0, x1, latent) pool
    (see load_synthetic_pool) -- the same underlying samples are reused
    across dataset-size experiments so a bigger experiment is a strict
    superset of a smaller one."""

    def __init__(self, x0_pool, x1_pool, latent_pool, prefix_size):
        pool_size = x0_pool.shape[0]
        if prefix_size > pool_size:
            raise ValueError(
                f"Requested dataset size {prefix_size} exceeds the synthetic pool size "
                f"{pool_size} -- regenerate with a larger --dataset_size in "
                f"generate_teacher_datasets.py."
            )
        self.x0 = x0_pool[:prefix_size]
        self.x1 = x1_pool[:prefix_size]
        self.latent = latent_pool[:prefix_size]

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        x0 = torch.from_numpy(np.array(self.x0[idx])).to(torch.float32)
        # Inverts generate_teacher_datasets.py's (x1 * 127.5 + 128).clip(0, 255).to(uint8) encoding.
        x1 = (torch.from_numpy(np.array(self.x1[idx])).to(torch.float32) - 128.0) / 127.5
        latent = torch.from_numpy(np.array(self.latent[idx])).to(torch.float32)
        return x0, x1, latent


def plot_loss(history, dim, n_samples, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [i * FLAGS.log_interval for i in range(1, len(history) + 1)]
    ax.plot(steps, history, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("One-step distillation loss")
    ax.set_title(f"Student Loss  (dim={dim}, n={n_samples:,})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(plots_dir / f"student_loss_{dim}_{n_samples}.png"), dpi=150)
    plt.close()


def _infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


def train_student(dim, n_samples, device, base_dir, x0_pool, x1_pool, latent_pool):
    students_dir = base_dir / "students"
    students_dir.mkdir(parents=True, exist_ok=True)
    out_path = students_dir / f"student_{dim}_{n_samples}.pt"
    if out_path.exists() and not FLAGS.overwrite:
        print(f"  [skip] {out_path.name} already exists.", flush=True)
        return

    dataset = NoiseLatentSamplePairs(x0_pool, x1_pool, latent_pool, n_samples)
    loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    student = build_student(dim, device)
    ema_student = create_ema(student, device)

    print(f"  latent_dim        : {dim}", flush=True)
    print(f"  dataset_size      : {n_samples:,} (prefix of {x0_pool.shape[0]:,}-sample pool)", flush=True)
    print(f"  latent_embed_dim  : {FLAGS.latent_embed_dim}", flush=True)
    print(f"  hidden_channels   : {FLAGS.student_hidden_channels}", flush=True)
    print(f"  n_blocks          : {FLAGS.student_n_blocks}", flush=True)
    print(f"  n_examples        : {len(dataset):,}", flush=True)
    print(f"  student params    : {param_count(student)}", flush=True)
    print(f"  conditioning      : ae_latent", flush=True)
    print(f"  total_steps       : {FLAGS.total_steps:,}", flush=True)

    x0_chk, x1_chk, latent_chk = next(iter(loader))
    print(
        f"  Batch shape check : x0={tuple(x0_chk.shape)}  x1={tuple(x1_chk.shape)}  "
        f"latent={tuple(latent_chk.shape)}",
        flush=True,
    )

    optimizer = AdamW(student.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=FLAGS.total_steps, eta_min=FLAGS.lr * 0.01)

    batches = _infinite_batches(loader)
    history = []
    running_loss = 0.0
    global_step = 0

    student.train()
    with tqdm(total=FLAGS.total_steps, desc=f"    dim={dim} n={n_samples:,}") as pbar:
        while global_step < FLAGS.total_steps:
            x0, x1, latent = next(batches)
            x0 = x0.to(device)
            x1 = x1.to(device)
            latent = latent.to(device)

            v_target = x1 - x0
            v_pred = student(x0, latent)
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), FLAGS.grad_clip)
            optimizer.step()
            scheduler.step()
            update_ema(ema_student, student, FLAGS.ema_decay)

            global_step += 1
            running_loss += loss.item()
            pbar.update(1)

            if global_step % FLAGS.log_interval == 0:
                avg = running_loss / FLAGS.log_interval
                history.append(avg)
                print(f"[dim={dim} n={n_samples} step {global_step:06d}/{FLAGS.total_steps}] loss={avg:.5f}", flush=True)
                running_loss = 0.0

    final_loss = history[-1] if history else float("nan")
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": ema_student.state_dict(),
            "latent_dim": dim,
            "latent_embed_dim": FLAGS.latent_embed_dim,
            "dataset_size": n_samples,
            "n_samples": n_samples,
            "student_hidden_channels": FLAGS.student_hidden_channels,
            "student_n_blocks": FLAGS.student_n_blocks,
            "conditioning": "ae_latent",
            "loss_history": history,
            "total_steps": FLAGS.total_steps,
            "global_step": global_step,
            "final_loss": final_loss,
        },
        tmp_path,
    )
    os.replace(tmp_path, out_path)
    print(f"  Saved -> {out_path} (global_step={global_step}, final_loss={final_loss:.5f})", flush=True)

    plot_loss(history, dim, n_samples, students_dir / "plots")


def main(argv):
    torch.manual_seed(FLAGS.seed)

    dims = [FLAGS.dim] if FLAGS.dim else [int(d) for d in FLAGS.latent_dims]
    sizes = [FLAGS.size] if FLAGS.size else [int(s) for s in FLAGS.dataset_sizes]

    for dim in dims:
        device = get_device(dim)
        print(f"\n{'=' * 60}\nDistilling students  latent_dim={dim}  device={device}\n{'=' * 60}", flush=True)
        base_dir = (Path(FLAGS.output_dir) if FLAGS.output_dir else Path(FLAGS.input_dir)) / f"latent_{dim}" / FLAGS.model
        synthetic_dir = base_dir / "synthetic"
        if not (synthetic_dir / "images.done").exists():
            print(f"  [ERROR] {synthetic_dir} has no completed synthetic pool "
                  f"-- run generate_teacher_datasets.py first.", flush=True)
            continue

        x0_pool, x1_pool, latent_pool = load_synthetic_pool(synthetic_dir, dim, FLAGS.load_to_ram)
        print(f"  Synthetic pool loaded: {x0_pool.shape[0]:,} samples", flush=True)

        for n_samples in sizes:
            train_student(dim, n_samples, device, base_dir, x0_pool, x1_pool, latent_pool)

    print("\nStudent distillation complete.", flush=True)


if __name__ == "__main__":
    app.run(main)
