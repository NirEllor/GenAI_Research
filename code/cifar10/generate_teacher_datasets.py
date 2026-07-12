# Generate synthetic datasets from trained Latent-CFM teacher models
# (code/cifar10/train_cifar10_ddp_vae_cond_ic.py), for later distillation into
# smaller "student" models. For each AE latent dim, produces:
#   - one independent synthetic image dataset per --dataset_sizes entry
#     (final Euler-integrated samples + the conditioning latent used per sample)
#   - one trajectory dataset (a small set of samples with several intermediate
#     Euler states each, for future students trained on "middle-time" targets)
#
# Usage:
#   python code/cifar10/generate_teacher_datasets.py \
#     --input_dir ./code/cifar10/runs/ --model icfm --latest True
#
#   python code/cifar10/generate_teacher_datasets.py \
#     --latent_dims 256 --step 600000 --dataset_sizes 50000,100000

import json
import sys

sys.path.append("./code/cifar10/")
sys.path.append("./code/torchcfm/models/unet/")
sys.path.append("./code/torchcfm/models/")

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms

from utils_cifar import infiniteloop
from unet_resnetVAE import UNetModelWrapper
from conv_autoencoder import ConvAutoencoder

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", "./code/cifar10/runs/", "base dir containing per-dim teacher checkpoints")
flags.DEFINE_string("output_dir", None, "base dir to write synthetic datasets; defaults to --input_dir")
flags.DEFINE_string("model", "icfm", "flow matching model subdirectory (must match --model at train time)")
flags.DEFINE_list("latent_dims", ["64", "128", "256", "384", "512", "1024"], "AE latent dims to process")
flags.DEFINE_string("ae_checkpoint_template", "checkpoints/ae_{dim}.pt", "format string for the AE checkpoint path")

flags.DEFINE_integer("step", None, "load Cifar10_weights_step_{step}_latent{dim}_Lcfm.pt")
flags.DEFINE_integer("epoch", None, "load Cifar10_weights_epoch_{epoch}_latent{dim}_Lcfm.pt")
flags.DEFINE_bool("latest", False, "load latest_latent{dim}_Lcfm.pt (overrides --step/--epoch)")
flags.DEFINE_bool("ema", True, "use ema_model weights instead of net_model")

flags.DEFINE_integer("num_channel", 128, "UNet base channel count (must match training)")
flags.DEFINE_integer("unet_latent_dim", 256, "UNet's own internal latent bottleneck width (must match training)")

flags.DEFINE_list("dataset_sizes", ["50000", "100000", "150000", "200000"], "sizes of independent image datasets to generate per dim")
flags.DEFINE_integer("gen_batch_size", 500, "batch size used for Euler sampling")
flags.DEFINE_integer("integration_steps", 100, "Euler integration steps (fixed grid)")

flags.DEFINE_integer("traj_num_samples", 2000, "sample count for the trajectory dataset")
flags.DEFINE_integer("traj_frame_stride", 10, "keep every Nth Euler step, plus always the final step")

flags.DEFINE_bool("overwrite", False, "regenerate outputs that already exist")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_integer("num_workers", 4, "CIFAR10 dataloader workers")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn-compatible format (copied from compute_fid.py)."""

    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        if y is not None:
            self.y = y

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x, y=self.y)[0]


def _load_state_dict_tolerant(module, state_dict):
    """Load a state dict, stripping a DistributedDataParallel 'module.' prefix on failure."""
    try:
        module.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        module.load_state_dict(new_state_dict)


def teacher_run_dir(latent_dim):
    return Path(FLAGS.input_dir) / f"latent_{latent_dim}" / FLAGS.model


def resolve_checkpoint_path(latent_dim):
    run_dir = teacher_run_dir(latent_dim)
    if FLAGS.latest:
        path = run_dir / f"latest_latent{latent_dim}_Lcfm.pt"
    elif FLAGS.step is not None:
        path = run_dir / f"Cifar10_weights_step_{FLAGS.step}_latent{latent_dim}_Lcfm.pt"
    elif FLAGS.epoch is not None:
        path = run_dir / f"Cifar10_weights_epoch_{FLAGS.epoch}_latent{latent_dim}_Lcfm.pt"
    else:
        raise ValueError("One of --latest, --step, --epoch must be given to select a teacher checkpoint.")
    if not path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    return path


def build_teacher(latent_dim, ckpt_path):
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_latents=latent_dim,
        latent_dim=FLAGS.unet_latent_dim,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["ema_model"] if FLAGS.ema else checkpoint["net_model"]
    _load_state_dict_tolerant(net_model, state_dict)
    net_model.eval()
    net_model.training = False

    ae_checkpoint_path = FLAGS.ae_checkpoint_template.format(dim=latent_dim)
    ae = ConvAutoencoder(latent_dim=latent_dim).to(device)
    ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device)
    _load_state_dict_tolerant(ae, ae_checkpoint["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    print(f"[dim={latent_dim}] loaded teacher from {ckpt_path} (ema={FLAGS.ema}), AE from {ae_checkpoint_path}", flush=True)
    return net_model, ae


def make_datalooper(batch_size):
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return infiniteloop(dataloader)


def encode_conditioning_latent(net_model, ae, x1):
    """Reparameterized conditioning latent for a real CIFAR batch x1 in [-1,1]
    (copied from compute_fid.py's gen_1_img)."""
    with torch.no_grad():
        img = x1 / 2 + 0.5  # AE trained on [0,1] images
        latent = ae.encode(img)[0]
        proj = net_model.latent_encodings(latent)
        mu, logvar = proj.chunk(2, dim=1)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    return latent


def euler_trajectory(net_model, latent, batch_size, integration_steps):
    """Full Euler trajectory from image noise to a generated sample, shape
    (integration_steps+1, batch_size, 3, 32, 32)."""
    node = NeuralODE(torch_wrapper(net_model, y=latent), solver="euler")
    t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
    with torch.no_grad():
        x0 = torch.randn(batch_size, 3, 32, 32, device=device)
        traj = node.trajectory(x0, t_span=t_span)
    return traj


def _is_done(out_dir, tag):
    return (out_dir / f"{tag}.done").exists()


def _mark_done(out_dir, tag):
    (out_dir / f"{tag}.done").touch()


def generate_image_dataset(dim, size, net_model, ae, datalooper, out_dir):
    tag = f"images_n{size}"
    if _is_done(out_dir, tag) and not FLAGS.overwrite:
        print(f"  [skip] {tag} already exists.", flush=True)
        return

    images_path = out_dir / f"images_uint8_n{size}.npy.tmp"
    latents_path = out_dir / f"latents_fp32_n{size}.npy.tmp"
    images_out = np.lib.format.open_memmap(str(images_path), mode="w+", dtype=np.uint8, shape=(size, 3, 32, 32))
    latents_out = np.lib.format.open_memmap(str(latents_path), mode="w+", dtype=np.float32, shape=(size, dim))

    generated = 0
    while generated < size:
        batch_n = min(FLAGS.gen_batch_size, size - generated)
        x1 = next(datalooper)[:batch_n].to(device)
        latent = encode_conditioning_latent(net_model, ae, x1)
        traj = euler_trajectory(net_model, latent, batch_n, FLAGS.integration_steps)
        final = traj[-1].clip(-1, 1)
        img_uint8 = (final * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy()

        images_out[generated:generated + batch_n] = img_uint8
        latents_out[generated:generated + batch_n] = latent.detach().cpu().numpy()
        generated += batch_n
        print(f"  [dim={dim}] {tag}: {generated}/{size}", flush=True)

    images_out.flush()
    latents_out.flush()
    del images_out, latents_out
    final_images_path = out_dir / f"images_uint8_n{size}.npy"
    final_latents_path = out_dir / f"latents_fp32_n{size}.npy"
    os.replace(images_path, final_images_path)
    os.replace(latents_path, final_latents_path)
    _mark_done(out_dir, tag)
    print(f"  [dim={dim}] saved {tag} -> {final_images_path}", flush=True)


def generate_trajectory_dataset(dim, net_model, ae, datalooper, out_dir):
    tag = "trajectory"
    if _is_done(out_dir, tag) and not FLAGS.overwrite:
        print(f"  [skip] {tag} already exists.", flush=True)
        return

    frame_indices = list(range(0, FLAGS.integration_steps, FLAGS.traj_frame_stride))
    if frame_indices[-1] != FLAGS.integration_steps:
        frame_indices.append(FLAGS.integration_steps)
    num_frames = len(frame_indices)

    images_path = out_dir / "trajectory_images_fp16.npy.tmp"
    latents_path = out_dir / "trajectory_latents_fp32.npy.tmp"
    images_out = np.lib.format.open_memmap(
        str(images_path), mode="w+", dtype=np.float16, shape=(FLAGS.traj_num_samples, num_frames, 3, 32, 32)
    )
    latents_out = np.lib.format.open_memmap(
        str(latents_path), mode="w+", dtype=np.float32, shape=(FLAGS.traj_num_samples, dim)
    )

    generated = 0
    while generated < FLAGS.traj_num_samples:
        batch_n = min(FLAGS.gen_batch_size, FLAGS.traj_num_samples - generated)
        x1 = next(datalooper)[:batch_n].to(device)
        latent = encode_conditioning_latent(net_model, ae, x1)
        traj = euler_trajectory(net_model, latent, batch_n, FLAGS.integration_steps)
        # traj: (integration_steps+1, batch_n, 3, 32, 32) -> keep stride frames, unclipped
        frames = traj[frame_indices].transpose(0, 1).to(torch.float16).cpu().numpy()

        images_out[generated:generated + batch_n] = frames
        latents_out[generated:generated + batch_n] = latent.detach().cpu().numpy()
        generated += batch_n
        print(f"  [dim={dim}] {tag}: {generated}/{FLAGS.traj_num_samples}", flush=True)

    images_out.flush()
    latents_out.flush()
    del images_out, latents_out
    final_images_path = out_dir / "trajectory_images_fp16.npy"
    final_latents_path = out_dir / "trajectory_latents_fp32.npy"
    os.replace(images_path, final_images_path)
    os.replace(latents_path, final_latents_path)

    with open(out_dir / "trajectory_frames.json", "w") as f:
        json.dump(
            {"integration_steps": FLAGS.integration_steps, "frame_indices": frame_indices,
             "t_values": [i / FLAGS.integration_steps for i in frame_indices]},
            f, indent=2,
        )
    _mark_done(out_dir, tag)
    print(f"  [dim={dim}] saved {tag} -> {final_images_path}", flush=True)


def write_manifest(dim, out_dir, ckpt_path):
    manifest = {
        "latent_dim": dim,
        "model": FLAGS.model,
        "teacher_checkpoint": str(ckpt_path),
        "ema": FLAGS.ema,
        "integration_steps": FLAGS.integration_steps,
        "gen_batch_size": FLAGS.gen_batch_size,
        "seed": FLAGS.seed,
        "dataset_sizes": [int(s) for s in FLAGS.dataset_sizes],
        "traj_num_samples": FLAGS.traj_num_samples,
        "traj_frame_stride": FLAGS.traj_frame_stride,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def generate_for_dim(dim):
    print(f"\n{'=' * 60}\nGenerating synthetic data  latent_dim={dim}  device={device}\n{'=' * 60}", flush=True)

    ckpt_path = resolve_checkpoint_path(dim)
    net_model, ae = build_teacher(dim, ckpt_path)

    out_dir = (Path(FLAGS.output_dir) if FLAGS.output_dir else Path(FLAGS.input_dir)) / f"latent_{dim}" / FLAGS.model / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    datalooper = make_datalooper(FLAGS.gen_batch_size)

    for size in FLAGS.dataset_sizes:
        size = int(size)
        print(f"  Generating independent dataset  n={size:,} ...", flush=True)
        generate_image_dataset(dim, size, net_model, ae, datalooper, out_dir)

    print(f"  Generating trajectory dataset  n={FLAGS.traj_num_samples:,}  steps={FLAGS.integration_steps} ...", flush=True)
    generate_trajectory_dataset(dim, net_model, ae, datalooper, out_dir)

    write_manifest(dim, out_dir, ckpt_path)


def main(argv):
    torch.manual_seed(FLAGS.seed)
    for dim in FLAGS.latent_dims:
        generate_for_dim(int(dim))
    print("\nSynthetic dataset generation complete.", flush=True)


if __name__ == "__main__":
    app.run(main)
