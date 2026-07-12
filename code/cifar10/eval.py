# Evaluate teachers and one-step students: generate images, compute FID/IS,
# and plot FID vs latent dim / FID vs synthetic dataset size.
#
# Three phases, each independently restartable (unlike the source pipeline
# this was adapted from, there's no separate "decode" phase: neither the
# teacher -- compute_fid.py's Euler integration happens directly in image
# space -- nor the student -- a single forward pass, x1_pred = x0 +
# student(x0) -- ever produces an intermediate latent to decode):
#
#   --generate  --dim D [--size N | --teacher]  -> {base}/eval/generated/<tag>/*.png
#   --metrics   --dim D [--size N | --teacher]  -> {base}/eval/metrics/metrics_<tag>_dim<D>.json
#   --plot                                      -> {input_dir}/eval_summary/{model}/metrics_all.json + plots/*.png
#
# Omit --size and --teacher to evaluate a student. Pass --teacher to evaluate
# the teacher. Omit --dim/--size to sweep every combination sequentially.
#
# Usage:
#   python code/cifar10/eval.py --generate --dim 128 --size 200000
#   python code/cifar10/eval.py --generate --dim 128 --teacher
#   python code/cifar10/eval.py --metrics  --dim 128 --size 200000 --overwrite
#   python code/cifar10/eval.py --plot

import sys

sys.path.append("./code/cifar10/")
sys.path.append("./code/torchcfm/models/unet/")
sys.path.append("./code/torchcfm/models/")

import json
import shutil
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from PIL import Image
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import tqdm

from unet_resnetVAE import UNetModelWrapper
from conv_autoencoder import ConvAutoencoder
from student_denoiser import load_student
from utils_cifar import infiniteloop

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", "./code/cifar10/runs/", "base dir containing per-dim teacher/student checkpoints")
flags.DEFINE_string("output_dir", None, "base dir to write eval outputs; defaults to --input_dir")
flags.DEFINE_string("model", "icfm", "flow matching model subdirectory (must match training/generation)")

flags.DEFINE_integer("dim", None, "single latent dim to evaluate (omit for all --latent_dims)")
flags.DEFINE_integer("size", None, "single dataset size (student) to evaluate (omit for all --dataset_sizes)")
flags.DEFINE_bool("teacher", False, "evaluate the teacher instead of a student")
flags.DEFINE_list("latent_dims", ["64", "128", "256", "384", "512", "1024"], "AE latent dims to sweep")
flags.DEFINE_list("dataset_sizes", ["50000", "100000", "150000", "200000"], "synthetic dataset sizes to sweep")

flags.DEFINE_string("ae_checkpoint_template", "checkpoints/ae_{dim}.pt", "format string for the AE checkpoint path")
flags.DEFINE_integer("step", None, "load a teacher's Cifar10_weights_step_{step}_latent{dim}_Lcfm.pt")
flags.DEFINE_integer("epoch", None, "load a teacher's Cifar10_weights_epoch_{epoch}_latent{dim}_Lcfm.pt")
flags.DEFINE_bool("latest", False, "load the teacher's latest_latent{dim}_Lcfm.pt (overrides --step/--epoch)")
flags.DEFINE_bool("ema", True, "use the teacher's ema_model weights")
flags.DEFINE_integer("num_channel", 128, "teacher UNet base channel count (must match training)")
flags.DEFINE_integer("unet_latent_dim", 256, "teacher UNet's internal latent bottleneck width (must match training)")

flags.DEFINE_integer("n_samples", 10_000, "number of images to generate for FID/IS (compute_fid.py's single-model default is 50k; this sweeps 30 models so defaults lower)")
flags.DEFINE_integer("integration_steps", 200, "teacher Euler integration steps")
flags.DEFINE_integer("gen_batch_size", 512, "batch size for generation")
flags.DEFINE_integer("num_workers", 2, "dataloader workers")

flags.DEFINE_bool("generate", False, "phase: generate images")
flags.DEFINE_bool("metrics", False, "phase: compute FID/IS")
flags.DEFINE_bool("plot", False, "phase: aggregate metrics + plot")
flags.DEFINE_bool("overwrite", False, "regenerate/recompute outputs that already exist")
flags.DEFINE_integer("seed", 0, "RNG seed")

SIZE_LABELS = {50_000: "50k", 100_000: "100k", 150_000: "150k", 200_000: "200k"}
DIM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SIZE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ── path helpers ──────────────────────────────────────────────────────────────

def _root():
    return Path(FLAGS.output_dir) if FLAGS.output_dir else Path(FLAGS.input_dir)


def base_dir(dim):
    return _root() / f"latent_{dim}" / FLAGS.model


def tag(size):
    return "teacher" if size is None else f"student_n{size}"


def label(size):
    return "teacher" if size is None else SIZE_LABELS[size]


def gen_dir(dim, size):
    return base_dir(dim) / "eval" / "generated" / tag(size)


def ae_recon_dir(dim):
    return _root() / f"latent_{dim}" / "ae_recon"


def metrics_path(dim, size):
    return base_dir(dim) / "eval" / "metrics" / f"metrics_{tag(size)}_dim{dim}.json"


def eval_summary_dir():
    return _root() / "eval_summary" / FLAGS.model


def _is_done(d):
    return (d / ".done").exists()


def _mark_done(d):
    (d / ".done").touch()


def _reset_dir(d):
    """Delete any stale contents (e.g. from a previous --overwrite run with a
    different --n_samples) before regenerating, so leftover PNGs can't
    contaminate the FID/IS computation."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


# ── teacher loading / sampling (duplicated from generate_teacher_datasets.py;
#    see that file for why -- importing it would collide on absl flag names) ──

class torch_wrapper(torch.nn.Module):
    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        if y is not None:
            self.y = y

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x, y=self.y)[0]


def _load_state_dict_tolerant(module, state_dict):
    try:
        module.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        module.load_state_dict(new_state_dict)


def resolve_teacher_checkpoint_path(dim):
    run_dir = base_dir(dim)
    if FLAGS.latest:
        path = run_dir / f"latest_latent{dim}_Lcfm.pt"
    elif FLAGS.step is not None:
        path = run_dir / f"Cifar10_weights_step_{FLAGS.step}_latent{dim}_Lcfm.pt"
    elif FLAGS.epoch is not None:
        path = run_dir / f"Cifar10_weights_epoch_{FLAGS.epoch}_latent{dim}_Lcfm.pt"
    else:
        raise ValueError("One of --latest, --step, --epoch must be given to select a teacher checkpoint.")
    if not path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    return path


def build_teacher(dim, ckpt_path):
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_latents=dim,
        latent_dim=FLAGS.unet_latent_dim,
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["ema_model"] if FLAGS.ema else checkpoint["net_model"]
    _load_state_dict_tolerant(net_model, state_dict)
    net_model.eval()
    net_model.training = False
    return net_model


def build_ae(dim):
    ae_checkpoint_path = FLAGS.ae_checkpoint_template.format(dim=dim)
    ae = ConvAutoencoder(latent_dim=dim).to(device)
    ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device)
    _load_state_dict_tolerant(ae, ae_checkpoint["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def encode_conditioning_latent(net_model, ae, x1):
    with torch.no_grad():
        img = x1 / 2 + 0.5
        latent = ae.encode(img)[0]
        proj = net_model.latent_encodings(latent)
        mu, logvar = proj.chunk(2, dim=1)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    return latent


def teacher_sample(net_model, latent, batch_size, integration_steps):
    node = NeuralODE(torch_wrapper(net_model, y=latent), solver="euler")
    t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
    with torch.no_grad():
        x0 = torch.randn(batch_size, 3, 32, 32, device=device)
        traj = node.trajectory(x0, t_span=t_span)
    return traj[-1].clip(-1, 1)


def make_datalooper(batch_size):
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True, pin_memory=True,
    )
    return infiniteloop(dataloader)


# ── image saving ──────────────────────────────────────────────────────────────

def save_images_uint8(imgs_float, out_dir, start_idx):
    """imgs_float: (N, 3, H, W) tensor in [-1, 1]."""
    imgs_uint8 = (imgs_float * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy()
    imgs_uint8 = imgs_uint8.transpose(0, 2, 3, 1)
    for i, img_arr in enumerate(imgs_uint8):
        Image.fromarray(img_arr).save(out_dir / f"{start_idx + i:05d}.png")
    return len(imgs_uint8)


def ensure_ae_recon(dim, ae):
    """AE reconstruction of the real CIFAR-10 test set, dim-scoped and shared
    across the teacher + all students for that dim -- used for the AE-FID
    upper-bound metric (generation quality is bounded by AE reconstruction
    fidelity, since the teacher conditions on AE-encoded real-image latents).
    """
    out_dir = ae_recon_dir(dim)
    if _is_done(out_dir) and not FLAGS.overwrite:
        return
    _reset_dir(out_dir)

    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.gen_batch_size, shuffle=False, num_workers=FLAGS.num_workers)

    img_idx = 0
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"  [dim={dim}] AE recon", leave=False):
            z = ae.encode(imgs.to(device))[0]
            recon = ae.decode(z).clamp(0, 1).mul(255).byte().cpu().numpy()
            recon = recon.transpose(0, 2, 3, 1)
            for img_arr in recon:
                Image.fromarray(img_arr).save(out_dir / f"{img_idx:05d}.png")
                img_idx += 1
    _mark_done(out_dir)
    print(f"  [dim={dim}] AE recon -> {out_dir}  ({img_idx} files)", flush=True)


# ── phase 1: generate ──────────────────────────────────────────────────────────

def generate(dim, size):
    is_teacher = size is None
    out_dir = gen_dir(dim, size)
    if _is_done(out_dir) and not FLAGS.overwrite:
        print(f"[generate] [skip] dim={dim} {tag(size)} already exists.", flush=True)
        return
    _reset_dir(out_dir)

    ae = build_ae(dim)
    ensure_ae_recon(dim, ae)

    if is_teacher:
        ckpt_path = resolve_teacher_checkpoint_path(dim)
        print(f"[generate] dim={dim} model=teacher ckpt={ckpt_path.name} device={device}", flush=True)
        net_model = build_teacher(dim, ckpt_path)
        datalooper = make_datalooper(FLAGS.gen_batch_size)
    else:
        ckpt_path = base_dir(dim) / "students" / f"student_{dim}_{size}.pt"
        if not ckpt_path.exists():
            print(f"[generate] [ERROR] {ckpt_path} not found -- run train_students.py first.", flush=True)
            return
        print(f"[generate] dim={dim} model=student n={size:,} ckpt={ckpt_path.name} device={device}", flush=True)
        student = load_student(str(ckpt_path), latent_dim=dim, device=str(device))

    generated = 0
    with tqdm(total=FLAGS.n_samples, desc=f"  [dim={dim}] generating {label(size)}") as pbar:
        while generated < FLAGS.n_samples:
            batch_n = min(FLAGS.gen_batch_size, FLAGS.n_samples - generated)
            if is_teacher:
                real_img_batch = next(datalooper)[:batch_n].to(device)
                latent = encode_conditioning_latent(net_model, ae, real_img_batch)
                imgs = teacher_sample(net_model, latent, batch_n, FLAGS.integration_steps)
            else:
                with torch.no_grad():
                    x0 = torch.randn(batch_n, 3, 32, 32, device=device)
                    imgs = (x0 + student(x0)).clip(-1, 1)
            n = save_images_uint8(imgs, out_dir, generated)
            generated += n
            pbar.update(n)

    _mark_done(out_dir)
    print(f"  Saved {generated} images -> {out_dir}", flush=True)


# ── phase 2: metrics ────────────────────────────────────────────────────────────

def compute_fid(gen_dir_str):
    try:
        from cleanfid import fid
        return float(fid.compute_fid(
            gen_dir_str, dataset_name="cifar10", dataset_res=32,
            dataset_split="train", mode="legacy_tensorflow", verbose=False,
        ))
    except ImportError:
        print("  [warning] clean-fid not installed.", flush=True)
        return -1.0


def compute_inception_score(gen_dir_str):
    try:
        import torch_fidelity
        m = torch_fidelity.calculate_metrics(input1=gen_dir_str, isc=True, verbose=False)
        return float(m["inception_score_mean"])
    except ImportError:
        print("  [warning] torch-fidelity not installed.", flush=True)
        return -1.0


def metrics(dim, size):
    is_teacher = size is None
    gdir = gen_dir(dim, size)
    if not _is_done(gdir):
        print(f"[metrics] [ERROR] {gdir} not complete -- run --generate first.", flush=True)
        return

    out = metrics_path(dim, size)
    if out.exists() and not FLAGS.overwrite:
        print(f"[metrics] [skip] {out.name} already exists.", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[metrics] dim={dim} model={label(size)}", flush=True)
    fid_val = compute_fid(str(gdir))
    is_val = compute_inception_score(str(gdir))
    print(f"  FID={fid_val:.2f}  IS={is_val:.2f}", flush=True)

    ae_fid = -1.0
    aedir = ae_recon_dir(dim)
    if _is_done(aedir):
        ae_fid = compute_fid(str(aedir))
        print(f"  AE-FID={ae_fid:.2f}", flush=True)

    ckpt_path_str = str(resolve_teacher_checkpoint_path(dim)) if is_teacher else str(base_dir(dim) / "students" / f"student_{dim}_{size}.pt")

    with open(out, "w") as fh:
        json.dump({
            "fid": fid_val,
            "is": is_val,
            "ae_fid": ae_fid,
            "model": FLAGS.model,
            "latent_dim": dim,
            "model_type": "teacher" if is_teacher else "student",
            "n_samples": None if is_teacher else size,
            "checkpoint_path": ckpt_path_str,
        }, fh, indent=2)
    print(f"  Saved -> {out}", flush=True)


# ── phase 3: plot + unified JSON ─────────────────────────────────────────────

def plot():
    dims = [int(d) for d in FLAGS.latent_dims]
    sizes = [int(s) for s in FLAGS.dataset_sizes]

    student_metrics = {}
    for dim in dims:
        for size in sizes:
            p = metrics_path(dim, size)
            if p.exists():
                with open(p) as fh:
                    student_metrics.setdefault(dim, {})[size] = json.load(fh)
            else:
                print(f"[plot] [warning] {p} missing -- skipping.", flush=True)

    teacher_metrics = {}
    for dim in dims:
        p = metrics_path(dim, None)
        if p.exists():
            with open(p) as fh:
                teacher_metrics[dim] = json.load(fh)
        else:
            print(f"[plot] [warning] {p} missing -- skipping.", flush=True)

    if not student_metrics and not teacher_metrics:
        print("[plot] [ERROR] No metrics found.", flush=True)
        return

    out_dir = eval_summary_dir()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    unified = {
        "teacher": {str(d): v for d, v in teacher_metrics.items()},
        "student": {str(d): {str(s): v for s, v in sv.items()} for d, sv in student_metrics.items()},
    }
    unified_path = out_dir / "metrics_all.json"
    with open(unified_path, "w") as fh:
        json.dump(unified, fh, indent=2)
    print(f"  Unified metrics -> {unified_path}", flush=True)

    # ── FID vs dataset size, one line per dim + teacher dashes ──────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for i, dim in enumerate(dims):
        color = DIM_COLORS[i % len(DIM_COLORS)]
        if dim in student_metrics:
            s_sizes = sorted(student_metrics[dim].keys())
            valid = [(s, student_metrics[dim][s]["fid"]) for s in s_sizes if student_metrics[dim][s]["fid"] >= 0]
            if valid:
                xs, ys = zip(*valid)
                ax1.plot([SIZE_LABELS[x] for x in xs], ys, marker="o", linewidth=2, color=color, label=f"dim={dim}")
        if dim in teacher_metrics and teacher_metrics[dim]["fid"] >= 0:
            ax1.axhline(teacher_metrics[dim]["fid"], color=color, linewidth=1, linestyle="--", alpha=0.6)
    ax1.plot([], [], color="grey", linewidth=1, linestyle="--", alpha=0.6, label="teacher (per dim)")
    ax1.set_xlabel("Synthetic Dataset Size")
    ax1.set_ylabel("FID (↓)")
    ax1.set_title("FID vs Synthetic Dataset Size\n(dashed = teacher baseline per dim)")
    ax1.legend(title="Latent dim", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.grid(True, alpha=0.4)
    plt.tight_layout()
    out1 = plots_dir / "fid_vs_size.png"
    fig1.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Plot saved -> {out1}", flush=True)

    # ── FID vs latent dim, one line per dataset size + teacher (headline plot) ──
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for i, size in enumerate(sizes):
        dims_avail = [d for d in dims if d in student_metrics and size in student_metrics[d]]
        valid = [(d, student_metrics[d][size]["fid"]) for d in dims_avail if student_metrics[d][size]["fid"] >= 0]
        if valid:
            xs, ys = zip(*valid)
            ax2.plot(xs, ys, marker="s", linewidth=2, color=SIZE_COLORS[i % len(SIZE_COLORS)], label=SIZE_LABELS[size])
    teacher_pts = [(d, teacher_metrics[d]["fid"]) for d in dims if d in teacher_metrics and teacher_metrics[d]["fid"] >= 0]
    if teacher_pts:
        tx, ty = zip(*teacher_pts)
        ax2.plot(tx, ty, marker="*", markersize=12, linewidth=2, color="black", linestyle="--", label="teacher")
    ax2.set_xlabel("Latent Dim")
    ax2.set_ylabel("FID (↓)")
    ax2.set_title("FID vs Latent Dim\n(one line per synthetic dataset size + teacher)")
    ax2.legend(title="Dataset size", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    out2 = plots_dir / "fid_vs_dim.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved -> {out2}", flush=True)

    print("\nFID summary  (— = missing)")
    cols = ["teacher"] + [SIZE_LABELS[s] for s in sizes]
    header = f"{'dim':>6}  " + "  ".join(f"{c:>8}" for c in cols)
    print(header)
    for dim in dims:
        t_fid = teacher_metrics.get(dim, {}).get("fid", -1)
        t_str = f"{t_fid:>8.2f}" if t_fid >= 0 else f"{'—':>8}"
        row = f"{dim:>6}  {t_str}"
        for size in sizes:
            fid_val = student_metrics.get(dim, {}).get(size, {}).get("fid", -1)
            row += f"  {fid_val:>8.2f}" if fid_val >= 0 else f"  {'—':>8}"
        print(row)

    print("\nStep 4 complete.", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv):
    torch.manual_seed(FLAGS.seed)

    phases = [FLAGS.generate, FLAGS.metrics, FLAGS.plot]
    if sum(phases) != 1:
        raise ValueError("Exactly one of --generate, --metrics, --plot must be set.")

    if FLAGS.plot:
        plot()
        return

    if FLAGS.teacher and FLAGS.size is not None:
        raise ValueError("--teacher and --size are mutually exclusive.")

    dims = [FLAGS.dim] if FLAGS.dim else [int(d) for d in FLAGS.latent_dims]
    if FLAGS.teacher:
        size_list = [None]
    elif FLAGS.size is not None:
        size_list = [FLAGS.size]
    else:
        size_list = [int(s) for s in FLAGS.dataset_sizes]

    for dim in dims:
        for size in size_list:
            if FLAGS.generate:
                generate(dim, size)
            elif FLAGS.metrics:
                metrics(dim, size)


if __name__ == "__main__":
    app.run(main)
