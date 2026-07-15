# Code modified from https://github.com/atong01/conditional-flow-matching/tree/main.


import sys
sys.path.append('./code/cifar10/')

import copy
import math
import os
from collections import OrderedDict

import torch
from absl import app, flags
from cleanfid import fid
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop, setup, generate_sample_trajectories

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
sys.path.append('./code/torchcfm/models/unet/')
from unet_resnetVAE import UNetModelWrapper
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace
from torchvision.utils import make_grid, save_image
sys.path.append('./code/torchcfm/models/')
from conv_autoencoder import ConvAutoencoder


FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Deterministic AE (latent conditioning source)
flags.DEFINE_integer(
    "latent_dim", None, help="flat latent dimension of the AE, e.g. 64/128/256/384/512/1024"
)
flags.DEFINE_string(
    "ae_checkpoint", None, help="path to an ae_<dim>.pt checkpoint (dict with 'latent_dim'/'state_dict' keys)"
)
flags.mark_flag_as_required("latent_dim")
flags.mark_flag_as_required("ae_checkpoint")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("max_epochs", 1000, help="hard cap on number of epochs, regardless of total_steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_string(
    "master_addr", "localhost", help="master address for Distributed Data Parallel"
)
flags.DEFINE_string("master_port", "12355", help="master port for Distributed Data Parallel")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_integer("log_every_steps", 100, help="print loss/best_loss every N steps")
flags.DEFINE_bool("eval_fid", True, help="periodically compute FID during training (on ema_model)")
flags.DEFINE_integer("fid_every_epochs", 50, help="compute FID every N epochs (requires --eval_fid)")
flags.DEFINE_integer("fid_num_gen", 5000, help="number of generated images used for the periodic FID estimate")
flags.DEFINE_integer(
    "fid_batch_size", 250, help="batch size used when generating images for the periodic FID estimate"
)
flags.DEFINE_integer(
    "fid_integration_steps",
    50,
    help="Euler integration steps for the periodic FID estimate (fewer = faster, less accurate than compute_fid.py)",
)

flags.DEFINE_string("restart_dir", None, "Directory to restart training from")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        if y is not None:
            self.y = y

    def forward(self, t, x, *args, **kwargs):
        return self.model(t,x,y=self.y)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def generate_sample_trajectories(model, parallel, savedir, step, net_="normal",train_sample=None,ae=None):
    """Save 10 generated images trajectories for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    ae_ = copy.deepcopy(ae)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)
        x1 = train_sample.to(device)
        ae_ = ae_.to(device)
    else:
        x1 = train_sample

    traj_id = [j for j in range(0,100,10)]
    with torch.no_grad():
        latent = ae_.encode(x1 / 2 + 0.5)[0]  # AE trained on [0,1] images, x1 is [-1,1]
        node_ = NeuralODE(torch_wrapper(model_,y=latent.to(device)), solver="euler", sensitivity="adjoint")
        traj = node_.trajectory(
                torch.randn(10,3,32,32).to(device),
                t_span=torch.linspace(0, 1, 100, device=device),
            )
        traj = traj.transpose(0,1)
        traj = traj[:,traj_id].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5

    save_image(traj, savedir + f"{net_}_generated_KDE_FM_images_step_{step}_ae_cond_latent{FLAGS.latent_dim}.png", nrow=10)

    model.train()



def compute_train_fid(model, ae, fid_datalooper, parallel, num_gen, batch_size, integration_steps):
    """Rough FID estimate for periodic training-time monitoring (not a substitute
    for the full evaluation in compute_fid.py: fewer generated images, fewer
    integration steps, and it runs on every rank when --parallel is set).
    """
    model.eval()
    model_ = copy.deepcopy(model)
    if parallel:
        model_ = model_.module.to(device)

    def gen_fn(unused_latent):
        with torch.no_grad():
            x1 = next(fid_datalooper).to(device)
            latent = ae.encode(x1 / 2 + 0.5)[0]  # AE trained on [0,1] images, x1 is [-1,1]
            node_ = NeuralODE(torch_wrapper(model_, y=latent), solver="euler")
            t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
            traj = node_.trajectory(torch.randn(x1.size(0), 3, 32, 32, device=device), t_span=t_span)
        img = traj[-1]
        return (img * 127.5 + 128).clip(0, 255).to(torch.uint8)

    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        mode="legacy_tensorflow",
        num_gen=num_gen,
        batch_size=batch_size,
    )
    model.train()
    return score


def _strip_ddp_prefix(state_dict):
    """Strip a DistributedDataParallel 'module.' prefix from checkpoint keys."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    return new_state_dict


def _load_module_state(module, state_dict, module_name, ckpt_path, latent_dim):
    try:
        module.load_state_dict(state_dict)
    except RuntimeError:
        try:
            module.load_state_dict(_strip_ddp_prefix(state_dict))
        except RuntimeError as e2:
            raise RuntimeError(
                f"[dim={latent_dim}] Failed to load '{module_name}' state dict from "
                f"checkpoint '{ckpt_path}' into a model built with latent_dim={latent_dim}. "
                f"This usually means the checkpoint was trained with a different "
                f"architecture or --latent_dim. Original error: {e2}"
            ) from e2


def load_checkpoint_state(path, net_model, ema_model, optim, sched, device, latent_dim, steps_per_epoch):
    """Load training state from a checkpoint file into net_model/ema_model/optim/sched
    (in place) and return (global_step, start_epoch, best_loss, best_fid). Tolerant of
    older checkpoints missing 'epoch'/'best_loss'/'best_fid'/'latent_dim'.
    """
    checkpoint = torch.load(path, map_location=device)

    ckpt_conditioning = checkpoint.get("conditioning")
    if ckpt_conditioning != "deterministic_ae_latent":
        raise RuntimeError(
            f"Checkpoint '{path}' has conditioning={ckpt_conditioning!r}, "
            f"but this script requires conditioning='deterministic_ae_latent'. "
            f"The checkpoint may be from the old variational-latent code path (before the refactor) "
            f"and cannot be loaded here."
        )

    ckpt_latent_dim = checkpoint.get("latent_dim")
    if ckpt_latent_dim is not None and ckpt_latent_dim != latent_dim:
        raise RuntimeError(
            f"Checkpoint '{path}' was saved with latent_dim={ckpt_latent_dim}, "
            f"but the current run uses --latent_dim={latent_dim}."
        )

    _load_module_state(net_model, checkpoint["net_model"], "net_model", path, latent_dim)
    _load_module_state(ema_model, checkpoint["ema_model"], "ema_model", path, latent_dim)
    optim.load_state_dict(checkpoint["optim"])
    sched.load_state_dict(checkpoint["sched"])

    global_step = checkpoint["step"]
    start_epoch = checkpoint.get("epoch")
    if start_epoch is None:
        start_epoch = global_step // steps_per_epoch
    best_loss = checkpoint.get("best_loss", float("inf"))
    best_fid = checkpoint.get("best_fid", float("inf"))

    return global_step, start_epoch, best_loss, best_fid


def build_checkpoint_dict(net_model, ema_model, optim, sched, global_step, epoch, best_loss, best_fid, latent_dim):
    """Assemble the standard checkpoint dict shared by all save sites."""
    return {
        "net_model": net_model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "sched": sched.state_dict(),
        "optim": optim.state_dict(),
        "step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "best_fid": best_fid,
        "latent_dim": latent_dim,
        "conditioning": "deterministic_ae_latent",
    }


def train(rank, total_num_gpus, argv):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and rank != "cpu" else "cpu")
    print(
        f"[dim={FLAGS.latent_dim}] lr={FLAGS.lr} total_steps={FLAGS.total_steps} "
        f"max_epochs={FLAGS.max_epochs} ema_decay={FLAGS.ema_decay} save_step={FLAGS.save_step}",
        flush=True,
    )

    if FLAGS.parallel and total_num_gpus > 1:
        # When using `DistributedDataParallel`, we need to divide the batch
        # size ourselves based on the total number of GPUs of the current node.
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    print(f"[dim={FLAGS.latent_dim}] CIFAR10 dataset loaded: {len(dataset)} train images", flush=True)
    sampler = DistributedSampler(dataset) if FLAGS.parallel else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=False if FLAGS.parallel else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True
    )

    datalooper = infiniteloop(dataloader)

    fid_datalooper = None
    if FLAGS.eval_fid:
        fid_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        fid_dataloader = torch.utils.data.DataLoader(
            fid_dataset,
            batch_size=FLAGS.fid_batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )
        fid_datalooper = infiniteloop(fid_dataloader)

    # Calculate number of epochs
    steps_per_epoch = math.ceil(len(dataset) / FLAGS.batch_size)
    num_epochs = min(math.ceil(FLAGS.total_steps / steps_per_epoch), FLAGS.max_epochs)
    print(
        f"[dim={FLAGS.latent_dim}] steps_per_epoch={steps_per_epoch} num_epochs={num_epochs} "
        f"(capped at max_epochs={FLAGS.max_epochs})",
        flush=True,
    )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    latest_ckpt_path = savedir + f"latest_latent{FLAGS.latent_dim}_Lcfm_det.pt"
    resume_ckpt_path = FLAGS.restart_dir
    auto_resumed = False
    if resume_ckpt_path is None and os.path.exists(latest_ckpt_path):
        latest_ckpt_meta = torch.load(latest_ckpt_path, map_location="cpu")
        completed_epoch = latest_ckpt_meta.get("epoch", 0)
        if completed_epoch >= num_epochs:
            print(
                f"[dim={FLAGS.latent_dim}] found completed checkpoint at {latest_ckpt_path} "
                f"(epoch {completed_epoch}/{num_epochs}) — skipping training.",
                flush=True,
            )
            return
        resume_ckpt_path = latest_ckpt_path
        auto_resumed = True

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_latents=FLAGS.latent_dim,
    ).to(
        rank
    )  # new dropout + bs of 128
    net_model.train()

    ae = ConvAutoencoder(latent_dim=FLAGS.latent_dim).to(device)
    ae_checkpoint = torch.load(
        FLAGS.ae_checkpoint,
        map_location=device
    )
    try:
        ae.load_state_dict(ae_checkpoint["state_dict"])
    except RuntimeError:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in ae_checkpoint["state_dict"].items():
            new_state_dict[k[7:]] = v
        ae.load_state_dict(new_state_dict)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    print(f"[dim={FLAGS.latent_dim}] AE checkpoint loaded from {FLAGS.ae_checkpoint}", flush=True)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    if resume_ckpt_path is not None:
        global_step, start_epoch, best_loss, best_fid = load_checkpoint_state(
            resume_ckpt_path, net_model, ema_model, optim, sched, device,
            FLAGS.latent_dim, steps_per_epoch,
        )
        resume_kind = "auto-detected latest checkpoint" if auto_resumed else "--restart_dir"
        print(
            f"[dim={FLAGS.latent_dim}] Resuming from {resume_ckpt_path} ({resume_kind}): "
            f"epoch={start_epoch}, step={global_step}, "
            f"best_loss={best_loss:.6f}, best_fid={best_fid:.4f}",
            flush=True,
        )
    else:
        global_step = 0
        start_epoch = 0
        best_loss = float("inf")
        best_fid = float("inf")

    if FLAGS.parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[rank])
        ema_model = DistributedDataParallel(ema_model, device_ids=[rank])

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print(f"[dim={FLAGS.latent_dim}] Model params: {model_size / 1024 / 1024:.2f} M", flush=True)

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    with trange(start_epoch, num_epochs, dynamic_ncols=True) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"[dim={FLAGS.latent_dim}] Epoch {epoch + 1}/{num_epochs}")
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss_sum = 0.0
            with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                for step in step_pbar:
                    global_step += 1

                    optim.zero_grad()
                    x1 = next(datalooper).to(device)
                    with torch.no_grad():
                        latent = ae.encode(x1 / 2 + 0.5)[0]  # AE trained on [0,1] images, x1 is [-1,1]

                    x0 = torch.randn_like(x1)
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                    vt = net_model(t, xt, y=latent)
                    loss = torch.mean((vt - ut) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
                    optim.step()
                    sched.step()
                    ema(net_model, ema_model, FLAGS.ema_decay)  # new

                    loss_value = loss.item()
                    epoch_loss_sum += loss_value
                    if loss_value < best_loss:
                        best_loss = loss_value
                    step_pbar.set_postfix(loss=f"{loss_value:.4f}", best_loss=f"{best_loss:.4f}")
                    if global_step % FLAGS.log_every_steps == 0 or global_step == 1:
                        print(
                            f"[dim={FLAGS.latent_dim}] [step {global_step}] loss={loss_value:.6f} best_loss={best_loss:.6f}",
                            flush=True,
                        )

                    # sample and Saving the weights
                    if (FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0) or global_step == 1:
                        print(
                            f"[dim={FLAGS.latent_dim}] [step {global_step}] saving checkpoint + generating sample trajectories",
                            flush=True,
                        )
                        generate_sample_trajectories(
                            net_model, FLAGS.parallel, savedir, global_step, net_="normal", train_sample=x1[:10], ae=ae
                        )
                        generate_sample_trajectories(
                            ema_model, FLAGS.parallel, savedir, global_step, net_="ema", train_sample=x1[:10], ae=ae
                        )
                        torch.save(
                            build_checkpoint_dict(
                                net_model, ema_model, optim, sched,
                                global_step, epoch + 1, best_loss, best_fid, FLAGS.latent_dim,
                            ),
                            savedir + f"Cifar10_weights_step_{global_step}_latent{FLAGS.latent_dim}_Lcfm_det.pt",
                        )

            epoch_avg_loss = epoch_loss_sum / steps_per_epoch
            print(
                f"[dim={FLAGS.latent_dim}] [epoch {epoch + 1}/{num_epochs}] step={global_step} "
                f"avg_loss={epoch_avg_loss:.6f} best_loss={best_loss:.6f}",
                flush=True,
            )

            if (epoch + 1) % FLAGS.fid_every_epochs == 0:
                if FLAGS.eval_fid:
                    fid_score = compute_train_fid(
                        ema_model,
                        ae,
                        fid_datalooper,
                        FLAGS.parallel,
                        num_gen=FLAGS.fid_num_gen,
                        batch_size=FLAGS.fid_batch_size,
                        integration_steps=FLAGS.fid_integration_steps,
                    )
                    if fid_score < best_fid:
                        best_fid = fid_score
                    print(
                        f"[dim={FLAGS.latent_dim}] [epoch {epoch + 1}/{num_epochs}] step={global_step} "
                        f"FID({FLAGS.fid_num_gen})={fid_score:.4f} best_FID={best_fid:.4f}",
                        flush=True,
                    )

                print(
                    f"[dim={FLAGS.latent_dim}] [epoch {epoch + 1}/{num_epochs}] saving periodic checkpoint",
                    flush=True,
                )
                torch.save(
                    build_checkpoint_dict(
                        net_model, ema_model, optim, sched,
                        global_step, epoch + 1, best_loss, best_fid, FLAGS.latent_dim,
                    ),
                    savedir + f"Cifar10_weights_epoch_{epoch + 1}_latent{FLAGS.latent_dim}_Lcfm_det.pt",
                )

            torch.save(
                build_checkpoint_dict(
                    net_model, ema_model, optim, sched,
                    global_step, epoch + 1, best_loss, best_fid, FLAGS.latent_dim,
                ),
                savedir + f"latest_latent{FLAGS.latent_dim}_Lcfm_det.pt",
            )



def main(argv):
    total_num_gpus = int(os.getenv("WORLD_SIZE", 1))

    if FLAGS.parallel and total_num_gpus > 1:
        train(rank=int(os.getenv("RANK", 0)), total_num_gpus=total_num_gpus, argv=argv)
    else:
        train(rank=0 if torch.cuda.is_available() else "cpu", total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)
