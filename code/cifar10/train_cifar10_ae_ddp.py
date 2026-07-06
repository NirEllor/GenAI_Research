# Modeled on code/darcy_flow/vae_cifar10_ddp_medium.py, adapted for a deterministic
# (no KL, no reparameterization) autoencoder trained from scratch on CIFAR-10 at a
# configurable flat latent dimension.

import sys
sys.path.append('./code/cifar10/')
sys.path.append('./code/torchcfm/models/StableDiffusion-PyTorch/')

import copy
import math
import os

import torch
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import trange

from utils_cifar import infiniteloop, setup, tile_image
from vae import DeterministicAE, ae_model_config, latent_dim_to_z_channels

print(f"Visible GPUs: {torch.cuda.device_count()}")
print(f"Available GPU Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./results/", help="output directory")
flags.DEFINE_integer("latent_dim", None, help="flat latent dimension of the AE, e.g. 64/128/256/384/512/1024")
flags.mark_flag_as_required("latent_dim")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 400001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_string("master_addr", "localhost", help="master address for Distributed Data Parallel")
flags.DEFINE_string("master_port", "12355", help="master port for Distributed Data Parallel")

# Evaluation
flags.DEFINE_integer("save_step", 20000, help="frequency of saving checkpoints, 0 to disable during training")
flags.DEFINE_string("restart_dir", None, "Checkpoint path to restart training from")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def save_reconstruction_preview(ae, parallel, savedir, step, train_sample):
    """Save a real-vs-reconstruction image grid for a sanity check along training."""
    ae.eval()
    ae_ = copy.deepcopy(ae)
    ae_ = ae_.module if parallel else ae_

    n = min(8, math.isqrt(train_sample.size(0)))
    sample = train_sample[: n * n]
    with torch.no_grad():
        recon, _ = ae_(sample)
        recon = recon.clip(-1, 1) / 2 + 0.5
        real = sample / 2 + 0.5

    save_image(tile_image(real, n), savedir + f"real_step_{step}_latent{FLAGS.latent_dim}.png")
    save_image(tile_image(recon, n), savedir + f"recon_step_{step}_latent{FLAGS.latent_dim}.png")
    ae.train()


def train(rank, total_num_gpus, argv):
    print(
        "lr, total_steps, save_step, latent_dim:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.save_step,
        FLAGS.latent_dim,
    )

    if FLAGS.parallel and total_num_gpus > 1:
        # When using `DistributedDataParallel`, we need to divide the batch
        # size ourselves based on the total number of GPUs of the current node.
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # DATASET/DATALOADER
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
    sampler = DistributedSampler(dataset) if FLAGS.parallel else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=False if FLAGS.parallel else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    datalooper = infiniteloop(dataloader)

    steps_per_epoch = math.ceil(len(dataset) / FLAGS.batch_size)
    num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)

    # MODEL
    z_channels = latent_dim_to_z_channels(FLAGS.latent_dim)
    model_config = ae_model_config(z_channels)
    ae = DeterministicAE(im_channels=3, model_config=model_config).to(rank)

    optim = torch.optim.Adam(ae.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    global_step = 0
    if FLAGS.restart_dir is not None:
        checkpoint = torch.load(
            FLAGS.restart_dir, map_location=f"cuda:{rank}" if use_cuda else "cpu"
        )
        try:
            ae.load_state_dict(checkpoint["ae"])
        except RuntimeError:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["ae"].items():
                new_state_dict[k[7:]] = v
            ae.load_state_dict(new_state_dict)
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        global_step = checkpoint["step"]

    if FLAGS.parallel:
        ae = DistributedDataParallel(ae, device_ids=[rank])

    # show model size
    ae_size = sum(param.data.nelement() for param in ae.parameters())
    print("AE params: %.2f M" % (ae_size / 1e6))

    savedir = FLAGS.output_dir + f"ae_latent{FLAGS.latent_dim}/"
    os.makedirs(savedir, exist_ok=True)

    if FLAGS.restart_dir is not None:
        num_epochs_run = math.ceil(global_step / steps_per_epoch)
        num_epochs = num_epochs - num_epochs_run

    with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            if sampler is not None:
                sampler.set_epoch(epoch)

            with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                for step in step_pbar:
                    global_step += 1

                    optim.zero_grad()
                    x1 = next(datalooper).to(rank)
                    out, z = ae(x1)
                    loss = torch.mean((x1 - out) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ae.parameters(), FLAGS.grad_clip)
                    optim.step()
                    sched.step()
                    step_pbar.set_postfix(loss=loss.item())

                    # sample and saving the weights
                    if (FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0) or global_step == 1:
                        save_reconstruction_preview(
                            ae, FLAGS.parallel, savedir, global_step, x1
                        )
                        torch.save(
                            {
                                "ae": ae.state_dict(),
                                "sched": sched.state_dict(),
                                "optim": optim.state_dict(),
                                "step": global_step,
                            },
                            savedir + f"cifar10_ae_weights_step_{global_step}_latent{FLAGS.latent_dim}.pt",
                        )


def main(argv):
    # get world size (number of GPUs)
    total_num_gpus = int(os.getenv("WORLD_SIZE", 1))

    if FLAGS.parallel and total_num_gpus > 1:
        train(rank=int(os.getenv("RANK", 0)), total_num_gpus=total_num_gpus, argv=argv)
    else:
        train(rank=device, total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)


'''
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train_cifar10_ae_ddp.py \
  --latent_dim 256 \
  --output_dir "./code/cifar10/runs/" \
  --lr 2e-4 \
  --batch_size 128 \
  --num_workers 4 \
  --total_steps 100001 \
  --save_step 10000 \
  --parallel True \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT
'''
