# Code modified from https://github.com/atong01/conditional-flow-matching/tree/main.


import sys
sys.path.append('./code/cifar10/')

import copy
import math
import os

import torch
from absl import app, flags
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
import argparse
from types import SimpleNamespace
from torchvision.utils import make_grid, save_image
from pl_bolts.models.autoencoders import VAE
from pl_bolts.datamodules import CIFAR10DataModule

print(f"Visible GPUs: {torch.cuda.device_count()}")
print(f"Available GPU Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
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
        return self.model(t,x,y=self.y)[0]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def generate_sample_trajectories(model, parallel, savedir, step, net_="normal",train_sample=None,vae=None):
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
    vae_ = copy.deepcopy(vae)
    train_sample,mean,std = train_sample
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)
        model_.training = True
        x1 = train_sample.to(device)
        mean = mean.to(device)
        std = std.to(device)
        vae_ = vae_.to(device)

    
    traj_id = [j for j in range(0,100,10)]
    with torch.no_grad():
        renormalized_input = x1*0.5 + 0.5
        renormalized_input = (renormalized_input - mean) / std
        latent = vae_.encoder(renormalized_input).view(renormalized_input.size(0),-1)
        # mu = vae_.fc_mu(l0)
        # log_var = vae_.fc_var(l0)
        # _,_,latent = vae_.sample(mu, log_var)
        node_ = NeuralODE(torch_wrapper(model_,y=latent.to(device)), solver="euler", sensitivity="adjoint")
        traj = node_.trajectory(
                torch.randn(10,3,32,32).to(device),
                t_span=torch.linspace(0, 1, 100, device=device),
            )
        traj = traj.transpose(0,1)
        traj = traj[:,traj_id].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    
    save_image(traj, savedir + f"{net_}_generated_KDE_FM_images_step_{step}_vae_cond_kl_rep2.png", nrow=10)

    model.train()

def kl_loss(mu, logvar):
    return -0.5 * (torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)).mean()


def train(rank, total_num_gpus, argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    print("Check 1")
    if FLAGS.parallel and total_num_gpus > 1:
        # When using `DistributedDataParallel`, we need to divide the batch
        # size ourselves based on the total number of GPUs of the current node.
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="/lcrc/project/FastBayes/Anirban_VI/Diffusion_models/data",
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
    print("Check 2")
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

    # Calculate number of epochs
    steps_per_epoch = math.ceil(len(dataset) / FLAGS.batch_size)
    num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)

    print("Check 3")

    print("Check 4")
    ### Load KDE
    seed = 0
    args_data = argparse.Namespace()
    args_data.type = 'cifar10'


    print("Check 5")

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
        num_latents=512,
    ).to(
        rank
    )  # new dropout + bs of 128
    net_model.training = True

    vae = VAE(32, lr=0.00001)
    vae = vae.from_pretrained("cifar10-resnet18").to(rank)
    vae.eval()
    dm = CIFAR10DataModule("/lcrc/project/FastBayes/Anirban_VI/Diffusion_models/data/", normalize=True)
    mean = torch.tensor(dm.default_transforms().transforms[1].mean).to(rank)[None,:,None,None]
    std = torch.tensor(dm.default_transforms().transforms[1].std).to(rank)[None,:,None,None]



    print("Check 6")

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    if FLAGS.restart_dir is not None:
        checkpoint = torch.load(FLAGS.restart_dir, map_location=f"cuda:{rank}")
        try:
            net_model.load_state_dict(checkpoint["net_model"])
        except RuntimeError:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["net_model"].items():
                new_state_dict[k[7:]] = v
            net_model.load_state_dict(new_state_dict)
        try:
            ema_model.load_state_dict(checkpoint["ema_model"])
        except RuntimeError:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["ema_model"].items():
                new_state_dict[k[7:]] = v
            ema_model.load_state_dict(new_state_dict)
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        global_step = checkpoint["step"]

    if FLAGS.parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[rank])
        ema_model = DistributedDataParallel(ema_model, device_ids=[rank])

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

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

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    if FLAGS.restart_dir is not None:
        #global_step = 100000  # Chnage this according to the last run
        num_epochs_run = math.ceil(global_step / steps_per_epoch)
        num_epochs = num_epochs - num_epochs_run
    else:
        global_step = 0
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
                    with torch.no_grad():
                        renormalized_input = x1*0.5 + 0.5
                        renormalized_input = (renormalized_input - mean) / std
                        latent = vae.encoder(renormalized_input).view(renormalized_input.size(0),-1)
                        # mu = vae.fc_mu(l0)
                        # log_var = vae.fc_var(l0)
                        # _,_,latent = vae.sample(mu, log_var)
                    
                    x0 = torch.randn_like(x1)
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                    vt,mu,logvar = net_model(t, xt,y=latent)
                    loss = torch.mean((vt - ut) ** 2) + 0.001*kl_loss(mu, logvar)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
                    optim.step()
                    sched.step()
                    ema(net_model, ema_model, FLAGS.ema_decay)  # new

                    # sample and Saving the weights
                    if (FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0) or global_step == 1:
                        generate_sample_trajectories(
                            net_model, FLAGS.parallel, savedir, global_step, net_="normal", train_sample=[x1[:10],mean,std], vae=vae
                        )
                        generate_sample_trajectories(
                            ema_model, FLAGS.parallel, savedir, global_step, net_="ema", train_sample=[x1[:10],mean,std], vae=vae
                        )
                        torch.save(
                            {
                                "net_model": net_model.state_dict(),
                                "ema_model": ema_model.state_dict(),
                                "sched": sched.state_dict(),
                                "optim": optim.state_dict(),
                                "step": global_step,
                            },
                            savedir + f"Cifar10_weights_step_{global_step}_Lcfm.pt",
                        )



def main(argv):
    # get world size (number of GPUs)
    total_num_gpus = int(os.getenv("WORLD_SIZE", 1))

    if FLAGS.parallel and total_num_gpus > 1:
        train(rank=int(os.getenv("RANK", 0)), total_num_gpus=total_num_gpus, argv=argv)
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train(rank=device, total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)
