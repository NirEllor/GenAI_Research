# Code modified from https://github.com/atong01/conditional-flow-matching/tree/main.

# Authors: Anirban Samaddar
import copy
import os

import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from KNIFE_Org_func import sample, base_mean_sampler

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

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
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()


def generate_sample_trajectories(model, parallel, savedir, step, net_="normal",base_dist=None, residual=False):
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
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    traj_id = [j for j in range(0,100,10)]
    with torch.no_grad():
        if not residual:
            traj = node_.trajectory(
                torch.randn(10, 3, 32, 32, device=device) if base_dist is None else sample((10,),base_dist).to(device).view(-1,3,32,32),
                t_span=torch.linspace(0, 1, 100, device=device),
            )
        else:
            x0 = base_mean_sampler((10,),base_dist).to(device).view(-1,3,32,32)
            traj = node_.trajectory(
                torch.randn(10, 3, 32, 32, device=device),
                t_span=torch.linspace(0, 1, 100, device=device),
            )
            traj = x0[None, ...] + traj
        traj = traj.transpose(0,1)
        traj = traj[:,traj_id].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    
    if not residual:
        save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=10) if base_dist is None else save_image(traj, savedir + f"{net_}_generated_KDE_FM_images_step_{step}.png", nrow=10)
    else:
        save_image(traj, savedir + f"{net_}_generated_KDE_FM_images_step_{step}_residual.png", nrow=10)

    model.train()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        if y is not None:
            self.y = y

    def forward(self, t, x, *args, **kwargs):
        return self.model(t,x,y=self.y)


def generate_sample_trajectories_class_cond(model, parallel, savedir, step, net_="normal",base_dist=None):
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
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    
    traj_id = [j for j in range(0,100,10)]
    with torch.no_grad():
        x0,I = sample(10,base_dist)
        node_ = NeuralODE(torch_wrapper(model_,y=I), solver="euler", sensitivity="adjoint")
        traj = node_.trajectory(
                x0.to(device).view(-1,3,32,32),
                t_span=torch.linspace(0, 1, 100, device=device),
            )
        traj = traj.transpose(0,1)
        traj = traj[:,traj_id].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    
    save_image(traj, savedir + f"{net_}_generated_KDE_FM_images_step_{step}_dispatcher.png", nrow=10)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader, retrun_class = False):
    while True:
        for x, y in iter(dataloader):
            yield x if not retrun_class else (x,y)
