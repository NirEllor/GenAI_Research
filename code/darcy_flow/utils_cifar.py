import copy
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE
import os.path as osp
# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Darcy_Dataset(Dataset):
    def __init__(self, path):
        self.root = path

        # load the sample names
        sample_names = os.listdir(osp.join(path, "data"))
        self.P_names, self.U1_names, self.U2_names = self.seperate_img_names(sample_names)
        self.P_names.sort()
        self.U1_names.sort()
        self.U2_names.sort() # all files are stored as P_xxx.npy, U1_xxx.npy, U2_xxx.npy
        self.img_mean = np.array([0, 0.194094975, 0.115737872]) # P, U1, U2
        self.img_std = np.array([0.08232874, 0.27291843, 0.12989907])

        # load permeability fields
        self.perm_names = os.listdir(osp.join(path, "permeability"))
        self.perm_names.sort()
        self.perm_mean = 1.14906847
        self.perm_std = 7.81547992

        # load the parameter values
        self.param_names = os.listdir(osp.join(path, "params"))
        self.param_names.sort()
        self.param_mean = 1.248473
        self.param_std = 0.7208982

    def seperate_img_names(self, names):
        P, U1, U2 = [], [], []
        for name in names:
            if name[0] == "P":
                P.append(name)
            elif name[0:2] == "U1":
                U1.append(name)
            elif name[0:2] == "U2":
                U2.append(name)
            else:
                raise Exception("File "+name+" isn't a pressure or velocity field!")

        return P, U1, U2

    def __len__(self):
        return len(self.P_names)

    def __getitem__(self, idx):

        W = torch.from_numpy(np.load(osp.join(self.root, "params", self.param_names[idx]))).float()
        W = (np.squeeze(W) - self.param_mean) / self.param_std
        W = W

        K = torch.from_numpy(np.load(osp.join(self.root, "permeability", self.perm_names[idx]))).float()
        K = (np.expand_dims(K, axis=0) - self.perm_mean) / self.perm_std

        P = torch.from_numpy(np.load(osp.join(self.root, "data", self.P_names[idx]))).float()
        P = (np.expand_dims(P, axis=0) - self.img_mean[0]) / self.img_std[0]

        '''
        U1 = torch.from_numpy(np.load(osp.join(self.root, "data", self.U1_names[idx]))).float()
        U1 = (np.expand_dims(U1, axis=0) - self.img_mean[1]) / self.img_std[1]

        U2 = torch.from_numpy(np.load(osp.join(self.root, "data", self.U2_names[idx]))).float()
        U2 = (np.expand_dims(U2, axis=0) - self.img_mean[2]) / self.img_std[2]
        '''

        Data = np.concatenate([P, K], axis=0)

        return Data, W


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


def plot_darcy(traj, savename):
    """Save 10 generated images trajectories"""

    num_rows, num_cols = 10, 10  # 10x10 grid
    assert traj.shape[0] == num_rows * num_cols, "traj must contain 100 images"


    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        img = traj[idx].squeeze()  # from (1, 64, 64) to (64, 64)
        ax.imshow(img, cmap="jet")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.close(fig)