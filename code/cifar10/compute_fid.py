# Code modified from https://github.com/atong01/conditional-flow-matching/tree/main.



import os
import sys
sys.path.append('./code/cifar10/')

import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

sys.path.append('./code/torchcfm/models/unet/')
from unet_resnetVAE import UNetModelWrapper
from pathlib import Path
import argparse
sys.path.append('./code/torchcfm/models/')
from conv_autoencoder import ConvAutoencoder
from utils_cifar import infiniteloop, tile_image
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import numpy as np


FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="Batch size to compute FID")
flags.DEFINE_bool('ema',True, help='Use EMA model')
flags.DEFINE_integer("class_cond", 0, help="Residual type - 0: no class, 1: dispatcher, 2: clust_id")
flags.DEFINE_integer("latent_dim", None, help="flat latent dimension of the AE, e.g. 64/128/256/384/512/1024 (required if class_cond=1)")
flags.DEFINE_string("ae_checkpoint", None, help="path to an ae_<dim>.pt checkpoint (dict with 'latent_dim'/'state_dict' keys, required if class_cond=1)")

FLAGS(sys.argv)

if FLAGS.class_cond == 1:
    assert FLAGS.latent_dim is not None, "--latent_dim is required when --class_cond=1"
    assert FLAGS.ae_checkpoint is not None, "--ae_checkpoint is required when --class_cond=1"

# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        if y is not None:
            self.y = y

    def forward(self, t, x, *args, **kwargs):
        return self.model(t,x,y=self.y) if hasattr(self, 'y') else self.model(t,x)


# DATASETS/DATALOADER
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
        batch_size=FLAGS.batch_size_fid,
        shuffle=True,
        num_workers=12,
        drop_last=True,
    )

datalooper = infiniteloop(dataloader)


new_net = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_classes=None,
        num_latents=FLAGS.latent_dim if FLAGS.class_cond == 1 else None,
        class_cond= False,
    ).to(device)


if FLAGS.class_cond == 1:
    ae = ConvAutoencoder(latent_dim=FLAGS.latent_dim).to(device)
    ae_checkpoint = torch.load(FLAGS.ae_checkpoint, map_location=device)
    try:
        ae.load_state_dict(ae_checkpoint["state_dict"])
    except RuntimeError:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in ae_checkpoint["state_dict"].items():
            new_state_dict[k[7:]] = v
        ae.load_state_dict(new_state_dict)
    ae.eval()



# Load the model
if FLAGS.class_cond == 0:
    PATH = f"{FLAGS.input_dir}/{FLAGS.model}/Cifar10_weights_step_{FLAGS.step}.pt"
elif FLAGS.class_cond == 1:
    PATH = f"{FLAGS.input_dir}/{FLAGS.model}/Cifar10_weights_step_{FLAGS.step}_latent{FLAGS.latent_dim}_Lcfm_det.pt"

print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)

if FLAGS.class_cond == 1:
    ckpt_conditioning = checkpoint.get("conditioning")
    if ckpt_conditioning != "deterministic_ae_latent":
        raise RuntimeError(
            f"Checkpoint '{PATH}' has conditioning={ckpt_conditioning!r}, "
            f"but this script requires conditioning='deterministic_ae_latent'. "
            f"The checkpoint may be from the old variational-latent code path and cannot be loaded here."
        )
state_dict = checkpoint["ema_model"] if FLAGS.ema else checkpoint["net_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()
new_net.training = False
# print(new_net,flush=True)



def gen_1_img(unused_latent):
    with torch.no_grad():
        if FLAGS.class_cond == 0:
            x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device) if FLAGS.base_dist == 'normal' else sample((FLAGS.batch_size_fid,), params).to(device).view(-1, 3, 32, 32)
        elif FLAGS.class_cond == 1:
            x1 = next(datalooper).to(device)
            output_img = x1 / 2 + 0.5  # real conditioning images, [-1,1] -> [0,1] for display
            latent = ae.encode(output_img)[0]  # AE trained on [0,1] images

            output_tiled = tile_image(output_img[:100,], 10).cpu().numpy().transpose(1, 2, 0)
            output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
            output_tiled = np.squeeze(output_tiled)

            x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)


        if FLAGS.integration_method == "euler":
            # Define the integration method if euler is used
            if FLAGS.class_cond == 1:
                node = NeuralODE(torch_wrapper(new_net,y=latent), solver=FLAGS.integration_method)
            else:
                node = NeuralODE(new_net, solver=FLAGS.integration_method)
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, 2, device=device)
            if FLAGS.class_cond == 1:
                traj = odeint(
                    torch_wrapper(new_net,latent), x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
                )
            else:
                traj = odeint(new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method)
    traj = traj[-1, :]
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)

    traj = traj[:100,].view([-1, 3, 32, 32]).clip(-1, 1)
    traj = traj / 2 + 0.5
    if FLAGS.class_cond == 1:
        save_image(traj, f"{FLAGS.input_dir}/{FLAGS.model}/Generated_images_cifar10_weights_step_{FLAGS.step}_latent{FLAGS.latent_dim}_Lcfm.png", nrow=10)
    else:
        save_image(traj, f"{FLAGS.input_dir}/{FLAGS.model}/Generated_images_cifar10_weights_step_{FLAGS.step}_icfm.png", nrow=10)

    if FLAGS.class_cond == 1:
            plt.imshow(output_tiled)
            plt.savefig(f"{FLAGS.input_dir}/{FLAGS.model}/Original_images_cifar10_weights_step_{FLAGS.step}_latent{FLAGS.latent_dim}_Lcfm.png")
            plt.close()
    return img


print("Start computing FID")
score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name="cifar10",
    batch_size=FLAGS.batch_size_fid,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train",
    mode="legacy_tensorflow",
)
print()
print("FID has been computed")
print()
# print("Total NFE: ", new_net.nfe)
# print()
print("FID: ", score)


'''
Usage:

nohup python3 compute_fid.py --integration_method 'euler' --class_cond 1 --model "icfm" --step 600000 --input_dir ./code/cifar10/runs/ --latent_dim 256 --ae_checkpoint ./code/cifar10/runs/ae_latent256/cifar10_ae_weights_step_400000_latent256.pt &> ./logs/FID_cifar_ema_600K_Lcfm_euler.log &
nohup python3 compute_fid.py --integration_method 'euler' --integration_steps 1000 --class_cond 1 --model "icfm" --step 600000 --input_dir ./code/cifar10/runs/ --latent_dim 256 --ae_checkpoint ./code/cifar10/runs/ae_latent256/cifar10_ae_weights_step_400000_latent256.pt &> ./logs/FID_cifar_ema_600K_Lcfm_euler_1000.log &
nohup python3 compute_fid.py --integration_method 'dopri5' --class_cond 1 --model "icfm" --step 600000 --input_dir ./code/cifar10/runs/ --latent_dim 256 --ae_checkpoint ./code/cifar10/runs/ae_latent256/cifar10_ae_weights_step_400000_latent256.pt &> ./logs/FID_cifar_ema_600K_Lcfm_dopri.log &


'''