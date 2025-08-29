# Latent CFM
This repository contains the codebase of the paper "Efficient Flow Matching using Latent Variables".

**Abstarct**
Flow matching models have shown great potential in image generation tasks among probabilistic generative models. However, most flow matching models in the literature do not explicitly model the underlying structure/manifold in the target data when learning the flow from a simple source distribution like the standard Gaussian. This leads to inefficient learning, especially for many high-dimensional real-world datasets, which often reside in a low-dimensional manifold. Existing strategies of incorporating manifolds, including data with underlying multi-modal distribution, often require expensive training and hence frequently lead to suboptimal performance. To this end, we present 𝙻𝚊𝚝𝚎𝚗𝚝-𝙲𝙵𝙼, which provides simplified training/inference strategies to incorporate multi-modal data structures using pretrained deep latent variable models. Through experiments on multi-modal synthetic data and widely used image benchmark datasets, we show that 𝙻𝚊𝚝𝚎𝚗𝚝-𝙲𝙵𝙼 exhibits improved generation quality with significantly less training (up to ∼50% less) and computation than state-of-the-art flow matching models by incorporating extracted data features using pretrained lightweight latent variable models. Moving beyond natural images to generating fields arising from processes governed by physics, using a 2d Darcy flow dataset, we demonstrate that our approach generates more physically accurate samples than competitive approaches. In addition, through latent space analysis, we demonstrate that our approach can be used for conditional image generation conditioned on latent features, which adds interpretability to the generation process.

<div align="center">
  <img src="https://github.com/AnirbanSamaddar/Latent_CFM/blob/main/img/Schematic4.png?raw=true" width="700" height="500" />
</div>

**Figure:** Schematic of Latent-CFM framework. Given a data x1, 𝙻𝚊𝚝𝚎𝚗𝚝-𝙲𝙵𝙼 extracts latent features using a frozen encoder and a trainable stochastic layer. The features are embedded using
a linear layer and added to the learned vector field. The framework resembles an encoder-decoder architecture like VAEs.

**Usage**
To train 𝙻𝚊𝚝𝚎𝚗𝚝-𝙲𝙵𝙼 on CIFAR10 using DDP with 2 GPUs, use the following code:
```
MASTER_ADDR=$(hostname)  # Using the current node as the master address
MASTER_PORT=12357        # Fixed port; ensure it's free on the system
torchrun --standalone --nnodes=1 --nproc_per_node=2 ./code/cifar10/train_cifar10_ddp_vae_cond_ic.py   --model "icfm"   --output_dir "./code/cifar10/runs/"   --lr 2e-4   --ema_decay 0.9999   --batch_size 128   --num_workers 4   --total_steps 600001   --save_step 100000   --parallel True   --master_addr $MASTER_ADDR  --master_port $MASTER_PORT
```
To compute the FID from the saved model, use the following code:
```
python3 ./code/cifar10/compute_fid.py --integration_method 'euler' --integration_steps 100 --class_cond 1 --model "icfm" --step 600000 --input_dir ./code/cifar10/runs/
```
Change the ```integration_method``` and ```integration_steps``` accordingly to explore other solvers such as the ```dopri5```. 
