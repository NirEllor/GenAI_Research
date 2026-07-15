# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is the research codebase for the paper "Efficient Flow Matching using Latent Variables" (Latent-CFM). It extends the `torchcfm` conditional flow matching library with pretrained/learned VAE latent conditioning, to improve image/field generation by exploiting low-dimensional manifold structure in the data. Two experiment pipelines exist: CIFAR-10 image generation (`code/cifar10/`) and 2D Darcy-flow PDE field generation (`code/darcy_flow/`).

There is no `requirements.txt`/`setup.py`/`pyproject.toml`, no test suite, and no linter config in this repo. Dependencies are only discoverable via imports: `torch`, `torchvision`, `torchdyn`, `torchdiffeq`, `absl-py`, `pot` (Python Optimal Transport), `cleanfid`, `pytorch-lightning-bolts` (`pl_bolts`), `lpips`, `wandb`, `matplotlib`, `tqdm`.

## Running training / evaluation

Scripts assume they are invoked from the repository root (they append relative paths like `./code/cifar10/` and `./code/torchcfm/models/unet/` to `sys.path` themselves) and use `absl.flags` for CLI args, not `argparse`.

Train Latent-CFM (VAE-latent-conditioned) on CIFAR-10 with DDP:
```
MASTER_ADDR=$(hostname)
MASTER_PORT=12357
torchrun --standalone --nnodes=1 --nproc_per_node=2 ./code/cifar10/train_cifar10_ddp_vae_cond_ic.py \
  --model "icfm" --output_dir "./code/cifar10/runs/" --lr 2e-4 --ema_decay 0.9999 \
  --batch_size 128 --num_workers 4 --total_steps 600001 --save_step 100000 \
  --parallel True --master_addr $MASTER_ADDR --master_port $MASTER_PORT
```
`--model` selects the flow-matching variant: `otcfm` (exact OT), `icfm` (independent CFM), `fm` (Lipman et al. target CFM), `si` (variance-preserving/stochastic interpolant) — see the `conditional_flow_matching.py` bullet under Architecture below.

Train the plain (no latent conditioning) baseline: same flags, via `code/cifar10/train_cifar10_ddp.py`.

Compute FID from a saved checkpoint:
```
python3 ./code/cifar10/compute_fid.py --integration_method 'euler' --integration_steps 100 \
  --class_cond 1 --model "icfm" --step 600000 --input_dir ./code/cifar10/runs/
```
`--class_cond 1` evaluates the VAE-latent-conditioned (`_Lcfm`) checkpoint; `0` evaluates the plain baseline checkpoint. `--integration_method` is `euler` or `dopri5` (adaptive-step, default).

Darcy-flow pipeline is analogous but requires training the VAE first (`vae_cifar10_ddp_medium.py`), since no pretrained VAE exists for PDE data, then the flow-matching model (`train_cifar10_ddp_vae_cond_ic_medium.py`), which loads that VAE checkpoint by path and logs to Weights & Biases (`wandb.init(project="flow_matching")`).

There are no automated tests; validating a change means running training for a short number of steps and/or `compute_fid.py`.

The teacher/student distillation pipeline (`generate_teacher_datasets.py`, `train_students.py`, `eval.py` — see `architecture.md`) has matching SLURM launchers in `slurm/`: `run_train_fm.sh` (teacher), `run_generate_datasets.sh`, `run_eval.sh`, each submitting one job per AE latent dim and optionally chained via `bash slurm/run_X.sh afterok:JID1:JID2:...`.

## Git workflow

Always commit changes made in this repository with a detailed commit message, and push to GitHub (`origin main`), once a change is complete — do not leave work only in the local working tree.

## Architecture

See `architecture.md` for a full inventory of every autoencoder (AE/VAE) and model in the repository (CIFAR-10's pretrained `pl_bolts` VAE, Darcy-flow's from-scratch `StableDiffusion-PyTorch` VAE, the latent-conditioned vs. plain UNet variants, and which classes are actually wired into training vs. dead code).

### `code/torchcfm/` — vendored/adapted flow-matching library

This mirrors github.com/atong01/conditional-flow-matching (MIT), with the model layer replaced/extended for this paper.

- `conditional_flow_matching.py` — the flow-matching variants, all sharing the `sample_location_and_conditional_flow(x0, x1)` → `(t, xt, ut)` interface:
  - `ConditionalFlowMatcher` — base/independent CFM ("icfm").
  - `ExactOptimalTransportConditionalFlowMatcher` — OT-CFM ("otcfm"), draws couplings via `OTPlanSampler(method="exact")`; also has `guided_sample_location_and_conditional_flow` for label-conditioned coupling.
  - `TargetConditionalFlowMatcher` — Lipman et al. 2023 ("fm").
  - `SchrodingerBridgeConditionalFlowMatcher` — entropic-OT Schrödinger bridge.
  - `VariancePreservingConditionalFlowMatcher` — Albergo et al. trigonometric interpolant ("si").
- `optimal_transport.py` — `OTPlanSampler` wraps POT solvers (`exact`/`sinkhorn`/`unbalanced`/`partial`) for minibatch OT coupling; `wasserstein(...)` helper.
- `utils.py` — 2D toy-data helpers (moons/gaussians) and plotting; not used by the image/PDE pipelines.
- `models/models.py` — toy `MLP` model for 2D experiments; not used by the image pipelines either.
- `models/unet/` — the real model backbone:
  - `unet.py` — a plain OpenAI guided-diffusion-style `UNetModel`/`UNetModelWrapper`, no latent conditioning. Used by the baseline `train_cifar10_ddp.py`.
  - `unet_resnetVAE.py` — **the model class actually used for Latent-CFM training.** Same UNet backbone, plus an `EncoderVAE` (dead code) and a `UNetModel` extended with `num_latents` arg: a deterministic sequential projection (Linear → SiLU → Linear) maps the input latent `y` directly to the timestep embedding scale (`forward` returns `(vt)`, a plain tensor). This is how frozen AE latents get injected into the vector field.
  - `unet_resnetVAE2.py` — an experimental variant hardcoded for a 2D latent space; not imported by any driver script — leave it alone unless specifically asked to extend the 2D-latent experiments.
  - `blocks.py`, `nn.py` — shared resnet/attention blocks and low-level ops (`timestep_embedding`, `zero_module`, gradient checkpointing, etc.) from guided-diffusion.
  - `fp16_util.py`, `logger.py` — vendored guided-diffusion utilities (mixed-precision trainer, stdout/TensorBoard logger) that are **not used** by any current training script (training uses plain `torch.optim.Adam`/`print`/`tqdm`/`wandb`) — dead weight kept from the upstream source, not part of the active pipeline.
- `models/StableDiffusion-PyTorch/vae.py` + `blocks.py` — a from-scratch Stable-Diffusion-style VAE (`VAE`, `VAE_Encoder`, built from `DownBlock`/`MidBlock`/`UpBlock`), adapted from the public `explainingai-code/StableDiffusion-PyTorch` reimplementation (no explicit attribution header, but the class/config structure matches). This is the VAE **trained from scratch** for the Darcy-flow pipeline (no pretrained option exists for PDE data), as opposed to CIFAR-10 which uses a pretrained VAE (see below).

### CIFAR-10 pipeline (`code/cifar10/`)

- `train_cifar10_ddp_vae_cond_ic.py` — Latent-CFM training. Loads a **frozen, pretrained** `ConvAutoencoder` checkpoint (trained by `train_cifar10_ae.py`), encodes each batch to a latent, passes it as `y` into `unet_resnetVAE.UNetModelWrapper` (built with `num_latents=<latent_dim>`), and trains with MSE loss only: `MSE(vt, ut)`. No KL term or reparameterization; the AE latent conditions the UNet's timestep embedding deterministically. Maintains an EMA copy of the model (`utils_cifar.ema`). DDP is set up manually via `WORLD_SIZE`/`RANK` env vars from `torchrun` + `utils_cifar.setup` (`nccl` backend), gated by `--parallel`. Checkpoints save to `{output_dir}/{model}/Cifar10_weights_step_{step}_Lcfm_det.pt` (net, ema, scheduler, optimizer, step, conditioning metadata).
- `train_cifar10_ddp.py` — the no-latent-conditioning baseline: same training loop/flags, but uses the plain `unet.UNetModelWrapper` (no `num_latents`, no `y`), plain MSE loss, and checkpoints without the `_Lcfm` suffix. Use this as the reference when checking whether a change to the VAE-conditioned script has drifted from the base training logic.
- `compute_fid.py` — loads a checkpoint (`_Lcfm` suffix if `--class_cond 1`, else the plain baseline naming), integrates the learned vector field via `torchdyn.NeuralODE` (`euler`) or `torchdiffeq.odeint` (`dopri5` etc, controlled by `--tol`), and computes FID against real CIFAR-10 with `cleanfid` (`mode="legacy_tensorflow"`).
- `generate_teacher_datasets.py` — data-generation step for a teacher/student distillation study: for each already-trained per-latent-dim Latent-CFM checkpoint (the "teacher", from `train_cifar10_ddp_vae_cond_ic.py`), Euler-samples independent synthetic image datasets at several sizes (`--dataset_sizes`; each sample stores the starting noise `x0` (t=0), the final generated image `x1` (t=1), and the conditioning latent used) and one trajectory dataset (a strided subset of intermediate Euler states, for future students trained on "middle-time" targets instead of only the final image). Uses deterministic conditioning: real CIFAR batch → AE latent → Euler-integrate noise to image (no reparameterization); writes self-describing `.npy` files via `np.lib.format.open_memmap` plus a `manifest.json` (tagged with `conditioning: deterministic_ae_latent`), with atomic `.tmp` + rename and a `.done` marker per output so partial runs can't be mistaken for complete ones on `--overwrite`-less reruns. Expects the checkpoint layout produced by `slurm/run_train_fm.sh` (`{--input_dir}/latent_{dim}/{--model}/...`).
- `train_students.py` — distills each per-latent-dim teacher into a one-step "student": for every (latent dim, dataset size) pair (6×4=24 by default), loads that dim/size's `(x0, x1)` pairs from `generate_teacher_datasets.py`'s output and trains a `StudentDenoiser` (`code/torchcfm/models/student_denoiser.py`) to predict the single global velocity `v = x1 - x0` evaluated at `x0` with `t` fixed at 0 — so a trained student generates a sample in one forward pass (`x1_pred = x0 + student(x0)`), no ODE integration needed. AdamW + cosine LR + EMA + early stopping, mirroring the hyperparameters of the source distillation script this was adapted from. Checkpoints go to `{...}/students/student_{dim}_{n_samples}.pt`; skips any that already exist unless `--overwrite`.
- `code/torchcfm/models/student_denoiser.py` — `StudentDenoiser`: the one-step student's architecture, deliberately much simpler than the teacher UNet since it's always evaluated at a single fixed `t=0` — a pure residual conv net over full `(3,32,32)` images (no down/up-sampling, no timestep embedding, no FiLM conditioning, no attention), `input_proj` → `n_blocks` residual `GroupNorm+GELU+Conv3x3` blocks → `output_head`. Also provides `param_count` and `load_student(ckpt_path, latent_dim, device)` (reconstructs the architecture from the checkpoint's stored `student_hidden_channels`/`student_n_blocks`, validates `latent_dim` matches).
- `eval.py` — evaluates teachers + students: `--generate` samples images directly to PNGs (no separate decode phase — neither model produces an intermediate latent) and computes a shared per-dim AE-reconstruction folder for an AE-FID upper-bound metric; `--metrics` computes FID (`cleanfid`, `mode="legacy_tensorflow"`, matching `compute_fid.py`) and Inception Score (`torch_fidelity`, optional); `--plot` aggregates every dim/size's metrics into `metrics_all.json` and produces `fid_vs_size.png` / `fid_vs_dim.png` (the headline figure — FID vs. latent dim, one line per dataset size plus the teacher baseline). See `architecture.md`'s "Teacher/student distillation pipeline" section for the full step-by-step data flow.
- `train_cifar10_ae.py` — trains `ConvAutoencoder` (from `code/torchcfm/models/conv_autoencoder.py`) from scratch on CIFAR-10 across a range of latent dimensions (`[64, 128, 256, 384, 512, 1024]`). Uses LPIPS perceptual loss (VGG backbone, frozen), MSE, and KL divergence with configurable weights. Saves best checkpoints (by validation loss) to `checkpoints/<exp_name>/ae_<dim>.pt` and experiment config to `checkpoints/<exp_name>/config.json`. Usage: `python train_cifar10_ae.py` (all dims sequentially), `python train_cifar10_ae.py --dim 64` (single dim), or `python train_cifar10_ae.py --exp-name my_run` (custom experiment name).
- `utils_cifar.py` — shared DDP setup, EMA update, infinite dataloader iterator, sample/trajectory generation (plain and class/latent-conditioned variants), image tiling.

### Darcy-flow pipeline (`code/darcy_flow/`)

A parallel pipeline for 2-channel, 64×64 Darcy-flow PDE fields (pressure + permeability), reusing the same `torchcfm` flow-matching core and `unet_resnetVAE.UNetModelWrapper`, but:
- `vae_cifar10_ddp_medium.py` trains the StableDiffusion-PyTorch VAE **from scratch** on a custom `Darcy_Dataset` (`im_channels=2`, `z_channels=4`), since CIFAR-10's pretrained VAE doesn't apply here.
- `train_cifar10_ddp_vae_cond_ic_medium.py` mirrors `train_cifar10_ddp_vae_cond_ic.py` but loads that from-scratch Darcy VAE checkpoint instead of a pretrained one, builds the UNet with `dim=(2,64,64)` and a configurable `--latent_dim`, wraps the VAE in DDP too, and logs to `wandb` instead of local prints.
- `utils_cifar.py` (darcy_flow's own copy, not shared with `code/cifar10/utils_cifar.py`) adds `Darcy_Dataset` and `plot_darcy` on top of the same EMA/DDP/infinite-loop helpers.

When modifying shared logic (e.g., EMA, DDP setup, checkpoint format), check both `code/cifar10/utils_cifar.py` and `code/darcy_flow/utils_cifar.py` — they are separate copies, not a shared module, and have drifted from each other (Darcy's has the added `Darcy_Dataset`/`plot_darcy`).
