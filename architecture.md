# Architecture: Autoencoders and Models

This document catalogs every autoencoder (AE/VAE) and neural network model in this repository, and how they connect inside the CIFAR-10 and Darcy-flow training pipelines. See `CLAUDE.md` for commands; this file is a deeper reference for the model layer only.

## Overview

Latent-CFM trains a conditional flow-matching vector field (a UNet) whose timestep embedding is additionally conditioned on a latent code `y` drawn from a VAE encoder. Two different VAEs are used depending on the dataset:

| Pipeline | VAE used | VAE source | Trained how |
|---|---|---|---|
| CIFAR-10 (`code/cifar10/`) | `pl_bolts.models.autoencoders.VAE` | external package (`pytorch-lightning-bolts`), not in this repo | pretrained checkpoint `"cifar10-resnet18"`, loaded frozen |
| Darcy-flow (`code/darcy_flow/`) | `VAE` in `models/StableDiffusion-PyTorch/vae.py` | vendored in this repo | trained from scratch on Darcy data by `vae_cifar10_ddp_medium.py`, then loaded frozen for flow-matching training |

The flow-matching vector field itself is always `UNetModelWrapper` from `code/torchcfm/models/unet/unet_resnetVAE.py` (the latent-conditioned UNet) — except the non-latent baseline scripts, which use the plain `UNetModelWrapper` from `unet.py`.

## Autoencoders

### 1. `pl_bolts` VAE (CIFAR-10, pretrained, external)

- Class: `pl_bolts.models.autoencoders.VAE` (PyTorch-Lightning-Bolts package, not vendored in-repo).
- Used in `code/cifar10/train_cifar10_ddp_vae_cond_ic.py:221-223` and `code/cifar10/compute_fid.py:106`:
  ```python
  vae = VAE(32, lr=0.00001)
  vae = vae.from_pretrained("cifar10-resnet18").to(rank)
  vae.eval()
  ```
- Frozen throughout training (`.eval()`, no gradient updates, not wrapped in DDP).
- Input images are renormalized using `CIFAR10DataModule("./data/", normalize=True)`'s mean/std before encoding.
- `vae.encoder(x)` produces a flattened latent vector (512-dim, matching `num_latents=512` on the UNet) passed as `y` into `net_model(t, xt, y=latent)`.

### 2. `VAE` / `VAE_Encoder` — `code/torchcfm/models/StableDiffusion-PyTorch/vae.py` (Darcy-flow, trained from scratch)

Vendored/adapted from the public `explainingai-code/StableDiffusion-PyTorch` reimplementation of a Stable-Diffusion-style VAE (attribution header present in the sibling `blocks.py`, not in `vae.py` itself, but the class/config structure matches exactly).

- **`VAE(im_channels, model_config)`** — full encoder-decoder:
  - Encoder: `encoder_conv_in` → stack of `DownBlock`s (`down_channels`) → stack of `MidBlock`s (`mid_channels`) → `GroupNorm`+`SiLU` → `encoder_conv_out` (outputs `2*z_channels`, i.e. mean+logvar) → `pre_quant_conv`.
  - `encode(x)` splits the output into `mean, logvar`, reparameterizes (`sample = mean + std * randn`), and returns `(sample, out)` where `out` is the pre-split `2*z_channels` tensor (used downstream to derive mu/logvar again).
  - Decoder: `post_quant_conv` → `decoder_conv_in` → `MidBlock`s (reversed) → `UpBlock`s (reversed) → `GroupNorm`+`SiLU` → `decoder_conv_out`.
  - `forward(x)` returns `(reconstruction, encoder_output)`.
  - Config used for Darcy (`vae_cifar10_ddp_medium.py`, `train_cifar10_ddp_vae_cond_ic_medium.py`): `im_channels=2` (pressure + permeability fields), `down_channels=[16,32,64,128]`, `mid_channels=[128]`, `down_sample=[True,True,True]`, `z_channels=4`, `norm_channels=8`, `attn_down=[False,False,False]`.
- **`VAE_Encoder(im_channels, model_config)`** — encoder-only duplicate of the above (same layers/forward as `VAE.encode`), not imported/used by any driver script found in this repo; a leftover standalone class.
- Built from shared blocks in the sibling `blocks.py` in the same directory: `DownBlock`, `MidBlock`, `UpBlock` (resnet + optional self-attention, optional cross-attention; time-embedding conditioning is unused here since `t_emb_dim=None` is passed everywhere — this VAE is not timestep-conditioned).

**Training** (`code/darcy_flow/vae_cifar10_ddp_medium.py`): trains this VAE from scratch on `Darcy_Dataset` with reconstruction MSE + `0.001 * KL(mu, logvar)` loss, logged via `wandb`. Checkpoint saved as `{model}_cifar10_kde_weights_step_{step}_base_vae_medium.pt`.

**Reuse** (`code/darcy_flow/train_cifar10_ddp_vae_cond_ic_medium.py:209-219`): loads a specific pretrained checkpoint (`./code/darcy_flow/runs/icfm/icfm_cifar10_kde_weights_step_100000_base_vae_medium_pdeLoss.pt`), sets `.eval()`, and calls `vae.encode(x1)` each step to get the latent passed as `y` into the UNet. Note: the VAE is still wrapped in `DistributedDataParallel` and its weights are saved alongside the UNet checkpoint (`"vae": vae.state_dict()`), even though it's not being trained further — this looks like leftover fine-tuning infrastructure rather than intentional joint training (no `vae.parameters()` are given to the optimizer).

### 3. `EncoderVAE` — `code/torchcfm/models/unet/unet_resnetVAE.py` (defined, effectively unused)

- A third, separate encoder-only VAE implementation living inside the UNet file itself, built from the *same* `DownBlock`/`MidBlock` pattern as StableDiffusion-PyTorch's blocks (this file has its own copy in `code/torchcfm/models/unet/blocks.py`).
- `encode(x)` → `(mu, logvar)` implicitly via `2*z_channels` output (reparameterization logic is present but commented out in the class itself; only `encoder_output` is returned from `forward`).
- Imported by `code/darcy_flow/train_cifar10_ddp_vae_cond_ic_medium.py:29` (`from unet_resnetVAE import UNetModelWrapper, EncoderVAE`) but **never instantiated** in that file — only `UNetModelWrapper` and the separate `VAE` (from `StableDiffusion-PyTorch/vae.py`) are actually used. Treat `EncoderVAE` as dead code/an unused import unless a script not covered here is found to use it.
- An identical copy of this class also exists in `unet_resnetVAE2.py` (see below) — same caveat applies.

## Flow-matching vector field models (UNets)

All three UNet files share the same OpenAI guided-diffusion backbone (`ResBlock`, `AttentionBlock`, `Downsample`/`Upsample`, `TimestepEmbedSequential`) defined at the top of each file; they differ only in the latent-conditioning logic added to `UNetModel.__init__`/`forward`.

### `code/torchcfm/models/unet/unet.py` — plain UNet, no latent conditioning

- `UNetModel(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, ...)` — standard guided-diffusion UNet; `forward(x, timesteps, y=None)` only supports class-conditioning (`num_classes`), no latent `y`.
- `UNetModelWrapper(UNetModel)` — convenience constructor taking `dim=(C,H,W)` and picking a default `channel_mult` by image size (32→`(1,2,2,2)`, 64→`(1,2,3,4)`, etc.).
- `EncoderUNetModel` / `EncoderUNetModelWrapper` — encoder-only half-UNet for classification/regression heads (pooling modes: `adaptive`, `attention`, `spatial`, `spatial_v2`); not used by any training script found.
- Used by the **baseline** (non-latent) training scripts: `code/cifar10/train_cifar10_ddp.py`.

### `code/torchcfm/models/unet/unet_resnetVAE.py` — latent-conditioned UNet (the model actually used for Latent-CFM)

Same backbone as `unet.py`, plus:
- Constructor gains `num_latents` (dimensionality of the incoming VAE latent) and `latent_dim` (default 256).
- When `num_latents is not None`:
  - `self.latent_encodings = nn.Linear(num_latents, latent_dim * 2)` — maps the raw VAE latent to `(mu, logvar)` for a second-stage reparameterization.
  - `self.latent_mlp = nn.Linear(latent_dim, time_embed_dim)` — projects the reparameterized latent sample into the same space as the timestep embedding.
- `forward(t, x, y=None)`:
  - In training mode (`self.training=True`): `proj = latent_encodings(y)` → split into `mu, logvar` → reparameterize → `latent_mlp(proj)` → **added to** the timestep embedding `emb`. Returns `(vt, mu, logvar)` so the caller can compute a KL term.
  - In eval mode: `y` is assumed to already be the reparameterized latent sample; it's passed straight through `latent_mlp` (masked to zero for near-zero inputs) and added to `emb`. Returns `(vt, None, None)`.
- `UNetModelWrapper(UNetModel)` in this file additionally forwards `num_latents`/`latent_dim` through to the base class.
- Also defines `EncoderVAE` (see above, effectively dead code in this repo's current scripts).
- **Used by**: `code/cifar10/train_cifar10_ddp_vae_cond_ic.py` (`num_latents=512`, default `latent_dim=256`) and `code/darcy_flow/train_cifar10_ddp_vae_cond_ic_medium.py` (`num_latents=8*8*8=512`, `latent_dim` CLI-configurable via `--latent_dim`, e.g. set to `2` for the 2D-latent experiments).

### `code/torchcfm/models/unet/unet_resnetVAE2.py` — experimental 2D-latent variant

- Near-identical file, explicitly commented "for Latent CFM with 2d latent space."
- Hardcodes the latent projection to a fixed 2D bottleneck regardless of the `latent_dim` argument:
  ```python
  self.latent_mlp = nn.Linear(math.floor(4/2), time_embed_dim)   # hardcoded input size 2
  self.latent_encodings = nn.Linear(self.num_latents, 4)          # hardcoded output size 4 (mu/logvar of dim 2)
  ```
- Not imported by any driver script in this repo (all training scripts import from `unet_resnetVAE`, not `unet_resnetVAE2`). Keep this in mind before "fixing" it — it looks like a scratch variant for a specific 2D-latent experiment, not a maintained alternative; check with whoever is doing 2D-latent work before modifying or deleting it.

## Teacher/student distillation pipeline

A downstream pipeline built on top of the CIFAR-10 Latent-CFM teachers above, to study distilling each per-latent-dim teacher into a much cheaper one-step "student" generator. Four scripts, run in order:

| Step | Script | Produces |
|---|---|---|
| 1 | `code/cifar10/train_cifar10_ae_ddp.py` + `code/cifar10/train_cifar10_ddp_vae_cond_ic.py` (the "teacher") | `checkpoints/ae_<dim>.pt`, then `{runs}/latent_<dim>/<model>/Cifar10_weights_*_Lcfm.pt` |
| 2 | `code/cifar10/generate_teacher_datasets.py` | `{runs}/latent_<dim>/<model>/synthetic/` — per-dim synthetic image datasets |
| 3 | `code/cifar10/train_students.py` | `{runs}/latent_<dim>/<model>/students/student_<dim>_<n>.pt` — 24 one-step students (6 dims × 4 dataset sizes) |
| 4 | `code/cifar10/eval.py` | `{runs}/latent_<dim>/<model>/eval/`, `{runs}/eval_summary/<model>/` — FID/IS + the FID-vs-dim plot |

All four steps share the same on-disk convention: `--input_dir` is a base directory (default `./code/cifar10/runs/`), and every script derives `{input_dir}/latent_{dim}/{model}/...` for that dim/model's own artifacts (matching the layout `slurm/run_train_fm.sh` already produces via `train_cifar10_ddp_vae_cond_ic.py --output_dir ./code/cifar10/runs/latent_${DIM}/`). Matching SLURM launchers: `slurm/run_train_fm.sh` (step 1, teacher), `slurm/run_generate_datasets.sh` (step 2), `slurm/run_eval.sh` (step 4). `train_students.py` has no SLURM launcher yet.

### Step 2 — `generate_teacher_datasets.py`

For each latent dim, loads that dim's teacher checkpoint + matching `checkpoints/ae_<dim>.pt`, and reuses the exact conditioning/sampling logic from `compute_fid.py`'s `gen_1_img` (real CIFAR batch → AE-encode → reparameterize through the UNet's own `latent_encodings` → Euler-integrate noise to image) to produce:
- 4 independent `(x0, x1, latent)` triple datasets per dim, one per `--dataset_sizes` entry (`noise_fp16_n{size}.npy` = starting noise x0 at t=0, `images_uint8_n{size}.npy` = the teacher's final sample x1 at t=1, `latents_fp32_n{size}.npy` = the **raw AE latent** — `ae.encode(img)[0]`, shape `(size, dim)` — that conditioned that sample) — this repo's `x0`=noise/`x1`=data convention, per `train_cifar10_ddp_vae_cond_ic.py`, not the reversed convention some earlier drafts of this pipeline used. `encode_conditioning_latent` returns `(ae_latent, teacher_y)`: `ae_latent` is what's persisted (and what the student is conditioned on), while `teacher_y` is the teacher's own reparameterized projection of it (fixed width `--unet_latent_dim`, default 256, regardless of `dim`) used only to drive that sample's Euler integration — the two must not be conflated, since a prior version of this file saved `teacher_y` into a `(size, dim)`-shaped array, which silently only worked for `dim==256` and threw a numpy broadcast error for every other latent dim.
- 1 trajectory dataset per dim (`trajectory_images_fp16.npy`, a strided subset of intermediate Euler states, for possible future students trained on "middle-time" targets instead of only the two endpoints).

Outputs are self-describing `.npy` files (`np.lib.format.open_memmap`), written atomically (`.tmp` + rename) with a `.done` marker per output so a crash mid-generation can't be mistaken for a complete dataset on rerun.

### Step 3 — `train_students.py` / `StudentDenoiser`

For each of the 24 (dim, dataset size) combinations, loads that combination's `(x0, x1, latent)` triples and trains a **`StudentDenoiser`** (`code/torchcfm/models/student_denoiser.py`) via one-step distillation: `v_target = x1 - x0`, `v_pred = student(x0, latent)`, `loss = MSE(v_pred, v_target)` — no ODE integration, no timestep sampling, the student is always evaluated at the single fixed point `t=0`, conditioned on the same AE latent the teacher used to generate that `x1`. At inference, a trained student generates a sample in one forward pass: `x1_pred = x0 + student(x0, latent)`.

`StudentDenoiser` is deliberately much simpler than the teacher UNet, precisely because it never needs to condition on `t`, only on a flat latent vector: `input_proj` (`Conv2d(3, hidden_channels, 3)`) → add a spatially-broadcast latent embedding (`latent_mlp`: `Linear(latent_input_dim, latent_embed_dim) → GELU → Linear(latent_embed_dim, hidden_channels)`, broadcast to `[:, :, None, None]`) → `n_blocks` residual blocks (`GroupNorm+GELU+Conv3x3` ×2, added residually) → `output_head` (`GroupNorm+GELU+Conv2d(hidden_channels, 3, 1)`) — no down/up-sampling, no attention, no timestep embedding. `latent_input_dim` varies per experiment (64/128/256/384/512/1024, matching the AE dim); `latent_embed_dim` (default 256) is held fixed across all six so the students stay architecturally comparable except for that one input projection. `forward(x, latent)` raises `ValueError` if `latent`'s shape doesn't match `latent_input_dim`. `student_denoiser.py` also provides `param_count` and `load_student(ckpt_path, latent_dim, device)`, which reconstructs the architecture from the checkpoint's own stored `latent_embed_dim`/`student_hidden_channels`/`student_n_blocks`, validates the checkpoint's `latent_dim` matches what was requested, and raises `ValueError` for any checkpoint missing `conditioning == "ae_latent"` metadata (i.e. an old, latent-free `StudentDenoiser` checkpoint) rather than silently misloading it.

### Step 4 — `eval.py`

Three restartable phases (`--generate`, `--metrics`, `--plot`) evaluating both the teacher and every student. Unlike a from-scratch latent-space generative pipeline, there is **no intermediate latent to decode** for either model here — the teacher's Euler integration happens directly in image space, and the student's one-step forward pass outputs a full image directly — so `--generate` produces PNGs directly (no separate decode phase). Generating for a student now also loads that dim's **teacher checkpoint** (resolved the same way as `--teacher` mode), purely to compute the AE-latent conditioning via the teacher's own `latent_encodings` layer — this is what guarantees the teacher and student are evaluated with equivalent conditioning, not independently-sampled latents. `--generate` also computes a shared, dim-scoped AE reconstruction (`{input_dir}/latent_<dim>/ae_recon/`, real CIFAR-10 test images round-tripped through `ConvAutoencoder.encode`/`.decode`) used for an AE-FID upper-bound metric — generation quality is bounded by how well the AE reconstructs, since the teacher conditions on AE-encoded real-image latents. `--metrics` computes FID (via `cleanfid`, `mode="legacy_tensorflow"` — matching `compute_fid.py`'s convention, so numbers are comparable across the repo) and Inception Score (via the optional `torch_fidelity` package). `--plot` aggregates every dim/size's metrics JSON into `{input_dir}/eval_summary/<model>/metrics_all.json` and produces `fid_vs_size.png` and **`fid_vs_dim.png`** — the latter (FID vs. latent dim, one line per dataset size plus a teacher baseline) is the pipeline's headline figure.

A smoke test for the latent-conditioned architecture lives at `code/cifar10/tests/test_student_denoiser.py` (forward-shape checks across all six latent dims, checkpoint save/load round-trip, and rejection of mismatched/legacy checkpoints) — run with `python code/cifar10/tests/test_student_denoiser.py` (no pytest dependency required, though it's pytest-discoverable too).

## Other models (non-image, not part of the AE/UNet story)

- `code/torchcfm/models/models.py`:
  - `MLP(dim, out_dim=None, w=64, time_varying=False)` — a toy 4-layer SELU MLP used only for the 2D toy flow-matching examples in the upstream `torchcfm` library (moons/gaussians); not used by any script in `code/cifar10/` or `code/darcy_flow/`.
  - `GradModel(action)` — wraps a scalar "action" model and returns its gradient via `torch.autograd.grad`; also a toy-example utility, unused by the image/PDE pipelines.

## Shared building blocks

- `code/torchcfm/models/unet/blocks.py` and `code/torchcfm/models/StableDiffusion-PyTorch/blocks.py` are two near-duplicate copies of the same `DownBlock`/`MidBlock`/`UpBlock`/`UpBlockUnet` resnet+attention blocks (the StableDiffusion-PyTorch one carries the attribution comment: adapted from `explainingai-code/StableDiffusion-PyTorch`). Both support optional self-attention (`attn`) and cross-attention (`cross_attn`/`context_dim`), though cross-attention is unused by anything in this repo (no `context` is ever passed). If you need to change block behavior, check both copies — they are not shared/imported from one location.
- `code/torchcfm/models/unet/nn.py` — low-level ops shared by all `unet*.py` variants: `timestep_embedding`, `zero_module`, `checkpoint`/`CheckpointFunction` (gradient checkpointing), `normalization` (`GroupNorm32`), `conv_nd`/`linear`/`avg_pool_nd`.
- `code/torchcfm/models/unet/fp16_util.py`, `logger.py` — vendored guided-diffusion utilities (mixed-precision trainer, stdout/TensorBoard logger) that are imported by `unet.py`/`unet_resnetVAE*.py` for `convert_to_fp16`/`convert_to_f32` methods only; the actual training scripts never call these (`use_fp16` is never set to `True`), so in practice training always runs in fp32.

## Practical notes for future changes

- If you add or modify latent conditioning, the change belongs in `unet_resnetVAE.py`'s `UNetModel.__init__`/`forward` (the constructor args `num_latents`/`latent_dim` and the `latent_encodings`/`latent_mlp` layers) — this is the single model class used by both pretrained-VAE (CIFAR-10) and from-scratch-VAE (Darcy) pipelines.
- The two "extra" classes — `EncoderVAE` (in both `unet_resnetVAE.py` and `unet_resnetVAE2.py`) and `VAE_Encoder` (in `StableDiffusion-PyTorch/vae.py`) — are not wired into any current training script. Don't assume they're load-bearing; verify with a repo-wide `grep` for the class name before relying on or modifying them.
- CIFAR-10's VAE is pretrained and external (`pl_bolts`); Darcy's VAE is vendored and trained by this repo's own `vae_cifar10_ddp_medium.py`. If you need to swap datasets, decide up front which of these two conditioning strategies applies — there is no unified "train or load a VAE" abstraction, each pipeline hardcodes its own choice.
