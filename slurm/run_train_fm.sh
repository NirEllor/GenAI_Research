#!/bin/bash
# Train Latent-CFM flow matching models, one per AE latent dim, each conditioned on
# the already-trained ConvAutoencoder checkpoint at checkpoints/ae_<dim>.pt.
#
# Usage:
#   bash slurm/run_train_fm.sh                        # no dependency
#   bash slurm/run_train_fm.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

MODEL="icfm"
LR=2e-4
EMA_DECAY=0.9999
BATCH_SIZE=128
NUM_WORKERS=4
TOTAL_STEPS=600001
SAVE_STEP=100000

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G -c4 --time=2-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=fm_train_d${DIM} \
    --wrap "bash -c '$RUN python code/cifar10/train_cifar10_ddp_vae_cond_ic.py \
      --model $MODEL --output_dir ./code/cifar10/runs/latent_${DIM}/ \
      --lr $LR --ema_decay $EMA_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
      --total_steps $TOTAL_STEPS --save_step $SAVE_STEP \
      --latent_dim $DIM --ae_checkpoint checkpoints/ae_${DIM}.pt'" \
    | awk '{print $NF}')
  echo "  Submitted fm_train dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "fm_train job IDs: ${IDS[*]}"
