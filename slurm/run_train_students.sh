#!/bin/bash
# Train one-step student denoisers via distillation from teachers, one job per
# AE latent dim. Each job trains all 4 dataset sizes (50k, 100k, 150k, 200k) for
# that dim. Requires synthetic datasets to already exist (run_generate_datasets.sh).
#
# Usage:
#   bash slurm/run_train_students.sh                        # no dependency
#   bash slurm/run_train_students.sh afterok:JID1:JID2:...  # e.g. after run_generate_datasets.sh's jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

MODEL="icfm"
BATCH_SIZE=256
LR=3e-4
TOTAL_STEPS=20000
STUDENT_HIDDEN_CHANNELS=64
STUDENT_N_BLOCKS=4

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G -c4 --time=0-06:00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=train_stud_d${DIM} \
    --wrap "bash -c '$RUN python code/cifar10/train_students.py \
      --input_dir ./code/cifar10/runs/ --model $MODEL \
      --dim $DIM \
      --batch_size $BATCH_SIZE --lr $LR --total_steps $TOTAL_STEPS \
      --student_hidden_channels $STUDENT_HIDDEN_CHANNELS --student_n_blocks $STUDENT_N_BLOCKS'" \
    | awk '{print $NF}')
  echo "  Submitted train_students dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "train_students job IDs: ${IDS[*]}"
