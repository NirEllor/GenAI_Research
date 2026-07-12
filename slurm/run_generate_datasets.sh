#!/bin/bash
# Generate synthetic teacher datasets (4 independent image datasets per dim,
# plus one trajectory dataset), one job per AE latent dim, for later student
# distillation via train_students.py. Requires that dim's teacher checkpoint
# to already exist (run_train_fm.sh).
#
# Usage:
#   bash slurm/run_generate_datasets.sh                        # no dependency
#   bash slurm/run_generate_datasets.sh afterok:JID1:JID2:...  # e.g. after run_train_fm.sh's jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

MODEL="icfm"

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G -c4 --time=1-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=gen_data_d${DIM} \
    --wrap "bash -c '$RUN python code/cifar10/generate_teacher_datasets.py \
      --input_dir ./code/cifar10/runs/ --model $MODEL \
      --latent_dims $DIM --latest True'" \
    | awk '{print $NF}')
  echo "  Submitted gen_data dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "gen_data job IDs: ${IDS[*]}"
