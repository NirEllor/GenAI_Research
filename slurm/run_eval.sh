#!/bin/bash
# Evaluate teachers + one-step students: generate images and compute FID/IS,
# one job per AE latent dim (teacher + all 4 synthetic dataset sizes).
# Requires that dim's teacher checkpoint (run_train_fm.sh) and student
# checkpoints (train_students.py) to already exist.
#
# After every dim's job completes, aggregate + plot locally (cheap, CPU-only,
# not worth a SLURM job):
#   python code/cifar10/eval.py --plot
#
# Usage:
#   bash slurm/run_eval.sh                        # no dependency
#   bash slurm/run_eval.sh afterok:JID1:JID2:...  # e.g. after run_generate_datasets.sh / train_students.py jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

MODEL="icfm"
SIZES=(50000 100000 150000 200000)

IDS=()
for DIM in "${DIMS[@]}"; do
  SIZE_CMDS=""
  for SIZE in "${SIZES[@]}"; do
    SIZE_CMDS="$SIZE_CMDS && python code/cifar10/eval.py --generate --dim $DIM --size $SIZE --model $MODEL"
    SIZE_CMDS="$SIZE_CMDS && python code/cifar10/eval.py --metrics  --dim $DIM --size $SIZE --model $MODEL"
  done
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G -c4 --time=1-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=eval_d${DIM} \
    --wrap "bash -c '$RUN python code/cifar10/eval.py --generate --dim $DIM --teacher --model $MODEL --latest True \
      && python code/cifar10/eval.py --metrics --dim $DIM --teacher --model $MODEL --latest True \
      $SIZE_CMDS'" \
    | awk '{print $NF}')
  echo "  Submitted eval dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "eval job IDs: ${IDS[*]}"
echo "After all jobs complete, run: python code/cifar10/eval.py --plot"
