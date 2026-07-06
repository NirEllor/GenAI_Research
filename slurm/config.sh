#!/bin/bash
# Shared configuration — sourced by every run_*.sh script.

EMAIL="ellorwaizner.nir@mail.huji.ac.il"
NODE_ARGS=""

PROJECT="/cs/labs/raananf/ellorw.nir/distillation/GenAI_Research"

RUN="source /cs/labs/raananf/ellorw.nir/venv/bin/activate && cd $PROJECT &&"

DIMS=(64 128 256 384 512 1024)
