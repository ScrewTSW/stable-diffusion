#!/usr/bin/env bash

set -x

SEED=$(od -An -td4 -N4 < /dev/urandom)
ITERATION_COUNT=16

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python optimizedSD/optimized_txt2img.py --from-file prompts.txt --ckpt checkpoints/sd-v1-4.ckpt --outdir generated-images --skip_grid --ddim_steps 90 --n_iter "${ITERATION_COUNT}" --n_samples 1 --scale 8.0 --seed "${SEED}"
