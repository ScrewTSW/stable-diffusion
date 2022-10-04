#!/usr/bin/env bash

set -x

SEED=$(od -An -td4 -N4 < /dev/urandom)
ITERATION_COUNT=4
STRENGTH=0.9
PROMPT=$1
IMG_PATH=$2

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python optimizedSD/optimized_img2img.py --prompt "${PROMPT}" --init-img "${IMG_PATH}" --ckpt checkpoints/sd-v1-4.ckpt --outdir generated-images --skip_grid --ddim_steps 90 --n_iter "${ITERATION_COUNT}" --n_samples 1 --scale 8.0 --strength "${STRENGTH}" --seed "${SEED}"
