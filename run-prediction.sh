#!/usr/bin/env bash

set -x

SEED=$(od -An -td4 -N4 < /dev/urandom)

#PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python scripts/txt2img.py --from-file prompts.txt --ckpt sd-v1-4.ckpt --outdir generated-images --skip_grid --ddim_steps 50 --n_iter 1 --n_samples 1 --scale 8.0 --seed ${SEED}
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python optimizedSD/optimized_txt2img.py --from-file prompts.txt --ckpt checkpoints/sd-v1-4.ckpt --outdir generated-images --skip_grid --ddim_steps 5 --n_iter 1 --n_samples 1 --scale 8.0 --seed "${SEED}"
