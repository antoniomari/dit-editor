#!/bin/bash
# Activate the environment
source venv/bin/activate

alpha_noise_list=(0 0.05 0.1 0.2 0.5)
fixed_arguments=" \
    --tau-alpha 0.4  \
    --tau-beta 0.8  \
    --timesteps 50  \
    --guidance-scale 3.0  \
    --run-on-first 1  \
    --random-samples \
    --random-samples-seed 91 \
    --inject-q  \
    --inject-k  \
    --inject-v  \
    --use-prompt \
    --layers all \
    --save-output-images"

for alpha_noise in "${alpha_noise_list[@]}"; do
    echo "Running with alpha noise: $alpha_noise"
    dit-run-benchmark --alpha-noise $alpha_noise $fixed_arguments --output-dir ./ablation_data/alpha_noise
done
