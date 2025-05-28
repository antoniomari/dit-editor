#!/bin/bash
# Activate the environment
source venv/bin/activate

guidance_scale_list=(0 1.5 3 6 15)
fixed_arguments=" \
    --tau-alpha 0.4  \
    --tau-beta 0.8  \
    --alpha-noise 0.05  \
    --timesteps 50  \
    --run-on-first 1  \
    --random-samples \
    --random-samples-seed 345 \
    --inject-q  \
    --inject-k  \
    --inject-v  \
    --use-prompt \
    --layers all \
    --save-output-images"

for guidance_scale in "${guidance_scale_list[@]}"; do
    echo "Running with guidance scale: $guidance_scale"
    dit-run-benchmark --guidance-scale $guidance_scale $fixed_arguments --output-dir ./ablation_data/guidance
done
