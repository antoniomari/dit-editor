#!/bin/bash
# Activate the environment
source venv/bin/activate

num_steps_list=(1 10 25 50 100)
fixed_arguments=" \
    --tau-alpha 0.4  \
    --tau-beta 0.8  \
    --guidance-scale 3.0   \
    --alpha-noise 0.05  \
    --run-on-first 1  \
    --random-samples \
    --random-samples-seed 69 \
    --layers all \
    --inject-k  \
    --inject-v  \
    --inject-q  \
    --use-prompt \
    --save-output-images"

for num_steps in "${num_steps_list[@]}"; do
    echo "Running with num_steps: $num_steps"
    dit-run-benchmark --timesteps $num_steps $fixed_arguments --output-dir "./ablation_data/timesteps/"
done
