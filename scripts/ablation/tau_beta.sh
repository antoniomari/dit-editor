#!/bin/bash
# Activate the environment
source venv/bin/activate

tau_beta_list=(0.2 0.4 0.6 0.8 1.0)
fixed_arguments=" \
    --tau-alpha 0.4  \
    --guidance-scale 3.0   \
    --alpha-noise 0.05  \
    --timesteps 50  \
    --run-on-first 1  \
    --random-samples \
    --random-samples-seed 42 \
    --inject-k  \
    --inject-v  \
    --inject-q  \
    --use-prompt \
    --save-output-images"

echo "Running with only vital layers"
for tau_beta in "${tau_beta_list[@]}"; do
    echo "Running with tau_beta: $tau_beta"
    dit-run-benchmark --tau-beta $tau_beta --layers vital $fixed_arguments
done
