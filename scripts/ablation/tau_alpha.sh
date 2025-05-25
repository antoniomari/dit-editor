#!/bin/bash
# Activate the environment
source venv/bin/activate

tau_alpha_list=(0.2 0.3 0.4 0.5 0.6 0.7)
fixed_arguments=" \
    --tau-beta 0.8  \
    --guidance-scale 3.0   \
    --alpha-noise 0.05  \
    --timesteps 50  \
    --run-on-first 1  \
    --inject-k  \
    --inject-v  \
    --inject-q  \
    --use-prompt \
    --save-output-images"

echo "Running with only injection in all layers"
for tau_alpha in "${tau_alpha_list[@]}"; do
    echo "Running with tau_alpha: $tau_alpha"
    dit-run-benchmark  --tau-alpha $tau_alpha --layers all $fixed_arguments
done

echo "Running with only vital layers"
for tau_alpha in "${tau_alpha_list[@]}"; do
    echo "Running with tau_alpha: $tau_alpha"
    dit-run-benchmark  --tau-alpha $tau_alpha --layers vital $fixed_arguments
done
