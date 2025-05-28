#!/bin/bash
# Activate the environment
source venv/bin/activate

fixed_arguments=" \
    --tau-alpha 0.4  \
    --tau-beta 0.8  \
    --alpha-noise 0.05  \
    --timesteps 50  \
    --run-on-first 2  \
    --random-samples \
    --random-samples-seed 1010 \
    --inject-q  \
    --inject-k  \
    --inject-v  \
    --guidance-scale 3.0 \
    --layers all \
    --save-output-images"


echo "Running with prompt"
dit-run-benchmark --use-prompt $fixed_arguments --output-dir ./ablation_data/use-prompt

echo "Running with prompt"
dit-run-benchmark --no-use-prompt $fixed_arguments --output-dir ./ablation_data/use-prompt
