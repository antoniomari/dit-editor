#!/bin/bash
# Activate the environment
source venv/bin/activate

inputs_dirs=(
    "puppy"
    "mailbox"
    "sheep"
)
root_dir=ablation_data/text-features/inputs

for input_dir in "${inputs_dirs[@]}"; do
    echo "Processing input directory: $input_dir"

    # Open CSV file to read the prompt
    csv_file="$root_dir/$input_dir/prompts.csv"
    if [[ ! -f "$csv_file" ]]; then
        echo "CSV file not found: $csv_file"
        continue
    fi

    while IFS=, read -r tag prompt; do
        tag=$(echo "$tag" | sed 's/^"\|"$//g')
        prompt=$(echo "$prompt" | sed 's/^"\|"$//g')
        echo "Tag: $tag, Prompt: $prompt"

        dit-run \
            --bg_path "$root_dir/$input_dir/bg.png" \
            --fg_path "$root_dir/$input_dir/fg.png" \
            --bbox_path "$root_dir/$input_dir/bb_mask.png" \
            --segm_mask_path "$root_dir/$input_dir/segm_mask.png" \
            --output_path "ablation_data/text-features/outputs/$input_dir/$tag/" \
            --prompt "$prompt" \
            --inject-q \
            --inject-k \
            --inject-v \
            --layers-for-injection vital
        echo "Finished processing tag: $tag"
    done < "$csv_file"
    echo "Finished processing input directory: $input_dir"
done
