import numpy as np
import torch
from PIL import Image
import argparse # Import argparse
import json # Added for saving metrics
import os
import gc
from accelerate.utils import release_memory

from data.benchmark_data import gather_images # Assuming this provides example objects with attributes like .prompt, .fg_image etc.
from cache_and_edit.inversion import place_image_in_bounding_box, get_inverted_input_noise, resize_bounding_box, compose_noise_masks

from importlib import reload
import cache_and_edit
reload(cache_and_edit)
from cache_and_edit import * # Make sure CachedPipeline is defined here or imported properly
import cache_and_edit.hooks
reload(cache_and_edit.hooks)

from cache_and_edit.flux_pipeline import EditedFluxPipeline
from tqdm import tqdm
from functools import partial
from cache_and_edit.qkv_cache import TFICONAttnProcessor
# from cache_and_edit.inversion import compose_noise_masks # Already imported
from evaluation.eval import get_scores_for_single_example


# Environment settings (keeping these as is, not part of argparse for this example)
os.environ['HF_HOME'] = '/scratch/nevali'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/nevali'
os.environ['HF_DATASETS_CACHE'] = '/scratch/nevali'

# --- EXAMPLE OF USAGE ---
# python run_with_params.py  --tau-alpha 0.4  --tau-beta 0.8  --guidance-scale 3.0   --alpha-noise 0.05  --timesteps 50  --run-on-first 2  --inject-k  --inject-v  --inject-q  --use-prompt --save-output-images

def clear_all_gpu_memory():
    # Run garbage collection
    gc.collect()

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s).")

    # Iterate through each GPU
    for device_id in range(num_gpus):
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            release_memory()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.ipc_collect()
    print("GPU memory cleared across all available devices.")

def main(args):
    # Use args from argparse
    LAYERS_FOR_INJECTION = args.layers
    TAU_ALPHA = args.tau_alpha
    TAU_BETA = args.tau_beta
    GUIDANCE_SCALE = args.guidance_scale
    INJECT_K_CLI = args.inject_k
    INJECT_Q_CLI = args.inject_q
    INJECT_V_CLI = args.inject_v
    TIMESTEPS = args.timesteps
    ALPHA_NOISE = args.alpha_noise
    USE_PROMPT = args.use_prompt
    RUN_ON_FIRST = args.run_on_first
    SAVE_OUTPUT_IMAGES = args.save_output_images

    print("Running with parameters:")
    print(f"  LAYERS_FOR_INJECTION: {LAYERS_FOR_INJECTION}")
    print(f"  TAU_ALPHA: {TAU_ALPHA}")
    print(f"  TAU_BETA: {TAU_BETA}")
    print(f"  GUIDANCE_SCALE: {GUIDANCE_SCALE}")
    print(f"  INJECT_K: {INJECT_K_CLI}")
    print(f"  INJECT_Q: {INJECT_Q_CLI}")
    print(f"  INJECT_V: {INJECT_V_CLI}")
    print(f"  TIMESTEPS: {TIMESTEPS}")
    print(f"  ALPHA_NOISE: {ALPHA_NOISE}")
    print(f"  USE_PROMPT: {USE_PROMPT}")
    print(f"  RUN_ON_FIRST: {RUN_ON_FIRST}")
    print(f"  SAVE_OUTPUT_IMAGES: {SAVE_OUTPUT_IMAGES}")


    dtype = torch.float16
    pipe = EditedFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                        device_map="balanced",
                                        torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    try:
        cached_pipe = CachedPipeline(pipe)
    except NameError:
        print("Warning: CachedPipeline class not found. Proceeding with 'pipe' directly for some operations if applicable, but 'run_inject_qkv' might fail if it's a CachedPipeline method.")
        cached_pipe = pipe # Fallback, may not work as intended if CachedPipeline has specific methods

    os_path = "benchmark_images_generations/"
    all_images = gather_images(os_path)


    vital_layers = [f"transformer.transformer_blocks.{i}" for i in [0, 1, 17, 18]] + \
                [f"transformer.single_transformer_blocks.{i-19}" for i in [25, 28, 53, 54, 56]]
    all_layers = [f"transformer.transformer_blocks.{i}" for i in range(19)] + \
                [f"transformer.single_transformer_blocks.{i-19}" for i in range(19, 57)]

    methods = ["naive", "tf-icon", "kv-edit", "ours"] # Used by get_scores_for_single_example

    metrics = {}
    image_counter = 0

    for category in tqdm(all_images, desc="Categories"):
        metrics[category] = []
        # Ensure RUN_ON_FIRST is an integer for slicing
        num_examples_to_run = int(RUN_ON_FIRST) if RUN_ON_FIRST >= 1 else len(all_images[category])

        for i, example in enumerate(tqdm(all_images[category][:num_examples_to_run], desc=f"Examples in {category}", leave=False)):

            print(f"\nProcessing: Category='{category}', Example Index='{i}', Q={INJECT_Q_CLI}, K={INJECT_K_CLI}, V={INJECT_V_CLI}")
            print('Composing noise...')
            # Assuming example object has attributes like fg_image, bg_image, target_mask, fg_mask
            example_noise = compose_noise_masks(cached_pipe,
                        example.fg_image,
                        example.bg_image,
                        example.target_mask,
                        example.fg_mask,
                        option="segmentation1",
                        num_inversion_steps=TIMESTEPS,
                        photoshop_fg_noise=True,)
            print('Running inject qkv...')

            # set the layers for injection based on the CLI argument
            if LAYERS_FOR_INJECTION == "vital":
                layers_for_injection = vital_layers
            elif LAYERS_FOR_INJECTION == "all":
                layers_for_injection = all_layers
            else:
                raise ValueError("Invalid value for --layers. Use 'vital' or 'all'.")
            
            # Set seed
            torch.manual_seed(42)
            current_images_output = cached_pipe.run_inject_qkv( # Renamed to avoid conflict
                ["", "", example.prompt if USE_PROMPT else ""],
                num_inference_steps=TIMESTEPS,
                seed=42, # Consider making this a CLI arg
                guidance_scale=GUIDANCE_SCALE,
                positions_to_inject=all_layers,
                positions_to_inject_foreground=layers_for_injection,
                empty_clip_embeddings=False,
                q_mask=example_noise["latent_masks"]["latent_segmentation_mask"],
                latents=torch.stack(
                                    [
                                    example_noise["noise"]["background_noise"],
                                    example_noise["noise"]["foreground_noise"],
                                    torch.where(
                                        example_noise["latent_masks"]["latent_segmentation_mask"] > 0,
                                        ALPHA_NOISE * torch.randn_like(example_noise["noise"]["foreground_noise"]) + (1 - ALPHA_NOISE) * example_noise["noise"]["foreground_noise"],
                                        example_noise["noise"]["background_noise"],
                                    ),
                                    ]
                                ),
                processor_class=partial(
                    TFICONAttnProcessor,
                    call_max_times=int(TAU_ALPHA * TIMESTEPS),
                    inject_q=INJECT_Q_CLI,
                    inject_k=INJECT_K_CLI,
                    inject_v=INJECT_V_CLI,
                    ),
                width=512,
                height=512,
                inverted_latents_list = list(zip(example_noise["noise"]["background_noise_list"], example_noise["noise"]["foreground_noise_list"])),
                tau_b=TAU_BETA,
                bg_consistency_mask=example_noise["latent_masks"]["latent_segmentation_mask"],
            )
            print('Computing scores...')

            example.output = current_images_output[0][2]
            scores = get_scores_for_single_example(example, methods)

            # Save the output image if flag is set
            if SAVE_OUTPUT_IMAGES:
                # Construct filename using all hyperparams passed in cli
                img_filename = f"alphanoise{ALPHA_NOISE}_timesteps{TIMESTEPS}_Q{INJECT_Q_CLI}_K{INJECT_K_CLI}_V{INJECT_V_CLI}_taua{TAU_ALPHA}_taub{TAU_BETA}_guidance{GUIDANCE_SCALE}_{LAYERS_FOR_INJECTION}-layers.png"
                output_dir = f"./benchmark_images_generations/{category}/{example.image_number} {example.prompt}"
                os.makedirs(output_dir, exist_ok=True)
                
                save_path = os.path.join(output_dir, img_filename)
                try:
                    example.output.save(save_path)
                    print(f"Saved output image to {save_path}")
                except Exception as e:
                    print(f"Error saving image {save_path}: {e}")
            
            metrics_filename = f"alphanoise{ALPHA_NOISE}_timesteps{TIMESTEPS}_Q{INJECT_Q_CLI}_K{INJECT_K_CLI}_V{INJECT_V_CLI}_taua{TAU_ALPHA}_taub{TAU_BETA}_guidance{GUIDANCE_SCALE}_{LAYERS_FOR_INJECTION}-layers.json"
            output_dir = f"./benchmark_images_generations/{category}/{example.image_number} {example.prompt}"
            metrics_filename = os.path.join(output_dir, metrics_filename)

            with open(metrics_filename, 'w') as f:
                json.dump(scores, f, indent=4)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for cache and edit with configurable parameters.")

    # Float arguments
    parser.add_argument('--tau-alpha', type=float, default=0.4, help='Value for TAU_ALPHA (default: 0.4)')
    parser.add_argument('--tau-beta', type=float, default=0.8, help='Value for TAU_BETA (default: 0.8)')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='Guidance scale factor (default: 3.0)')
    parser.add_argument('--alpha-noise', type=float, default=0.05, help='Alpha noise parameter (default: 0.05)')

    # Integer arguments
    parser.add_argument('--timesteps', type=int, default=50, help='Number of timesteps (default: 50)')
    # Corrected --run-on-first to use type=int and default directly
    parser.add_argument('--run-on-first', type=int, default=-1,
                        help='Run on the first N images from each category (default: -1 == run on all)')
    parser.add_argument('--layers', type=str, default='vital', help='Layers where to perform injection. Can be either "all" or "vital" (default: vital)')

    # Boolean flags (False by default, True if flag is present)
    parser.add_argument('--inject-k', action='store_true',
                        help='Enable K injection')
    parser.add_argument('--inject-q', action='store_true',
                        help='Enable Q injection')
    parser.add_argument('--inject-v', action='store_true',
                        help='Enable V injection')
    parser.add_argument('--use-prompt', action='store_true',
                        help='Use the prompt')
    parser.add_argument('--save-output-images', action='store_true',
                        help='Save the generated output images')


    args = parser.parse_args()
    # It's good practice to clear GPU memory once at the start if issues are common
    # clear_all_gpu_memory() # You might not need this if memory is managed well within the loop.
    main(args)