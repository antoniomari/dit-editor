import random
import torch
import argparse
import json
import os
import gc
import logging
from accelerate.utils import release_memory

from tqdm import tqdm
from functools import partial

from dit_edit.core.flux_pipeline import EditedFluxPipeline
from dit_edit.data.bulk_load import load_benchmark_data
from dit_edit.core.inversion import compose_noise_masks
from dit_edit.core.processors.dit_edit_processor import DitEditProcessor
from dit_edit.evaluation.eval import get_scores_for_single_example
from dit_edit.core.cached_pipeline import CachedPipeline
from dit_edit.data.benchmark_data import BenchmarkExample
from dit_edit.utils.logging import setup_logger

logger = setup_logger(logging.getLogger(__name__))

def clear_all_gpu_memory():
    # Run garbage collection
    gc.collect()

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU(s).")

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
    logger.info("GPU memory cleared across all available devices.")

def main(args):
    # Extract parameters from args
    layers_for_injection: str = args.layers
    tau_alpha: float = args.tau_alpha
    tau_beta: float = args.tau_beta
    guidance_scale: float = args.guidance_scale
    inject_k: bool = args.inject_k
    inject_q: bool = args.inject_q
    inject_v: bool = args.inject_v
    timesteps: int = args.timesteps
    alpha_noise: float = args.alpha_noise
    use_prompt: bool = args.use_prompt
    run_on_first: int = args.run_on_first
    save_output_images: bool = args.save_output_images
    random_samples: bool= args.random_samples
    random_samples_seed: int = args.random_samples_seed
    skip_available: bool = args.skip_available

    logger.info("Running with parameters:")
    logger.info(f"  LAYERS_FOR_INJECTION: {layers_for_injection}")
    logger.info(f"  TAU_ALPHA: {tau_alpha}")
    logger.info(f"  TAU_BETA: {tau_beta}")
    logger.info(f"  GUIDANCE_SCALE: {guidance_scale}")
    logger.info(f"  INJECT_K: {inject_k}")
    logger.info(f"  INJECT_Q: {inject_q}")
    logger.info(f"  INJECT_V: {inject_v}")
    logger.info(f"  TIMESTEPS: {timesteps}")
    logger.info(f"  ALPHA_NOISE: {alpha_noise}")
    logger.info(f"  USE_PROMPT: {use_prompt}")
    logger.info(f"  RUN_ON_FIRST: {run_on_first}")
    logger.info(f"  SAVE_OUTPUT_IMAGES: {save_output_images}")
    logger.info(f"  RANDOM_SAMPLES: {random_samples}")
    logger.info(f"  RANDOM_SAMPLES_SEED: {random_samples_seed}")

    # Setup pipeline
    dtype = torch.float16
    pipe = EditedFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                        device_map="balanced",
                                        torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    try:
        cached_pipe = CachedPipeline(pipe)
    except NameError:
        logger.warning("CachedPipeline class not found. Proceeding with 'pipe' directly for some operations if applicable, but 'run_inject_qkv' might fail if it's a CachedPipeline method.")
        cached_pipe = pipe # Fallback, may not work as intended if CachedPipeline has specific methods

    os_path = "benchmark_images_generations/"
    all_images = load_benchmark_data(os_path)


    vital_layers = [f"transformer.transformer_blocks.{i}" for i in [0, 1, 17, 18]] + \
                [f"transformer.single_transformer_blocks.{i-19}" for i in [25, 28, 53, 54, 56]]
    all_layers = [f"transformer.transformer_blocks.{i}" for i in range(19)] + \
                [f"transformer.single_transformer_blocks.{i-19}" for i in range(19, 57)]

    methods = ["naive", "tf-icon", "kv-edit", "ours"]

    metrics = {}
    for category in tqdm(all_images, desc="Categories"):
        metrics[category] = []
        num_examples_to_run = int(run_on_first) if run_on_first >= 1 else len(all_images[category])
        if random_samples and num_examples_to_run > 0:
            logger.info(f"Randomly sampling {num_examples_to_run} images from {category} with seed {random_samples_seed}")
            random.seed(random_samples_seed or 42)
            ids_examples_to_run = random.sample(range(len(all_images[category])), num_examples_to_run)
        else:
            ids_examples_to_run = range(num_examples_to_run)

        for i in tqdm(ids_examples_to_run, desc=f"Examples in {category}", leave=False):
            try:
                example: BenchmarkExample = all_images[category][i]

                img_filename = f"alphanoise{alpha_noise}_timesteps{timesteps}_Q{inject_q}_K{inject_k}_V{inject_v}_taua{tau_alpha}_taub{tau_beta}_guidance{guidance_scale}_{layers_for_injection}-layers.png"
                metrics_filename = f"alphanoise{alpha_noise}_timesteps{timesteps}_Q{inject_q}_K{inject_k}_V{inject_v}_taua{tau_alpha}_taub{tau_beta}_guidance{guidance_scale}_{layers_for_injection}-layers.json"
                output_dir = f"./benchmark_images_generations/{category}/{example.image_number} {example.prompt}"
                if skip_available and os.path.exists(os.path.join(output_dir, img_filename)) and os.path.exists(os.path.join(output_dir, metrics_filename)):
                    logger.info(f"Image and metrics already exist. Skipping...")
                    continue

                logger.info(f"\nProcessing: Category='{category}', Example Index='{i}', Q={inject_q}, K={inject_k}, V={inject_v}")
                logger.info('Composing noise...')
                # Assuming example object has attributes like fg_image, bg_image, target_mask, fg_mask
                example_noise = compose_noise_masks(cached_pipe,
                            example.fg_image,
                            example.bg_image,
                            example.target_mask,
                            example.fg_mask,
                            option="segmentation1",
                            num_inversion_steps=timesteps,
                            photoshop_fg_noise=True,)
                logger.info('Running inject qkv...')

                # set the layers for injection based on the CLI argument
                if layers_for_injection == "vital":
                    layers_for_injection = vital_layers
                elif layers_for_injection == "all":
                    layers_for_injection = all_layers
                else:
                    raise ValueError("Invalid value for --layers. Use 'vital' or 'all'.")
                
                # Set seed
                torch.manual_seed(42)
                current_images_output = cached_pipe.run_inject_qkv(
                    ["", "", example.prompt if use_prompt else ""],
                    num_inference_steps=timesteps,
                    seed=42, # Consider making this a CLI arg
                    guidance_scale=guidance_scale,
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
                                            alpha_noise * torch.randn_like(example_noise["noise"]["foreground_noise"]) + (1 - alpha_noise) * example_noise["noise"]["foreground_noise"],
                                            example_noise["noise"]["background_noise"],
                                        ),
                                        ]
                                    ),
                    processor_class=partial(
                        DitEditProcessor,
                        call_max_times=int(tau_alpha * timesteps),
                        inject_q=inject_q,
                        inject_k=inject_k,
                        inject_v=inject_v,
                        ),
                    width=example.bg_image.size[0],
                    height=example.bg_image.size[1],
                    inverted_latents_list = list(zip(example_noise["noise"]["background_noise_list"], example_noise["noise"]["foreground_noise_list"])),
                    tau_b=tau_beta,
                    bg_consistency_mask=example_noise["latent_masks"]["latent_segmentation_mask"],
                )
                logger.info('Computing scores...')

                example.output = current_images_output[0][2]
                scores = get_scores_for_single_example(example, methods)

                # Save the output image if flag is set
                if save_output_images:                
                    save_path = os.path.join(output_dir, img_filename)
                    try:
                        example.output.save(save_path)
                        logger.info(f"Saved output image to {save_path}")
                    except Exception as e:
                        logger.warning(f"Error saving image {save_path}: {e}")
                
                metrics_filename = f"alphanoise{alpha_noise}_timesteps{timesteps}_Q{inject_q}_K{inject_k}_V{inject_v}_taua{tau_alpha}_taub{tau_beta}_guidance{guidance_scale}_{layers_for_injection}-layers.json"
                output_dir = f"./benchmark_images_generations/{category}/{example.image_number} {example.prompt}"
                metrics_filename = os.path.join(output_dir, metrics_filename)

                with open(metrics_filename, 'w') as f:
                    json.dump(scores, f, indent=4)
            except Exception as e:
                logger.warning(f"Error processing example {i} in category {category}: {e}")
                continue
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This scrip allows to run DitEdit on the benchmark dataset, with various parameters for injection and noise control.")

    # Float arguments
    parser.add_argument('--tau-alpha', type=float, default=0.4, help='Value for TAU_ALPHA (default: 0.4)')
    parser.add_argument('--tau-beta', type=float, default=0.8, help='Value for TAU_BETA (default: 0.8)')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='Guidance scale factor (default: 3.0)')
    parser.add_argument('--alpha-noise', type=float, default=0.05, help='Alpha noise parameter (default: 0.05)')

    # Integer arguments
    parser.add_argument('--timesteps', type=int, default=50, help='Number of timesteps (default: 50)')
    parser.add_argument('--random-samples-seed', type=int, default=42, help='Seed for random sampling from benchmark data (default: 42)')
    # Corrected --run-on-first to use type=int and default directly
    parser.add_argument('--run-on-first', type=int, default=-1,
                        help='Run on the first N images from each category (default: -1 == run on all)')
    parser.add_argument('--layers', type=str, default='vital', help='Layers where to perform injection. Can be either "all" or "vital" (default: vital)')

    # Boolean flags (False by default, True if flag is present)
    parser.add_argument('--random-samples', action='store_true', help="If set together with a positive number of --run-on-first, it will randomly sample that number of images from each category.")
    parser.add_argument('--skip-available', action='store_true', help="If set, images that were already will not be regenerated.", default=False)
    parser.add_argument('--inject-k', action='store_true', help='Enable K injection')
    parser.add_argument('--inject-q', action='store_true', help='Enable Q injection')
    parser.add_argument('--inject-v', action='store_true', help='Enable V injection')
    parser.add_argument('--use-prompt', action='store_true', help='Use the prompt')
    parser.add_argument('--save-output-images', action='store_true', help='Save the generated output images')

    args = parser.parse_args()
    main(args)