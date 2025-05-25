import argparse
import gc
import json
import logging
import os
import random
from functools import partial

import torch
from accelerate.utils import release_memory
from tqdm import tqdm

from dit_edit.config import (  # Added import
    DitEditConfig,
    parse_script_specific_args_benchmark,
)
from dit_edit.core.cached_pipeline import CachedPipeline
from dit_edit.core.flux_pipeline import EditedFluxPipeline
from dit_edit.core.inversion import compose_noise_masks
from dit_edit.core.processors.dit_edit_processor import DitEditProcessor
from dit_edit.data.benchmark_data import BenchmarkExample
from dit_edit.data.bulk_load import load_benchmark_data
from dit_edit.evaluation.eval import get_scores_for_single_example
from dit_edit.utils.logging import setup_logger

logger = setup_logger(
    "run_on_benchmark",
)


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


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="This script allows to run DitEdit on the benchmark dataset, with various parameters for injection and noise control."
    )
    parser = parse_script_specific_args_benchmark(parser)
    DitEditConfig.add_arguments_to_parser(parser)

    # Parse arguments
    args = parser.parse_args()
    config = DitEditConfig.from_args(parser, args)

    # Extract script-specific parameters from args
    run_on_first: int = args.run_on_first
    save_output_images: bool = args.save_output_images
    random_samples: bool = args.random_samples
    random_samples_seed: int = args.random_samples_seed
    skip_available: bool = args.skip_available

    # Log the parameters
    logger.info("DiT-Edit parameters:")
    for key, value in config.to_dict().items():
        logger.info(f"\t{key.upper()}: {value}")
    logger.info("Script parameters:")
    logger.info(f"\tRun on first: {run_on_first}")
    logger.info(f"\tSave output images: {save_output_images}")
    logger.info(f"\tRandomly select samples: {random_samples}")
    logger.info(f"\tSeed for random samples selection: {random_samples_seed}")
    logger.info(f"\tSkip generation if available: {skip_available}")

    # Setup pipeline
    dtype = torch.float16
    pipe = EditedFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", device_map="balanced", torch_dtype=dtype
    )
    pipe.set_progress_bar_config(disable=True)

    try:
        cached_pipe = CachedPipeline(pipe)
    except NameError:
        logger.warning(
            "CachedPipeline class not found. Proceeding with 'pipe' directly for some operations if applicable, but 'run_inject_qkv' might fail if it's a CachedPipeline method."
        )
        cached_pipe = pipe

    os_path = "benchmark_images_generations/"
    all_images = load_benchmark_data(os_path, logger=logger)

    vital_layers = [f"transformer.transformer_blocks.{i}" for i in [0, 1, 17, 18]] + [
        f"transformer.single_transformer_blocks.{i-19}" for i in [25, 28, 53, 54, 56]
    ]
    all_layers = [f"transformer.transformer_blocks.{i}" for i in range(19)] + [
        f"transformer.single_transformer_blocks.{i-19}" for i in range(19, 57)
    ]

    methods = ["naive", "tf-icon", "kv-edit", "ours"]

    metrics = {}
    for category in tqdm(all_images, desc="Categories"):
        metrics[category] = []
        num_examples_to_run = (
            int(run_on_first) if run_on_first >= 1 else len(all_images[category])
        )
        if random_samples and num_examples_to_run > 0:
            logger.info(
                f"Randomly sampling {num_examples_to_run} images from {category} with seed {random_samples_seed}"
            )
            random.seed(random_samples_seed or 42)
            ids_examples_to_run = random.sample(
                range(len(all_images[category])), num_examples_to_run
            )
        else:
            ids_examples_to_run = range(num_examples_to_run)

        for i in tqdm(ids_examples_to_run, desc=f"Examples in {category}", leave=False):
            try:
                example: BenchmarkExample = all_images[category][i]

                img_filename = f"alphanoise{config.alpha_noise}_timesteps{config.timesteps}_Q{config.inject_q}_K{config.inject_k}_V{config.inject_v}_taua{config.tau_alpha}_taub{config.tau_beta}_guidance{config.guidance_scale}_{config.layers_for_injection}-layers.png"
                metrics_filename = f"alphanoise{config.alpha_noise}_timesteps{config.timesteps}_Q{config.inject_q}_K{config.inject_k}_V{config.inject_v}_taua{config.tau_alpha}_taub{config.tau_beta}_guidance{config.guidance_scale}_{config.layers_for_injection}-layers.json"
                output_dir = f"./benchmark_images_generations/{category}/{example.image_number} {example.prompt}"
                if (
                    skip_available
                    and os.path.exists(os.path.join(output_dir, img_filename))
                    and os.path.exists(os.path.join(output_dir, metrics_filename))
                ):
                    logger.info(
                        f"Image and metrics already exist for seed {config.seed}. Skipping..."
                    )
                    continue

                logger.info(
                    f"Processing: Category='{category}', Example Index='{i}', Seed='{config.seed}'"
                )
                logger.info("Composing noise...")
                example_noise = compose_noise_masks(
                    cached_pipe,
                    example.fg_image,
                    example.bg_image,
                    example.target_mask,
                    example.fg_mask,
                    option="segmentation1",
                    num_inversion_steps=config.timesteps,
                    photoshop_fg_noise=True,
                )
                logger.info("Running inject qkv...")

                if config.layers_for_injection == "vital":
                    layers_to_inject = vital_layers
                elif config.layers_for_injection == "all":
                    layers_to_inject = all_layers
                else:
                    raise ValueError(
                        f"Expected layers_for_injection to be 'vital' or 'all'. Got: {config.layers_for_injection}"
                    )

                torch.manual_seed(config.seed)

                current_images_output = cached_pipe.run_inject_qkv(
                    ["", "", example.prompt if config.use_prompt_in_generation else ""],
                    num_inference_steps=config.timesteps,
                    seed=config.seed,
                    guidance_scale=config.guidance_scale,
                    positions_to_inject=all_layers,  # This seems to be hardcoded to all_layers, review if this should use layers_to_inject
                    positions_to_inject_foreground=layers_to_inject,  # Uses the selection from config
                    empty_clip_embeddings=False,
                    q_mask=example_noise["latent_masks"]["latent_segmentation_mask"],
                    latents=torch.stack(
                        [
                            example_noise["noise"]["background_noise"],
                            example_noise["noise"]["foreground_noise"],
                            torch.where(
                                example_noise["latent_masks"][
                                    "latent_segmentation_mask"
                                ]
                                > 0,
                                config.alpha_noise
                                * torch.randn_like(
                                    example_noise["noise"]["foreground_noise"]
                                )
                                + (1 - config.alpha_noise)
                                * example_noise["noise"]["foreground_noise"],
                                example_noise["noise"]["background_noise"],
                            ),
                        ]
                    ),
                    processor_class=partial(
                        DitEditProcessor,
                        call_max_times=int(config.tau_alpha * config.timesteps),
                        inject_q=config.inject_q,
                        inject_k=config.inject_k,
                        inject_v=config.inject_v,
                    ),
                    width=example.bg_image.size[0],
                    height=example.bg_image.size[1],
                    inverted_latents_list=list(
                        zip(
                            example_noise["noise"]["background_noise_list"],
                            example_noise["noise"]["foreground_noise_list"],
                        )
                    ),
                    tau_b=config.tau_beta,
                    bg_consistency_mask=example_noise["latent_masks"][
                        "latent_segmentation_mask"
                    ],
                )
                logger.info("Computing scores...")

                example.output = current_images_output[0][2]
                scores = get_scores_for_single_example(example, methods)

                if save_output_images:
                    save_path = os.path.join(output_dir, img_filename)
                    try:
                        example.output.save(save_path)
                        logger.info(f"Saved output image to {save_path}")
                    except Exception as e:
                        logger.warning(f"Error saving image {save_path}: {e}")

                metrics_save_path = os.path.join(output_dir, metrics_filename)

                with open(metrics_save_path, "w") as f:
                    json.dump(scores, f, indent=4)
                logger.info(f"Saved metrics to {metrics_save_path}")

            except Exception as e:
                logger.warning(
                    f"Error processing example {i} in category {category} with seed {config.seed}: {e}"
                )
                clear_all_gpu_memory()
                continue
            finally:
                clear_all_gpu_memory()


if __name__ == "__main__":
    main()
