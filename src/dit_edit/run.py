import argparse
import logging
import os
from functools import partial

import numpy as np
import torch
from PIL import Image, ImageDraw
from rembg import new_session, remove

from dit_edit.config import DitEditConfig, parse_script_specific_args_run
from dit_edit.core.cached_pipeline import CachedPipeline
from dit_edit.core.flux_pipeline import EditedFluxPipeline
from dit_edit.core.inversion import compose_noise_masks
from dit_edit.core.processors.dit_edit_processor import DitEditProcessor
from dit_edit.utils.inference_utils import (
    get_bbox_from_mask_image,
    resize_image_to_max_side,
)
from dit_edit.utils.logging import setup_logger

logger = setup_logger("dit_edit_run")


def main(args, config: DitEditConfig):
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="End-to-end image composition script for DiT-Edit."
    )
    parser = parse_script_specific_args_run(parser)
    DitEditConfig.add_arguments_to_parser(parser)

    # Parse arguments
    args = parser.parse_args()
    config = DitEditConfig.from_args(parser, args)

    # Extract script-specific arguments
    debug = args.debug
    bg_path = args.bg_path
    fg_path = args.fg_path
    bbox_path = args.bbox_path
    prompt = args.prompt
    output_dir = args.output_path

    # Log the parameters
    logger.info("DiT-Edit parameters:")
    for key, value in config.to_dict().items():
        logger.info(f"\t{key.upper()}: {value}")
    logger.info("Script-specific parameters:")
    logger.info(f"\tBackground image path: {bg_path}")
    logger.info(f"\tForeground image path: {fg_path}")
    logger.info(f"\tBounding box mask path: {bbox_path}")
    logger.info(f"\tOutput image path: {output_dir}")
    logger.info(f"\tPrompt: {prompt if prompt else 'None'}")
    logger.info(f"\tDebug mode: {'Enabled' if debug else 'Disabled'}")

    # Setup debug directory
    if debug:
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(
            f"Debug mode enabled. Intermediate images will be saved to ./{debug_dir}"
        )

    # Load images
    try:
        bg_image_orig = Image.open(bg_path).convert("RGB")
        fg_image_orig = Image.open(fg_path).convert("RGB")
    except FileNotFoundError as e:
        logger.error(f"Error: Could not find image file: {e.filename}")
        return
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        return

    # Load bounding box mask image
    try:
        bbox_mask_image = Image.open(bbox_path).convert("L")
        if debug:
            bbox_mask_image.save(os.path.join(debug_dir, "bbox_mask_image_input.png"))
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading the bounding box mask: {e}"
        )
        return

    ##################################
    #     Handle image resizing      #
    ##################################
    logger.info("Resizing images...")
    bg_image, bg_resize_ratio = resize_image_to_max_side(bg_image_orig, patch_size=16)
    fg_image, _ = resize_image_to_max_side(fg_image_orig, patch_size=16)

    # Resize the bounding box mask to match the resized background
    if bg_resize_ratio != 1.0:
        logger.info(
            f"Background image resized to {bg_image.size} with resize ratio {bg_resize_ratio:.2f}"
        )
        bbox_mask_image = bbox_mask_image.resize(bg_image.size, Image.NEAREST)
        logger.info(f"Bounding box mask resized to {bbox_mask_image.size}")

    if debug:
        # Save resized images for debugging
        bg_image.save(os.path.join(debug_dir, "bg_image.png"))
        fg_image.save(os.path.join(debug_dir, "fg_image.png"))
        # Create a visual representation of the final bbox on the resized bg for debugging
        bg_with_bbox_debug = bg_image.copy()
        bbox_points = get_bbox_from_mask_image(bbox_mask_image)
        if bbox_points is None:
            logger.error(
                "Error: Bounding box could not be determined from the mask image."
            )
            return
        draw = ImageDraw.Draw(bg_with_bbox_debug)
        draw.rectangle(bbox_points, outline="red", width=2)
        bg_with_bbox_debug.save(os.path.join(debug_dir, "bg_image_with_bbox.png"))

    ##################################
    #    Create segmentation mask    #
    ##################################
    logger.info("Generating segmentation mask...")
    rembg_session = new_session()
    fg_mask_pil = remove(
        fg_image,
        session=rembg_session,
        only_mask=True,
        alpha_matting=True,
        alpha_matting_erode_size=0,
        alpha_matting_foreground_threshold=100,
    )
    segm_mask = (np.array(fg_mask_pil) > 127).astype(int)

    # Make sure the segmentation mask has selected enough area
    mask_area = np.sum(segm_mask)
    max_mask_area = segm_mask.shape[0] * segm_mask.shape[1]
    if max_mask_area == 0:
        logger.warning(
            "Warning: Foreground image has zero area. Using a full mask for the (zero-sized) foreground."
        )
        # fg_mask_np remains as is (likely all zeros if total_area is 0)
    elif (mask_area / max_mask_area) < config.min_mask_area_ratio:
        logger.info(
            f"Generated mask area ({mask_area / max_mask_area:.2f}) is below threshold ({config.min_mask_area_ratio}). Using a full foreground mask."
        )
        segm_mask = np.ones((fg_image.height, fg_image.width), dtype=int)

    # Convert the segmentation mask to PIL Image
    segm_mask_image = Image.fromarray((segm_mask * 255).astype(np.uint8), mode="L")

    ##################################
    #       Get starting noises      #
    ##################################
    logger.info("Initializing pipeline...")
    dtype = torch.float16
    try:
        pipe = EditedFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", device_map="balanced", torch_dtype=dtype
        )
        pipe.set_progress_bar_config(disable=True)
        cached_pipe = CachedPipeline(pipe)
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        return

    logger.info("Composing noise masks...")
    example_noise = compose_noise_masks(
        cached_pipe,
        foreground_image=fg_image,
        background_image=bg_image,
        target_mask=bbox_mask_image,
        foreground_mask=segm_mask_image,
        option="segmentation1",
        num_inversion_steps=config.timesteps,
        photoshop_fg_noise=True,
    )

    ##################################
    #         Compose images         #
    ##################################
    logger.info("Running image composition...")
    if config.layers_for_injection == "vital":
        vital_layers = [
            f"transformer.transformer_blocks.{i}" for i in [0, 1, 17, 18]
        ] + [
            f"transformer.single_transformer_blocks.{i-19}"
            for i in [25, 28, 53, 54, 56]
        ]
        layers_to_inject = vital_layers
    elif config.layers_for_injection == "all":
        layers_to_inject = [
            f"transformer.transformer_blocks.{i}" for i in range(19)
        ] + [f"transformer.single_transformer_blocks.{i-19}" for i in range(19, 57)]
    else:
        raise ValueError(
            "Invalid value for --layers-for-injection. Use 'vital' or 'all'."
        )

    torch.manual_seed(config.seed)
    prompt_text = prompt if config.use_prompt_in_generation and prompt else ""
    prompts_for_pipe = ["", "", prompt_text]

    output_image_data = cached_pipe.run_inject_qkv(
        prompts_for_pipe,
        num_inference_steps=config.timesteps,
        seed=config.seed,
        guidance_scale=config.guidance_scale,
        positions_to_inject=layers_to_inject,
        positions_to_inject_foreground=layers_to_inject,
        empty_clip_embeddings=False,
        q_mask=example_noise["latent_masks"]["latent_segmentation_mask"],
        latents=torch.stack(
            [
                example_noise["noise"]["background_noise"],
                example_noise["noise"]["foreground_noise"],
                torch.where(
                    example_noise["latent_masks"]["latent_segmentation_mask"] > 0,
                    config.alpha_noise
                    * torch.randn_like(example_noise["noise"]["foreground_noise"])
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
        width=bg_image.size[0],
        height=bg_image.size[1],
        inverted_latents_list=list(
            zip(
                example_noise["noise"]["background_noise_list"],
                example_noise["noise"]["foreground_noise_list"],
            )
        ),
        tau_b=config.tau_beta,
        bg_consistency_mask=example_noise["latent_masks"]["latent_segmentation_mask"],
    )

    try:
        final_image = output_image_data[0][2]
        os.makedirs(output_dir, exist_ok=True)
        final_image.save(os.path.join(output_dir, "output_image.png"))
        logger.info(f"Output image saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving output image: {e}")


if __name__ == "__main__":
    main()
