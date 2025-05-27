import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from functools import partial
import os
import argparse # Added for DitEditConfig
import logging # Added for DitEditConfig

from dit_edit.config import DitEditConfig
from dit_edit.core.cached_pipeline import CachedPipeline
from dit_edit.core.flux_pipeline import EditedFluxPipeline
from dit_edit.core.inversion import compose_noise_masks
from dit_edit.core.processors.dit_edit_processor import DitEditProcessor
from dit_edit.utils.inference_utils import (
    get_bbox_from_mask_image,
    resize_image_to_max_side,
)
from rembg import new_session, remove

# Setup logger - basic config for Gradio app
logger = logging.getLogger("gradio_app")
logging.basicConfig(level=logging.INFO)


# --- Helper function to convert drawn box to mask image ---
def create_bbox_mask(bg_image_pil_or_path, box_coords, original_bg_size):
    """
    Creates a binary mask image from bounding box coordinates drawn on an image.
    The mask will have the same dimensions as the original background image.
    The box_coords are expected to be (x_min, y_min, x_max, y_max).
    """
    if isinstance(bg_image_pil_or_path, str):
        # This case might not be used if Gradio provides PIL image directly
        # after drawing, but good to have for flexibility.
        img = Image.open(bg_image_pil_or_path).convert("RGB")
    else:
        img = bg_image_pil_or_path

    # Create a black image with the original background dimensions
    mask = Image.new("L", original_bg_size, 0)
    draw = ImageDraw.Draw(mask)

    if box_coords:
        # Ensure coordinates are integers
        x_min, y_min, x_max, y_max = [int(c) for c in box_coords]
        # Draw a white rectangle on the black mask
        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
    return mask

# --- Main image generation logic (adapted from run.py) ---
def generate_image(
    foreground_image_np,
    background_image_editor_data, # Changed from background_image_with_box
    prompt,
    # DitEditConfig parameters will be passed here
    tau_alpha, tau_beta, guidance_scale, alpha_noise, timesteps,
    layers_for_injection, inject_q, inject_k, inject_v,
    use_prompt_in_generation, min_mask_area_ratio, seed
):
    if foreground_image_np is None or background_image_editor_data is None:
        return None, "Please provide foreground and background images, and draw a bounding box."

    if background_image_editor_data.get('background') is None or \
       background_image_editor_data.get('layers') is None:
        logger.error(f"Invalid background_image_editor_data: {background_image_editor_data}")
        return None, "Background image data is missing or invalid. Please upload a background and draw on it."

    try:
        fg_image_orig = Image.fromarray(foreground_image_np).convert("RGB")
        
        # Process gr.ImageEditor output
        bg_image_orig_clean_pil = Image.fromarray(background_image_editor_data['background']).convert("RGB")
        
        drawn_layers = background_image_editor_data['layers']
        if not drawn_layers:
            return None, "Please draw a bounding box on the background image."

        # Combine all drawn layers to get a single mask of the drawing
        # Layers are RGBA numpy arrays. We'll use the alpha channel.
        h, w = drawn_layers[0].shape[:2]
        combined_drawings_alpha_mask_np = np.zeros((h, w), dtype=np.uint8)
        for layer_np in drawn_layers:
            if layer_np.shape[2] == 4:  # RGBA
                alpha_channel = layer_np[:, :, 3]
                combined_drawings_alpha_mask_np = np.maximum(combined_drawings_alpha_mask_np, alpha_channel)
            else: # Should not happen with current ImageEditor brush tools
                logger.warning(f"Unexpected layer format: shape {layer_np.shape}")


        if np.sum(combined_drawings_alpha_mask_np) == 0: # Empty drawing
             return None, "The drawing for the bounding box is empty or fully transparent. Please draw a clear, opaque rectangle."

        combined_drawings_mask_pil = Image.fromarray(combined_drawings_alpha_mask_np, mode='L')

        # Get the bounding box of all drawn elements
        bbox_coords = combined_drawings_mask_pil.getbbox()  # (left, upper, right, lower)

        if bbox_coords is None:
            return None, "Could not determine bounding box from the drawing. Please ensure the drawing is not empty and forms a rectangle."

        # Create the final bbox_mask_image (white rectangle on black background)
        # with the dimensions of the original clean background image
        bbox_mask_image = Image.new("L", bg_image_orig_clean_pil.size, 0)
        draw = ImageDraw.Draw(bbox_mask_image)
        draw.rectangle(bbox_coords, fill=255)
        logger.info(f"Created bbox_mask_image from drawing with coords: {bbox_coords} on a canvas of size {bg_image_orig_clean_pil.size}")

        # --- Initialize DitEditConfig ---
        # Create a dummy args namespace for DitEditConfig
        # In a real app, you might manage configs differently
        parser = argparse.ArgumentParser()
        DitEditConfig.add_arguments_to_parser(parser)
        # Create a list of arguments based on the function inputs
        # Note: Default values from DitEditConfig will be used if not provided by Gradio
        config_args_list = [
            f"--tau-alpha={tau_alpha}",
            f"--tau-beta={tau_beta}",
            f"--guidance-scale={guidance_scale}",
            f"--alpha-noise={alpha_noise}",
            f"--timesteps={timesteps}",
            f"--layers-for-injection={layers_for_injection}",
            f"--min-mask-area-ratio={min_mask_area_ratio}",
            f"--seed={seed}",
        ]
        if not inject_q: config_args_list.append("--no-inject-q")
        if not inject_k: config_args_list.append("--no-inject-k")
        if not inject_v: config_args_list.append("--no-inject-v")
        if not use_prompt_in_generation: config_args_list.append("--no-use-prompt-in-generation")

        # Add dummy required args for DitEditConfig if they are not part of the Gradio UI directly
        # These would normally come from args.bg_path etc. in run.py
        # Since we have images directly, we don't need these paths for config parsing itself.
        # However, DitEditConfig.from_args expects them.
        # We can bypass direct parsing if we manually create DitEditConfig instance.

        config = DitEditConfig(
            tau_alpha=tau_alpha,
            tau_beta=tau_beta,
            guidance_scale=guidance_scale,
            alpha_noise=alpha_noise,
            timesteps=timesteps,
            layers_for_injection=layers_for_injection,
            inject_q=inject_q,
            inject_k=inject_k,
            inject_v=inject_v,
            use_prompt_in_generation=use_prompt_in_generation,
            min_mask_area_ratio=min_mask_area_ratio,
            seed=seed
        )
        logger.info(f"Using DitEditConfig: {config}")

        # --- Image Preprocessing (similar to run.py) ---
        # Resize the original clean background image
        bg_image_resized, bg_resize_ratio = resize_image_to_max_side(bg_image_orig_clean_pil, patch_size=16)
        fg_image_resized, _ = resize_image_to_max_side(fg_image_orig, patch_size=16) # fg_image_orig defined earlier


        if bg_resize_ratio != 1.0:
            # Resize the bbox_mask_image (which was created based on original bg dimensions)
            # to match the resized background
            bbox_mask_image = bbox_mask_image.resize(bg_image_resized.size, Image.NEAREST)
        logger.info(f"Resized clean BG to {bg_image_resized.size}, FG to {fg_image_resized.size}, BBox Mask to {bbox_mask_image.size}")


        # --- Create segmentation mask for foreground ---
        logger.info("Generating segmentation mask for foreground...")
        rembg_session = new_session()
        fg_mask_pil = remove(
            fg_image_resized, # Use resized FG
            session=rembg_session,
            only_mask=True,
            alpha_matting=True,
            alpha_matting_erode_size=0,
            alpha_matting_foreground_threshold=100,
        )
        segm_mask_np = (np.array(fg_mask_pil) > 127).astype(int)

        mask_area = np.sum(segm_mask_np)
        max_mask_area = segm_mask_np.shape[0] * segm_mask_np.shape[1]
        if max_mask_area == 0:
            logger.warning("Foreground image has zero area after resize. Using full mask.")
            segm_mask_np = np.ones((fg_image_resized.height, fg_image_resized.width), dtype=int)
        elif (mask_area / max_mask_area) < config.min_mask_area_ratio:
            logger.info(f"Generated mask area ratio ({mask_area / max_mask_area:.2f}) is below threshold. Using full mask.")
            segm_mask_np = np.ones((fg_image_resized.height, fg_image_resized.width), dtype=int)
        segm_mask_image = Image.fromarray((segm_mask_np * 255).astype(np.uint8), mode="L")
        logger.info("Segmentation mask created.")

        # --- Initialize Pipeline ---
        logger.info("Initializing pipeline...")
        dtype = torch.float16
        # Consider adding model path to config or hardcoding
        pipe = EditedFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", device_map="auto", torch_dtype=dtype # Use "auto" for device map
        )
        pipe.set_progress_bar_config(disable=True)
        cached_pipe = CachedPipeline(pipe)
        logger.info("Pipeline initialized.")

        # --- Compose Noise Masks ---
        logger.info("Composing noise masks...")
        example_noise = compose_noise_masks(
            cached_pipe,
            foreground_image=fg_image_resized,
            background_image=bg_image_resized, # Use the clean, resized BG
            target_mask=bbox_mask_image,    # This is our drawn, resized bbox mask
            foreground_mask=segm_mask_image, # This is the rembg mask for FG
            option="segmentation1", # As per run.py
            num_inversion_steps=config.timesteps,
            photoshop_fg_noise=True, # As per run.py
        )
        logger.info("Noise masks composed.")

        # --- Prepare for Image Composition ---
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
            raise ValueError("Invalid value for --layers-for-injection. Use 'vital' or 'all'.")

        torch.manual_seed(config.seed)
        prompt_text = prompt if config.use_prompt_in_generation and prompt else ""
        prompts_for_pipe = ["", "", prompt_text]

        # --- Run Injection ---
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
            width=bg_image_resized.size[0],
            height=bg_image_resized.size[1],
            inverted_latents_list=list(
                zip(
                    example_noise["noise"]["background_noise_list"],
                    example_noise["noise"]["foreground_noise_list"],
                )
            ),
            tau_b=config.tau_beta,
            bg_consistency_mask=example_noise["latent_masks"]["latent_segmentation_mask"],
        )

        final_image_pil = output_image_data[0][2]
        logger.info("Image composition complete.")
        return final_image_pil, "Image generated successfully."

    except Exception as e:
        logger.error(f"Error during image generation: {e}", exc_info=True)
        return None, f"Error: {str(e)}"


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DiT-Edit: Image Composition Demo")
    gr.Markdown(
        "Upload a foreground image, a background image, draw a bounding box on the background where you want the foreground to be placed, "
        "and provide a prompt to guide the composition."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            fg_image_input = gr.Image(type="numpy", label="Foreground Image")
            # Use gr.ImageEditor for drawing. Output is a dict.
            bg_image_input = gr.ImageEditor(type="numpy", label="Background Image (Draw BBox Here)", interactive=True) # Changed from gr.Image
            prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., a cat sitting on a car")
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_image = gr.Image(type="pil", label="Composed Image")
            status_message = gr.Textbox(label="Status", interactive=False)

    with gr.Accordion("Advanced Settings (DitEditConfig)", open=False):
        # Get defaults from DitEditConfig
        default_config = DitEditConfig() # Gets all defaults

        tau_alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=default_config.tau_alpha, label="Tau Alpha (Foreground Blending/Consistency)")
        tau_beta_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=default_config.tau_beta, label="Tau Beta (Background Preservation)")
        guidance_scale_slider = gr.Slider(minimum=0.0, maximum=15.0, step=0.1, value=default_config.guidance_scale, label="Guidance Scale (CFG)")
        alpha_noise_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=default_config.alpha_noise, label="Alpha Noise (Noise in masked FG area)")
        timesteps_slider = gr.Slider(minimum=10, maximum=200, step=1, value=default_config.timesteps, label="Number of Timesteps (Diffusion Steps)")
        layers_injection_dropdown = gr.Dropdown(choices=["all", "vital"], value=default_config.layers_for_injection, label="Layers for Injection")
        
        with gr.Row():
            inject_q_checkbox = gr.Checkbox(value=default_config.inject_q, label="Inject Q")
            inject_k_checkbox = gr.Checkbox(value=default_config.inject_k, label="Inject K")
            inject_v_checkbox = gr.Checkbox(value=default_config.inject_v, label="Inject V")
        
        use_prompt_checkbox = gr.Checkbox(value=default_config.use_prompt_in_generation, label="Use Prompt in Generation")
        min_mask_area_slider = gr.Slider(minimum=0.0, maximum=0.5, step=0.01, value=default_config.min_mask_area_ratio, label="Min FG Mask Area Ratio (Fallback to full mask if below)")
        seed_number = gr.Number(value=default_config.seed, label="Seed (for reproducibility)", precision=0)


    generate_button = gr.Button("Generate Composed Image")

    generate_button.click(
        fn=generate_image,
        inputs=[
            fg_image_input,
            bg_image_input, # This will pass the dict from the 'select' tool
            prompt_input,
            tau_alpha_slider,
            tau_beta_slider,
            guidance_scale_slider,
            alpha_noise_slider,
            timesteps_slider,
            layers_injection_dropdown,
            inject_q_checkbox,
            inject_k_checkbox,
            inject_v_checkbox,
            use_prompt_checkbox,
            min_mask_area_slider,
            seed_number
        ],
        outputs=[output_image, status_message]
    )

if __name__ == "__main__":
    demo.launch()
