from typing import Optional, Tuple
import torch
import torchvision.transforms.functional as TF

from cache_and_edit import CachedPipeline

def image2latent(pipe, image, latent_nudging_scalar = 1.15):
    image = pipe.image_processor.preprocess(image).type(pipe.vae.dtype).to("cuda")
    latents = pipe.vae.encode(image)["latent_dist"].mean
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents * latent_nudging_scalar

    latents = pipe._pack_latents(
        latents=latents,
        batch_size=1,
        num_channels_latents=16,
        height=image.size(2) // 8,
        width= image.size(3) // 8
    )

    return latents


def get_inverted_input_noise(pipe: CachedPipeline, image, num_steps: int = 28):
    """_summary_

    Args:
        pipe (CachedPipeline): _description_
        image (_type_): _description_
        num_steps (int, optional): _description_. Defaults to 28.

    Returns:
        _type_: _description_
    """

    width, height = image.size 

    noise = pipe.run(
        "",
        num_inference_steps=num_steps,
        seed=42,
        guidance_scale=1.5,
        output_type="latent",
        latents=image2latent(pipe.pipe, image),
        inverse=True,
        width=width,
        height=height
    ).images[0]

    return noise


def resize_bounding_box(
    bb_mask: torch.Tensor,
    target_size: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """
    Given a bounding box mask, patches it into a mask with the target size.
    The mask is a 2D tensor of shape (H, W) where each element is either 0 or 1.
    Any patch that contains at least one 1 in the original mask will be set to 1 in the output mask.

    Args:
        bb_mask (torch.Tensor): The bounding box mask as a boolean tensor of shape (H, W).
        target_size (Tuple[int, int]): The size of the target mask as a tuple (H, W).

    Returns:
        torch.Tensor: The resized bounding box mask as a boolean tensor of shape (H, W).
    """
    
    w_mask, h_mask = bb_mask.shape[-2:]
    w_target, h_target = target_size

    # Make sure the sizes are compatible
    if w_mask % w_target != 0 or h_mask % h_target != 0:
        raise ValueError(
            f"Mask size {bb_mask.shape[-2:]} is not compatible with target size {target_size}"
        )
    
    # Compute the size of a patch
    patch_size = (w_mask // w_target, h_mask // h_target)

    # Iterate over the mask, one patch at a time, and save a 0 patch if the patch is empty or a 1 patch if the patch is not empty
    out_mask = torch.zeros((w_target, h_target), dtype=bb_mask.dtype, device=bb_mask.device)
    for i in range(w_target):
        for j in range(h_target):
            patch = bb_mask[
                i * patch_size[0] : (i + 1) * patch_size[0],
                j * patch_size[1] : (j + 1) * patch_size[1],
            ]
            if torch.sum(patch) > 0:
                out_mask[i, j] = 1
            else:
                out_mask[i, j] = 0

    return out_mask


def place_image_in_bounding_box(
    image_tensor_whc: torch.Tensor, 
    mask_tensor_wh: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resizes an input image to fit within a bounding box (from a mask)
    preserving aspect ratio, and places it centered on a new canvas.

    Args:
        image_tensor_whc: Input image tensor, shape [width, height, channels].
        mask_tensor_wh: Bounding box mask, shape [width, height]. Defines canvas size
                          and contains a rectangle of 1s for the BB.

    Returns:
        A tuple:
        - output_image_whc (torch.Tensor): Canvas with the resized image placed.
                                           Shape [canvas_width, canvas_height, channels].
        - new_mask_wh (torch.Tensor): Mask showing the actual placement of the image.
                                      Shape [canvas_width, canvas_height].
    """
    
    # Validate input image dimensions
    if not (image_tensor_whc.ndim == 3 and image_tensor_whc.shape[0] > 0 and image_tensor_whc.shape[1] > 0):
        raise ValueError(
            "Input image_tensor_whc must be a 3D tensor [width, height, channels] "
            "with width > 0 and height > 0."
        )
    img_orig_w, img_orig_h, num_channels = image_tensor_whc.shape

    # Validate mask tensor dimensions
    if not (mask_tensor_wh.ndim == 2):
        raise ValueError("Input mask_tensor_wh must be a 2D tensor [width, height].")
    canvas_w, canvas_h = mask_tensor_wh.shape

    # Prepare default empty outputs for early exit scenarios
    empty_output_image = torch.zeros(
        canvas_w, canvas_h, num_channels, 
        dtype=image_tensor_whc.dtype, device=image_tensor_whc.device
    )
    empty_new_mask = torch.zeros(
        canvas_w, canvas_h, 
        dtype=mask_tensor_wh.dtype, device=mask_tensor_wh.device
    )

    # 1. Find Bounding Box (BB) coordinates from the input mask_tensor_wh
    #    fg_coords shape: [N, 2], where N is num_nonzero. Each row: [x_coord, y_coord].
    fg_coords = torch.nonzero(mask_tensor_wh, as_tuple=False) 
    
    if fg_coords.numel() == 0: # No bounding box found in mask
        return empty_output_image, empty_new_mask

    # Determine min/max extents of the bounding box
    x_min_bb, y_min_bb = fg_coords[:, 0].min(), fg_coords[:, 1].min()
    x_max_bb, y_max_bb = fg_coords[:, 0].max(), fg_coords[:, 1].max()

    bb_target_w = x_max_bb - x_min_bb + 1
    bb_target_h = y_max_bb - y_min_bb + 1

    if bb_target_w <= 0 or bb_target_h <= 0: # Should not happen if fg_coords not empty
        return empty_output_image, empty_new_mask

    # 2. Prepare image for resizing: TF.resize expects [C, H, W]
    #    Input image_tensor_whc is [W, H, C]. Permute to [C, H_orig, W_orig].
    image_tensor_chw = image_tensor_whc.permute(2, 1, 0) 

    # 3. Calculate new dimensions for the image to fit in BB, preserving aspect ratio
    scale_factor_w = bb_target_w / img_orig_w
    scale_factor_h = bb_target_h / img_orig_h
    scale = min(scale_factor_w, scale_factor_h) # Fit entirely within BB

    resized_img_w = int(img_orig_w * scale)
    resized_img_h = int(img_orig_h * scale)
    
    if resized_img_w == 0 or resized_img_h == 0: # Image scaled to nothing
        return empty_output_image, empty_new_mask
        
    # 4. Resize the image. TF.resize expects size as [H, W].
    try:
        # antialias=True for better quality (requires torchvision >= 0.8.0 approx)
        resized_image_chw = TF.resize(image_tensor_chw, [resized_img_h, resized_img_w], antialias=True)
    except TypeError: # Fallback for older torchvision versions
        resized_image_chw = TF.resize(image_tensor_chw, [resized_img_h, resized_img_w])

    # Permute resized image back to [W, H, C] format
    resized_image_whc = resized_image_chw.permute(2, 1, 0)

    # 5. Create the output canvas image (initialized to zeros)
    output_image_whc = torch.zeros(
        canvas_w, canvas_h, num_channels, 
        dtype=image_tensor_whc.dtype, device=image_tensor_whc.device
    )

    # 6. Calculate pasting coordinates to center the resized image within the original BB
    offset_x = (bb_target_w - resized_img_w) // 2
    offset_y = (bb_target_h - resized_img_h) // 2

    paste_x_start = x_min_bb + offset_x
    paste_y_start = y_min_bb + offset_y

    paste_x_end = paste_x_start + resized_img_w
    paste_y_end = paste_y_start + resized_img_h
    
    # Place the resized image onto the canvas
    output_image_whc[paste_x_start:paste_x_end, paste_y_start:paste_y_end, :] = resized_image_whc

    # 7. Create the new mask representing where the image was actually placed
    new_mask_wh = torch.zeros(
        canvas_w, canvas_h, 
        dtype=mask_tensor_wh.dtype, device=mask_tensor_wh.device
    )
    new_mask_wh[paste_x_start:paste_x_end, paste_y_start:paste_y_end] = 1

    return output_image_whc, new_mask_wh