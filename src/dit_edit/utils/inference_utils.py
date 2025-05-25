from typing import Optional
from PIL import Image
import numpy as np


def resize_image_to_max_side(image, max_side=1024, patch_size: Optional[int] = None):
    """
    Resizes an image so its longest side is at most max_side, maintaining aspect ratio.
    Returns the resized image and the resize ratio. If a patch_size is provided, then the image is 
    resized so that both dimensions are multiples of patch_size.
    """
    w, h = image.size
    if max(w, h) <= max_side:
        return image, 1.0

    if w > h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))

    resize_ratio = new_w / w if w > h else new_h / h
    resized_image = image.resize((new_w, new_h), Image.LANCZOS)

    if patch_size is not None:
        # Ensure the resized dimensions are multiples of patch_size
        new_w = (new_w // patch_size) * patch_size
        new_h = (new_h // patch_size) * patch_size
        resized_image = resized_image.resize((new_w, new_h), Image.LANCZOS)
        resize_ratio = new_w / w if w > h else new_h / h

    return resized_image, resize_ratio


def get_bbox_from_mask_image(mask_image_pil: Image.Image):
    """
    Extracts bounding box coordinates from a mask image.
    The mask image should be black with a single white rectangle.
    Returns [x_min, y_min, x_max, y_max] (exclusive end coordinates).
    """
    if mask_image_pil.mode != 'L' and mask_image_pil.mode != '1':
        # Convert to grayscale if it's not already L or 1
        mask_image_pil = mask_image_pil.convert('L')

    # Ensure it's a binary mask (0 or 255)
    mask_np = np.array(mask_image_pil)
    if not ((mask_np == 0) | (mask_np == 255)).all():
        # If not strictly 0 or 255, threshold it (e.g., common for L mode from various sources)
        # Assuming white is high value, black is low.
        threshold = 128 # Common threshold
        mask_np = ((mask_np > threshold) * 255).astype(np.uint8)

    if mask_np.ndim == 3: # Should be 2D, take first channel if it's grayscale but still 3D
        mask_np = mask_np[:, :, 0]

    white_pixels = np.where(mask_np == 255)
    if white_pixels[0].size == 0 or white_pixels[1].size == 0:
        raise ValueError("No white pixels found in the bounding box mask image.")

    y_min = int(white_pixels[0].min())
    y_max = int(white_pixels[0].max() + 1)
    x_min = int(white_pixels[1].min())
    x_max = int(white_pixels[1].max() + 1)

    return [x_min, y_min, x_max, y_max]