"""

Script to define our evaluation metrics and implement scoring of our outputs.


"""

import torch
import hpsv2

# TODO: Luca - test and finish this


def background_mse(original_image, composed_image, original_mask):    
    """
    Calculate the background mean squared error (MSE) between the original image and the composed image.
    
    Args:
        original_image (torch.Tensor): The original image tensor.
        composed_image (torch.Tensor): The composed image tensor.
        original_mask (torch.Tensor): The mask tensor for the original image.
        
    Returns:
        float: The background MSE value.
    """
    # Ensure the tensors are on the same device
    device = original_image.device
    original_image = original_image.to(device)
    composed_image = composed_image.to(device)
    original_mask = original_mask.to(device)
    # Calculate the background mask
    background_mask = 1 - original_mask
    # Calculate the background MSE
    background_mse = torch.mean((original_image - composed_image) ** 2 * background_mask)
    return background_mse.item()

def foreground_semantic_clip(original_image, composed_image, foreground_mask):
    """
    Calculate the foreground semantic similarity between the original image and the composed image.
    
    Args:
        original_image (torch.Tensor): The original image tensor.
        composed_image (torch.Tensor): The composed image tensor.
        original_mask (torch.Tensor): The mask tensor for the original image.
        
    Returns:
        float: The foreground semantic similarity value.
    """
    # Ensure the tensors are on the same device
    device = original_image.device
    original_image = original_image.to(device)
    composed_image = composed_image.to(device)
    original_mask = original_mask.to(device)
    
    # Calculate the foreground mask
    foreground_mask = original_mask
    
    # Calculate the foreground semantic similarity
    # TODO: Luca
    
    return


def hpsv2_score(composed_image, prompt):
    """
    Calculate the HPSV2 score for the composed image.
    
    Args:
        composed_image (torch.Tensor): The composed image tensor.
        
    Returns:
        float: The HPSV2 score value.
    """
    
    # Calculate the HPSV2 score
    hpsv2_score = hpsv2.score(composed_image, prompt, hps_version="v2.1")
    
    return hpsv2_score