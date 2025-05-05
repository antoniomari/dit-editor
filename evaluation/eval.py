"""

Script to define our evaluation metrics and implement scoring of our outputs.


"""
from typing import Union
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from transformers import CLIPProcessor, CLIPModel

from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from data.benchmark_data import BenchmarkExample


def calculate_all_scores(images, num_samples=None, output_file="scores.csv"):
    """
    Calculate and save scores for benchmark images.
    
    Args:
        images: List of benchmark examples to evaluate
        num_samples: Number of samples to process (default: all)
        output_file: Path to save CSV results (default: "scores.csv")
        
    Returns:
        pandas.DataFrame: DataFrame containing all calculated scores
    """

    
    # Determine number of examples to process
    num_examples = num_samples if num_samples is not None else len(images)
    num_examples = min(num_examples, len(images))
    
    print(f"Processing {num_examples} benchmark examples...")
    all_scores = []
    
    # Process each image
    for i, image in tqdm(enumerate(images[:num_examples]), total=num_examples):
        try:
            # Get scores for this example
            example_score_dict = get_scores_for_single_example(image)
            
            # Unpack the nested dictionary
            for model_type, metrics in example_score_dict.items():
                row = {
                    'image_index': i, 
                    'model_type': model_type, 
                    'category': image.category
                }
                
                # Add all metrics as separate columns
                for metric_name, value in metrics.items():
                    # Convert tensor values to float for better display
                    if hasattr(value, 'item'):
                        value = value.item()
                    row[metric_name] = value
                    
                all_scores.append(row)
                
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    # Create DataFrame from collected data
    if not all_scores:
        print("No scores were successfully calculated.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_scores)
    
    # Reorder columns to have image_index and model_type first
    columns = ['image_index', 'model_type', 'category'] + [
        col for col in df.columns if col not in ['image_index', 'model_type', 'category']
    ]
    
    df = df[columns]
    
    # Save results if an output file is specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Scores saved to {output_file}")
    
    return df


def get_scores_for_single_example(example: BenchmarkExample):
    """ function calculate all the scoring metrics given a single benchmark example.
    It will look through all created images (baselines and new methods) and calculate the scores for each of them.
    
    Args:
        example (BenchmarkExample): The benchmark example to score.
    """
    # TODO: refactor function based on its final usage

    score_dict = defaultdict(dict)
    # check if all images are present

    if example.result_image: # "Photoshop Baseline"
        # calculate the score
        hpsv2_score = compute_hpsv2_score(example.result_image, example.prompt)
        aesthetics_score = compute_aesthetics_score(example.result_image)

        background_mse = compute_background_mse(example.bg_image, example.result_image, example.target_mask)
        clip_text_image = compute_clip_similarity(example.result_image, example.prompt)
        
        dinov2_similarity = compute_dinov2_similarity(example.result_image, example.fg_image, example.fg_mask) 

        score_dict["Photoshop"] = {
                            "hpsv2_score": hpsv2_score,
                            "aesthetics_score": aesthetics_score,
                            "background_mse": background_mse,
                            "clip_text_image": clip_text_image,
                            "dinov2_similarity": dinov2_similarity
                                        }
    if example.tf_icon_image:
        # calculate the score
        hpsv2_score = compute_hpsv2_score(example.tf_icon_image, example.prompt)
        aesthetics_score = compute_aesthetics_score(example.tf_icon_image)
        background_mse = compute_background_mse(example.bg_image, example.tf_icon_image, example.target_mask)
        clip_text_image = compute_clip_similarity(example.tf_icon_image, example.prompt)
        # TODO: how to compute dinov2 score for our purposes?
        dinov2_similarity = compute_dinov2_similarity(example.tf_icon_image, example.fg_image, example.fg_mask)

        score_dict["TF-ICON"] = {
                            "hpsv2_score": hpsv2_score,
                            "aesthetics_score": aesthetics_score,
                            "background_mse": background_mse,
                            "clip_text_image": clip_text_image,
                            "dinov2_similarity": dinov2_similarity
                                        }

    example.score_dict = score_dict

    return score_dict


def compute_background_mse(
    image1_pil: Image.Image, 
    image2_pil: Image.Image, 
    mask: Union[np.ndarray, torch.Tensor], 
) -> float:
    """
    Calculate the masked mean squared error (MSE) between two images, using a mask to exclude certain pixels.
    Args:
        
        image1_pil (PIL.Image.Image): The first image (original).
        image2_pil (PIL.Image.Image): The second image (composed).
        mask (Union[np.ndarray, torch.Tensor]): The representing the foreground in the second image (i.e., where to put the new object in the first image)."""
    # Convert images to float32 numpy arrays, normalized [0, 1]
    img1 = np.asarray(image1_pil).astype(np.float32) / 255.0
    img2 = np.asarray(image2_pil).astype(np.float32) / 255.0

    # assert mask is binary and numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask).astype(np.bool)


    # Invert mask: 1 = exclude → 0; 0 = include → 1
    include_mask = 1.0 - mask # represents the background

    # Compute squared difference
    diff_squared = (img1 - img2) ** 2
    masked_diff = diff_squared * include_mask[:, :, np.newaxis]  # Apply mask to each channel

    # Sum and normalize by valid (included) pixels
    valid_pixel_count = np.sum(include_mask)
    if valid_pixel_count == 0:
        raise ValueError("All pixels are masked out. Cannot compute MSE.")

    mse = np.sum(masked_diff) / valid_pixel_count
    return float(mse)


def compute_clip_similarity(image: Image.Image, prompt: str) -> float:
    """
    Compute CLIP similarity between a PIL image and a text prompt.
    Loads CLIP model only once and caches it.
    
    Args:
        image (PIL.Image.Image): Input image.
        prompt (str): Text prompt.
    
    Returns:
        float: Cosine similarity between image and text.
    """
    if not hasattr(compute_clip_similarity, "model"):
        compute_clip_similarity.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        compute_clip_similarity.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        compute_clip_similarity.model.eval()

    model = compute_clip_similarity.model
    processor = compute_clip_similarity.processor

    image = image.convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=[prompt], return_tensors="pt")


    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = (image_features @ text_features.T).item()

    return similarity


def compute_dinov2_similarity(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image
) -> float:
    """
    Compute perceptual similarity between two images using DINOv2 embeddings,
    applying a mask to select a region of the first image (image1).

    Args:
        image1 (PIL.Image.Image): Result image from which a masked region will be extracted.
        image2 (PIL.Image.Image): Reference image.
        mask (PIL.Image.Image): Binary mask (mode '1' or 'L') selecting the region on image1.

    Returns:
        float: Cosine similarity between DINOv2 embeddings of the masked-and-resized region of image1 and image2.
    """
    # Initialize model and processor once
    if not hasattr(compute_dinov2_similarity, "model"):
        compute_dinov2_similarity.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        compute_dinov2_similarity.model = AutoModel.from_pretrained("facebook/dinov2-base")
        compute_dinov2_similarity.model.eval()

    processor = compute_dinov2_similarity.processor
    model = compute_dinov2_similarity.model

    # Ensure mask is same size as image1
    if mask.size != image1.size:
        # NOTE: for some reason, final mask usually has drastically different size.
        mask = mask.resize(image1.size, resample=Image.NEAREST)

    # Convert mask to binary and get bounding box of masked region
    mask_binary = mask.convert("L").point(lambda p: p > 0 and 255)
    bbox = mask_binary.getbbox()
    if bbox is None:
        raise ValueError("Mask contains no non-zero region to compare.")

    # Crop the masked region and resize back to original size
    region = image1.crop(bbox)
    region = region.resize(image1.size, resample=Image.BILINEAR)

    # Prepare images for model: region (as image1) and reference (image2)
    img1_rgb = region.convert("RGB")
    img2_rgb = image2.convert("RGB")

    # Preprocess images
    inputs = processor(images=[img1_rgb, img2_rgb], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # Mean-pool token embeddings
        features = outputs.last_hidden_state.mean(dim=1)
        features = F.normalize(features, p=2, dim=-1)
        # Cosine similarity
        similarity = torch.matmul(features[0], features[1]).item()

    return similarity


def compute_hpsv2_score(composed_image: Image.Image, prompt):
    """
    Calculate the HPSV2 score for the composed image.
    
    Args:
        composed_image (torch.Tensor): The composed image tensor.
        
    Returns:
        float: The HPSV2 score value.
    """
    # from hpsv2.src.open_clip.factory import create_model_and_transforms, get_tokenizer
    from .hpsv2_open_clip_master.factory import create_model_and_transforms, get_tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hasattr(compute_hpsv2_score, "model"):
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        compute_hpsv2_score.model = model
        compute_hpsv2_score.tokenizer = get_tokenizer('ViT-H-14')
        compute_hpsv2_score.preprocess_val = preprocess_val
        compute_hpsv2_score.model.eval()


    # Calculate the score for the given image and prompt
    with torch.no_grad():
        # Process the image
        image = compute_hpsv2_score.preprocess_val(composed_image).unsqueeze(0).to(device=device, non_blocking=True)
        # Process the prompt
        text = compute_hpsv2_score.tokenizer([prompt]).to(device=device, non_blocking=True)
        # Calculate the HPS
        with torch.no_grad():
            outputs = compute_hpsv2_score.model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        
    return hps_score[0]
    

def compute_aesthetics_score(composed_image: Image.Image):
    """
    Calculate the aesthetics score for the composed image.
    Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
    NOTE: need to download the model weights from the above repo and put them in the same directory as this script.

    Args:
        composed_image (torch.Tensor): The Pil image
        
    Returns:
        float: The aesthetics score value.
    """
    from .aesthetics_model import get_aesthetic_model, normalized
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # only load the model once
    if not hasattr(compute_aesthetics_score, "model"):
        model, clip_model, preprocess = get_aesthetic_model(device=device)
        compute_aesthetics_score.model = model
        compute_aesthetics_score.clip_model = clip_model
        compute_aesthetics_score.preprocess = preprocess

    # load image.
    # pil_image = Image.open(img_path)
    pil_image = composed_image # NOTE: put potential transofrmations here

    # preprocess image
    image = compute_aesthetics_score.preprocess(images=pil_image, return_tensors="pt").to(device)
    

    # predict aesthetics score
    with torch.no_grad():
        # image_features = compute_aesthetics_score.clip_model.encode_image(image)
        image_features = compute_aesthetics_score.clip_model.get_image_features(**image)
    im_emb_arr = normalized(image_features.cpu().detach().numpy() )
    prediction = compute_aesthetics_score.model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    return prediction
