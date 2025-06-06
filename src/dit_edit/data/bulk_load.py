import logging
import os
import warnings
from typing import List, Optional

from dit_edit.data.benchmark_data import BenchmarkExample


def load_benchmark_data(
    image_dir: str, logger: Optional[logging.Logger] = None
) -> List[BenchmarkExample]:
    """
    Gather all images in the given directory and return a list of BenchmarkExample objects.
    """
    images = {}
    missing_image_counter = 0

    if logger is None:
        logger = logging.getLogger(__name__)

    # Get all directories in the image_dir
    for category in os.listdir(image_dir):
        images[category] = []
        category_path = os.path.join(image_dir, category)

        if not os.path.isdir(category_path):  # skip .DS_Store
            continue

        for image_folder in os.listdir(category_path):
            image_folder_path = os.path.join(category_path, image_folder)

            # Check if it's a directory
            if not os.path.isdir(image_folder_path):  # skip .DS_Store
                continue

            # Check if the folder name is a valid image folder
            try:
                images[category].append(BenchmarkExample(image_folder_path))
            except ValueError as e:
                logger.warning(f"Skipping {image_folder_path}: {e}")
                missing_image_counter += 1

    logger.info(f"Loaded {len(images)} images in {image_dir}")
    logger.warning(f"Missing {missing_image_counter} images in {image_dir}")

    return images
