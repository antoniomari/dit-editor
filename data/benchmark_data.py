
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import warnings
from typing import List

class BenchmarkExample:
    def __init__(self, image_path: str):
        self.image_path = image_path # path to folder with prompt

        base_folder = os.path.basename(image_path)
        self.category = os.path.dirname(image_path).split(os.sep)[-1] # get the category name

        # extract image number and prompt
        self.image_number = base_folder[:4]
        self.prompt = base_folder[5:]

        all_images = os.listdir(image_path)
        
        # extract load all individual parts of the benchmark
        self.bg_image = None
        self.fg_image = None
        self.fg_mask = None
        self.result_image = None
        self.target_mask = None
        self.final_mask = None
        self.tf_icon_image = None
        self.output = None
        self.testing_image = None
        
        for img_file in all_images:
            if img_file.startswith('bg'):
                self.bg_image = os.path.join(image_path, img_file)
            elif img_file.startswith('fg') and img_file.endswith('.jpg') and not 'mask' in img_file:
                self.fg_image = os.path.join(image_path, img_file)
            elif img_file.startswith('fg') and 'mask' in img_file:
                self.fg_mask = os.path.join(image_path, img_file)
            elif img_file == 'cp_bg_fg.jpg':
                self.result_image = os.path.join(image_path, img_file)
            elif img_file == 'mask_bg_fg.jpg':
                self.target_mask = os.path.join(image_path, img_file)
            elif img_file == 'dccf_image.jpg':
                self.final_mask = os.path.join(image_path, img_file)
            
            elif img_file == 'tf-icon.png':
                self.tf_icon_image = os.path.join(image_path, img_file)
            elif img_file == "kvedit.jpg":
                self.kvedit_image = os.path.join(image_path, img_file)
            elif img_file == "alphanoise0.05_timesteps50_QTrue_KTrue_VFalse_taua0.4_taub0.8_guidance3.0_all-layers.png":
                self.testing_image = os.path.join(image_path, img_file)

        # check if all images are present
        if not all([self.bg_image, self.fg_image, self.fg_mask, self.result_image, self.target_mask, self.final_mask]):
            # print which images are missing
            missing_images = []
            if not self.bg_image:
                missing_images.append('bg_image')
            if not self.fg_image:
                missing_images.append('fg_image')
            if not self.fg_mask:
                missing_images.append('fg_mask')
            if not self.result_image:
                missing_images.append('result_image')
            if not self.target_mask:
                missing_images.append('target_mask')
            if not self.final_mask:
                missing_images.append('final_mask')
            print(f"Missing images in {image_path}: {', '.join(missing_images)}")
            raise ValueError(f"Not all images are present in {image_path}")
            
        
        # load images
        self.bg_image = Image.open(self.bg_image).convert("RGB")
        self.fg_image = Image.open(self.fg_image).convert("RGB")
        self.fg_mask = Image.open(self.fg_mask).convert("L")
        self.result_image = Image.open(self.result_image).convert("RGB")
        self.target_mask = Image.open(self.target_mask).convert("L")
        self.final_mask = Image.open(self.final_mask).convert("L")

        # load results
        if self.tf_icon_image:
            self.tf_icon_image = Image.open(self.tf_icon_image).convert("RGB")
        if self.kvedit_image:
            self.kvedit_image = Image.open(self.kvedit_image).convert("RGB")
        if self.testing_image:
            self.testing_image = Image.open(self.testing_image).convert("RGB")

        # TODO: Load more results if we have them
    
    def set_output(self, output: Image.Image):
        """
        Set the output image for the benchmark example.
        """
        self.output = output

    def plot_results(self):
        """ Plot background, foreground and all results in a single row. """

        # Determine how many images we have (base images + results)
        num_images = 3  # bg, fg, result, final_mask
        if hasattr(self, 'tf_icon_image') and self.tf_icon_image:
            num_images += 1
        if hasattr(self, 'kvedit_image') and self.kvedit_image:
            num_images += 1

        # Create a new figure with a single row
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

        # Display base images
        axes[0].imshow(self.bg_image)
        axes[0].set_title("Background")
        axes[0].axis("off")

        axes[1].imshow(self.fg_image)
        axes[1].set_title("Foreground")
        axes[1].axis("off")

        axes[2].imshow(self.result_image)
        axes[2].set_title("Naive Composition")
        axes[2].axis("off")

        # Display additional results if available
        idx = 3
        if hasattr(self, 'tf_icon_image') and self.tf_icon_image:
            axes[idx].imshow(self.tf_icon_image)
            axes[idx].set_title("TF-Icon Result")
            axes[idx].axis("off")
            idx += 1

        if hasattr(self, 'kvedit_image') and self.kvedit_image:
            axes[idx].imshow(self.kvedit_image)
            axes[idx].set_title("KVEdit Result")
            axes[idx].axis("off")

        # Set the title
        fig.suptitle(f"Category: {self.category}, Image: {self.image_number}\nPrompt: {self.prompt}", fontsize=12)
        plt.tight_layout()
        plt.show()



    def plot_sample(self):
        """
        Plot a sample from the benchmark example with masks in a separate row.
        """
        # Create a new figure with 2 rows
        fig, ax = plt.subplots(2, 3, figsize=(18, 12))

        # First row: Original images
        ax[0, 0].imshow(self.bg_image)
        ax[0, 0].set_title("Background Image")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(self.fg_image)
        ax[0, 1].set_title("Foreground Image")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(self.result_image)
        ax[0, 2].set_title("Result Image")
        ax[0, 2].axis("off")

        # Second row: Mask images
        ax[1, 0].imshow(self.target_mask, cmap='gray')
        ax[1, 0].set_title("Foreground Mask")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(self.fg_mask, cmap='gray')
        ax[1, 1].set_title("Target Mask")
        ax[1, 1].axis("off")

        ax[1, 2].imshow(self.final_mask, cmap='gray')
        ax[1, 2].set_title("Final Mask")
        ax[1, 2].axis("off")

        # Set the title
        fig.suptitle(f"Category: {self.category}, Image: {self.image_number}\nPrompt: {self.prompt}", fontsize=12)
        plt.tight_layout()
        plt.show()

def gather_images(image_dir: str) -> List[BenchmarkExample]:
    """
    Gather all images in the given directory and return a list of BenchmarkExample objects.
    """
    images = {}
    missing_image_counter = 0

    # Get all directories in the image_dir
    for category in os.listdir(image_dir):
        images[category] = []
        category_path = os.path.join(image_dir, category)

        if not os.path.isdir(category_path): # skip .DS_Store
            continue
        
        for image_folder in os.listdir(category_path):
            image_folder_path = os.path.join(category_path, image_folder)
                
            # Check if it's a directory
            if not os.path.isdir(image_folder_path):# skip .DS_Store
                continue

            # Check if the folder name is a valid image folder
            try:
                images[category].append(BenchmarkExample(image_folder_path))
            except ValueError as e:
                print(f"Skipping {image_folder_path}: {e}")
                missing_image_counter += 1

    print(f"Loaded {len(images)} images in {image_dir}")
    # Warn if any images were missing     
    warnings.warn(f"Missing {missing_image_counter} images in {image_dir}")

    return images