
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch

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

