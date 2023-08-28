import torch
import requests
import os
import cv2
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
import sys
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class ImageProcessor:
    def __init__(self):
        sam_checkpoint = "/mnt/c/Users/indue/Downloads/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        device = "cuda"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        self.mask_generator = SamAutomaticMaskGenerator(sam)
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
    
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
    
    def create_masked_images(self, image, output_dir, mask_generator):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Generate the mask
        images = []
        masks = mask_generator.generate(image)
    
        # Apply each mask on the image
        for i, ann in enumerate(masks):
            mask = ann['segmentation']
            bbox = ann['bbox']
            # print(ann['bbox'])
            # Create an empty black image with the same size
            bbox_image = self.extract_bbox(image, ann['bbox'])
            print(bbox_image.shape)
            masked_image = np.zeros_like(image)
            
            # Ensure the mask is binary
            binary_mask = np.where(mask > 0, 1, 0)
            
            # Mask the image
            for c in range(3):  # For each color channel
                masked_image[:, :, c] = image[:, :, c] * binary_mask
            filename = "mask_{}.png".format(i)
            # Save the masked image to the output folder
            output_path = os.path.join(output_dir, filename)
            # masked_image_pil = Image.fromarray(masked_image)
            masked_image_pil = Image.fromarray(bbox_image)
            masked_image_pil.save(output_path)
            images.append(preprocess(masked_image_pil))
        return images
    
    def create_candidate_masks(self, frame_range):
        start_frame = frame_range["start"]
        end_frame = frame_range["end"]

        candidate_frame = (end_frame - start_frame) // 2 
        image = cv2.imread(f'frame_{candidate_frame}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.create_masked_images(image, f'frame_{candidate_frame}_masks', self.mask_generator)

    def extract_bbox(self, img, bbox):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        return img[y:y+h, x:x+w]

def main():    
    image_processor = ImageProcessor()
    