import clip
import json
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
from helpers import Timecode
class ImageProcessor:
    def __init__(self):
        sam_checkpoint = "/mnt/c/Users/indue/Downloads/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.model, self.preprocess = clip.load("ViT-B/32")
        self.preprocess

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

    def extract_candidate_frame_masks(self, frame_range):
        start_frame = frame_range["start"]
        end_frame = frame_range["end"]

        candidate_frame = (end_frame.convert_timecode_to_sec() + start_frame.convert_timecode_to_sec()) // 2
        FRAME_DIR = os.path.join("Data", "Frame.{}.0".format(int(candidate_frame)))

        MASK_METADATA_FILE = os.path.join(FRAME_DIR, "mask_data.txt")
        IMAGE_FILE = os.path.join(FRAME_DIR, "frame.jpg")
        # load image + bboxes
        image = cv2.imread(IMAGE_FILE)
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(MASK_METADATA_FILE, 'r') as f:
            data = f.read()
            mask_data = data.strip().split("\n")
            mask_data = [json.loads(mask) for mask in mask_data]
            print(mask_data[0])

        input_images = []
        input_bboxes = []

        for annotation in mask_data:
            crop = self.extract_bbox(image, annotation['bbox'])
            masked_image_pil = Image.fromarray(crop)
            input_images.append(self.preprocess(masked_image_pil))
            input_bboxes.append(annotation['bbox'])

        return input_images, input_bboxes, candidate_frame

    def extract_related_crop(self, input_text, input_bboxes, input_images, frame_id):
        images = torch.tensor(np.stack(input_images)).cuda()
        text = clip.tokenize(input_text).cuda()

        with torch.no_grad():
            image_features = self.model.encode_image(images).float()
            text_features = self.model.encode_text(text).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        ind = np.unravel_index(np.argmax(similarity), similarity.shape)
        return input_bboxes[ind[0]]

    def extract_bbox(self, img, bbox):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        return img[y:y+h, x:x+w]

def main():    
    image_processor = ImageProcessor()
    