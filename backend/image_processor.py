import clip
import json
import torch
import requests
import os
import cv2
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt
from transformers import * # CLIPTokenizer, CLIPModel, CLIPTextModel 
import sys
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from .helpers import Timecode

SEGMENTATION_DATA_PATH = "segmentation-data"

class ImageProcessor:
    def __init__(self):
        # sam_checkpoint = "/mnt/c/Users/indue/Downloads/sam_vit_h_4b8939.pth"
        # model_type = "vit_h"
        
        # sam_kwargs = {
        #     "points_per_side": 32,
        #     "pred_iou_thresh": 0.86,
        #     "stability_score_thresh": 0.92,
        #     "crop_n_layers": 1,
        #     "crop_n_points_downscale_factor": 2,
        #     "min_mask_region_area": 100,
        # }
        # device = "cuda"
        
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)
        
        self.model, self.preprocess = clip.load("ViT-B/32")
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model=sam,
        #     points_per_side= sam_kwargs["points_per_side"],
        #     pred_iou_thresh=sam_kwargs["pred_iou_thresh"],
        #     stability_score_thresh=sam_kwargs["stability_score_thresh"],
        #     crop_n_layers=sam_kwargs["crop_n_layers"],
        #     crop_n_points_downscale_factor=sam_kwargs["crop_n_points_downscale_factor"],
        #     min_mask_region_area=sam_kwargs["min_mask_region_area"],  # Requires open-cv to run post-processing
        # )

    # def show_anns(self, anns, image):
    #     if len(anns) == 0:
    #         return
    #     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    #     mask = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    #     mask[:,:,3] = 0
    #     for ann in sorted_anns:
    #         m = ann['segmentation']
    #         color_mask = np.concatenate([np.random.random(3), [0.35]])
    #         mask[m] = color_mask

    #     mask = (mask * 255).astype(np.uint8)
    #     image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
    #     colored_image = cv2.addWeighted(image_with_alpha, 1, mask, 1, 0)
    #     colored_image = (colored_image * 255).astype(np.uint8)

    #     cv2.imwrite('highlighted_img.jpg', colored_image[:,:, :3])

    # def create_masks(self, image):
    #     masks = self.mask_generator.generate(image)
    #     self.show_anns(masks, image)

    # def create_masked_images(self, image, output_dir):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     # Generate the mask
    #     images = []
    #     masks = self.mask_generator.generate(image)
    
    #     # Apply each mask on the image
    #     for i, ann in enumerate(masks):
    #         mask = ann['segmentation']
    #         bbox = ann['bbox']
    #         # print(ann['bbox'])
    #         # Create an empty black image with the same size
    #         bbox_image = self.extract_bbox(image, ann['bbox'])
    #         print(bbox_image.shape)
    #         masked_image = np.zeros_like(image)
            
    #         # Ensure the mask is binary
    #         binary_mask = np.where(mask > 0, 1, 0)
            
    #         # Mask the image
    #         for c in range(3):  # For each color channel
    #             masked_image[:, :, c] = image[:, :, c] * binary_mask
    #         filename = "mask_{}.png".format(i)
    #         # Save the masked image to the output folder
    #         output_path = os.path.join(output_dir, filename)
    #         # masked_image_pil = Image.fromarray(masked_image)
    #         masked_image_pil = Image.fromarray(bbox_image)
    #         masked_image_pil.save(output_path)
    #         images.append(transformers.preprocess(masked_image_pil))
    #     return images
    
    def extract_candidate_frame_masks(self, frame_range, video_id):
        start_frame = frame_range["start"]
        end_frame = frame_range["end"]
        
        candidate_frame = (end_frame.convert_timecode_to_sec() + start_frame.convert_timecode_to_sec()) // 2 
        FRAME_DIR = os.path.join(SEGMENTATION_DATA_PATH, video_id, "Frame.{}.0".format(int(candidate_frame)))

        MASK_METADATA_FILE = os.path.join(FRAME_DIR, "mask_data.txt")
        IMAGE_FILE = os.path.join(FRAME_DIR, "frame.jpg")    
        # load image + bboxes
        image = cv2.imread(IMAGE_FILE)
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(MASK_METADATA_FILE, 'r') as f:
            data = f.read()
            mask_data = data.strip().split("\n")
            mask_data = [json.loads(mask) for mask in mask_data]

        input_images = []
        input_bboxes = []

        for annotation in mask_data:
            crop = self.extract_bbox(image, annotation['bbox'])
            masked_image_pil = Image.fromarray(crop)
            input_images.append(self.preprocess(masked_image_pil))
            input_bboxes.append(annotation['bbox'])            

        return input_images, input_bboxes, candidate_frame, image

    def extract_related_crop(self, input_text, input_bboxes, input_images, frame_id, image):
        images = torch.tensor(np.stack(input_images)).cuda()
        text = clip.tokenize(input_text).cuda()

        with torch.no_grad():
            image_features = self.model.encode_image(images).float()
            text_features = self.model.encode_text(text).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        ind = np.unravel_index(np.argmax(similarity), similarity.shape)
        print(ind)
        # cv2.imwrite("Image.jpg", image)

        crop = self.extract_bbox(image, input_bboxes[ind[1]])
        # cv2.imwrite("Crop.jpg", crop)
        print("Frame: {}".format(frame_id))

        # print(ind.shape)
        return input_bboxes[ind[1]]

    def extract_bbox(self, img, bbox):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        return img[y:y+h, x:x+w]

    def iuo(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]

        union_area = bbox1_area + bbox2_area - intersection_area
        if union_area == 0:
            return 0

        iou = intersection_area / union_area
        return iou
    
    # new approach

    def get_candidates_from_frame(self, frame_sec, video_id):
        
        FRAME_DIR = os.path.join(SEGMENTATION_DATA_PATH, video_id, "Frame.{}.0".format(frame_sec))

        MASK_METADATA_FILE = os.path.join(FRAME_DIR, "mask_data.txt")
        IMAGE_FILE = os.path.join(FRAME_DIR, "frame.jpg")    
        # load image + bboxes
        image = cv2.imread(IMAGE_FILE)
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(MASK_METADATA_FILE, 'r') as f:
            data = f.read()
            mask_data = data.strip().split("\n")
            mask_data = [json.loads(mask) for mask in mask_data]

        input_images = []
        input_bboxes = []

        for annotation in mask_data:
            input_image = self.extract_crop(image, annotation['bbox'])
            input_images.append(input_image)
            input_bboxes.append(annotation['bbox'])            

        return input_images, input_bboxes, frame_sec, image

    def extract_crop(self, image, bbox):
        crop = self.extract_bbox(image, bbox)
        result = self.preprocess(Image.fromarray(crop))
        return result

    def __get_candidate_1(self, image, ref_image, candidate_bboxes, ref_bboxes):
        candidates = []
        refs = []

        for bbox in candidate_bboxes:
            candidates.append(self.extract_crop(image.copy(), bbox))
        for bbox in ref_bboxes:
            refs.append(self.extract_crop(ref_image.copy(), bbox))

        if len(candidates) == 0 or len(refs) == 0:
            return None, None

        candidates = torch.tensor(np.stack(candidates)).cuda()
        refs = torch.tensor(np.stack(refs)).cuda()

        with torch.no_grad():
            candidates_features = self.model.encode_image(candidates).float()
            refs_features = self.model.encode_image(refs).float()

        candidates_features /= candidates_features.norm(dim=-1, keepdim=True)
        refs_features /= refs_features.norm(dim=-1, keepdim=True)

        similarity = refs_features.cpu().numpy() @ candidates_features.cpu().numpy().T

        similarity_sum = np.sum(similarity, axis=0)

        argmax_idx = np.unravel_index(np.argmax(similarity), similarity.shape)
        sum_idx = np.argmax(similarity_sum)

        print(argmax_idx, sum_idx, len(candidate_bboxes), similarity.shape, similarity_sum.shape)

        candidate_bbox_argmax = candidate_bboxes[argmax_idx[1]]
        candidate_bbox_sum = candidate_bboxes[sum_idx]

        return candidate_bbox_argmax, candidate_bbox_sum

    def get_candidate_ref(self, image, ref_image, segmentations_bboxes, sketches_bboxes, relevants_bboxes):
        candidate_bbox_sketch_argmax, candidate_bbox_sketch_sum = self.__get_candidate_1(image, ref_image, segmentations_bboxes, sketches_bboxes)
        candidate_bbox_relevant_argmax, candidate_bbox_relevant_sum = self.__get_candidate_1(image, ref_image, segmentations_bboxes, relevants_bboxes)
        return candidate_bbox_sketch_argmax, candidate_bbox_relevant_argmax, candidate_bbox_sketch_sum, candidate_bbox_relevant_sum
    
    def get_candidate_ref_text_single(self, image, ref_image, candidate_bboxes, ref_texts, ref_bboxes):
        candidates = []
        refs = []

        for bbox in candidate_bboxes:
            candidates.append(self.extract_crop(image.copy(), bbox))
        for bbox in ref_bboxes:
            refs.append(self.extract_crop(ref_image.copy(), bbox))

        if len(candidates) == 0 or len(refs) == 0 or len(ref_texts) == 0:
            return None, None

        candidates = torch.tensor(np.stack(candidates)).cuda()
        refs = torch.tensor(np.stack(refs)).cuda()
        ref_texts = clip.tokenize(ref_texts).cuda()

        with torch.no_grad():
            candidates_features = self.model.encode_image(candidates).float()
            refs_features = self.model.encode_image(refs).float()
            ref_texts_features = self.model.encode_text(ref_texts).float()

        candidates_features /= candidates_features.norm(dim=-1, keepdim=True)
        refs_features /= refs_features.norm(dim=-1, keepdim=True)
        ref_texts_features /= ref_texts_features.norm(dim=-1, keepdim=True)

        similarity_images = refs_features.cpu().numpy() @ candidates_features.cpu().numpy().T
        similarity_texts = ref_texts_features.cpu().numpy() @ candidates_features.cpu().numpy().T
        
        similarity = np.concatenate([similarity_images, similarity_texts], axis=0)
        
        similarity_sum = np.sum(similarity, axis=0)

        argmax_idx = np.unravel_index(np.argmax(similarity), similarity.shape)
        sum_idx = np.argmax(similarity_sum)

        print(argmax_idx, sum_idx, len(candidate_bboxes), similarity.shape, similarity_sum.shape)

        candidate_bbox_argmax = candidate_bboxes[argmax_idx[1]]
        candidate_bbox_sum = candidate_bboxes[sum_idx]

        return candidate_bbox_argmax, candidate_bbox_sum

    def get_candidate_ref_text(self, image, ref_image, segmentations_bboxes, vs_texts, sketches_bboxes, relevants_bboxes):
        candidate_bbox_sketch_argmax, candidate_bbox_sketch_sum = self.get_candidate_ref_text_single(image, ref_image, segmentations_bboxes, vs_texts, sketches_bboxes)
        candidate_bbox_relevant_argmax, candidate_bbox_relevant_sum = self.get_candidate_ref_text_single(image, ref_image, segmentations_bboxes, vs_texts, relevants_bboxes)
        return candidate_bbox_sketch_argmax, candidate_bbox_relevant_argmax, candidate_bbox_sketch_sum, candidate_bbox_relevant_sum
    
    def get_candidate_text(self, image, segmentations_bboxes, vs_texts):
        candidates = []

        for bbox in segmentations_bboxes:
            candidates.append(self.extract_crop(image.copy(), bbox))

        if len(candidates) == 0 or len(vs_texts) == 0:
            return None

        candidates = torch.tensor(np.stack(candidates)).cuda()
        vs_texts = clip.tokenize(vs_texts).cuda()

        with torch.no_grad():
            candidates_features = self.model.encode_image(candidates).float()
            vs_texts_features = self.model.encode_text(vs_texts).float()

        candidates_features /= candidates_features.norm(dim=-1, keepdim=True)
        vs_texts_features /= vs_texts_features.norm(dim=-1, keepdim=True)

        similarity = vs_texts_features.cpu().numpy() @ candidates_features.cpu().numpy().T
        similarity_sum = np.sum(similarity, axis=0)

        argmax_idx = np.unravel_index(np.argmax(similarity), similarity.shape)
        sum_idx = np.argmax(similarity_sum)

        print(argmax_idx, sum_idx, len(segmentations_bboxes), similarity.shape, similarity_sum.shape)

        candidate_bbox_argmax = segmentations_bboxes[argmax_idx[1]]
        candidate_bbox_sum = segmentations_bboxes[sum_idx]

        return candidate_bbox_argmax, candidate_bbox_sum

def main():    
    image_processor = ImageProcessor()
    frame_range = {'start': Timecode("00:03:00"), 'end': Timecode("00:03:10")}
    input_images, input_bboxes, frame_id, img = image_processor.extract_candidate_frame_masks(frame_range, "4LdIvyfzoGY")
    # image_processor.create_masks(img)
    image_processor.extract_related_crop("cables on a table", input_bboxes, input_images, frame_id, img)

if __name__=='__main__':
    main()
