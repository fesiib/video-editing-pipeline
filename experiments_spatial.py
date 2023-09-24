import evaluation.evaluate_helpers as evaluate_helpers


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
from backend.helpers import Timecode

SEGMENTATION_DATA_PATH = "segmentation-data"

class ImageProcessor:
    model = None
    process = None
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32")

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

    # no vs_texts: relevant_rectangles/sketches x segmentations -> sum_max_cos_sim or arg_max_cos_sim -> candidate
    def get_candidate_ref(self, image, ref_image, segmentations_bboxes, sketches_bboxes, relevants_bboxes):
        candidate_bbox_sketch_argmax, candidate_bbox_sketch_sum = self.__get_candidate_1(image, ref_image, segmentations_bboxes, sketches_bboxes)
        candidate_bbox_relevant_argmax, candidate_bbox_relevant_sum = self.__get_candidate_1(image, ref_image, segmentations_bboxes, relevants_bboxes)
        return candidate_bbox_sketch_argmax, candidate_bbox_relevant_argmax, candidate_bbox_sketch_sum, candidate_bbox_relevant_sum
    
    def __get_candidate_2(self, image, ref_image, candidate_bboxes, ref_texts, ref_bboxes):
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
        candidate_bbox_sketch_argmax, candidate_bbox_sketch_sum = self.__get_candidate_2(image, ref_image, segmentations_bboxes, vs_texts, sketches_bboxes)
        candidate_bbox_relevant_argmax, candidate_bbox_relevant_sum = self.__get_candidate_2(image, ref_image, segmentations_bboxes, vs_texts, relevants_bboxes)
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

### delete all files in "./images"
import os

for file in os.listdir("./images"):
    os.remove(os.path.join("./images", file))

processor = ImageProcessor()

iou_threshold = 0.3

skip_intents = [
    [1, 2],
    [1, 3],
    [2, 7],
]

for task_id in range(2, 7):
    dataset = evaluate_helpers.get_dataset_for_task(task_id)
    for index in range(len(dataset)):
        if [task_id, index] in skip_intents:
            continue
        filename_prefix = "task_{}_index_{}".format(task_id, index)

        request, ground_truth = evaluate_helpers.get_data_point(dataset, index)

        video_shape = request['video_shape']
        video_id = request['videoId']
        sketches = request['sketch']
        sketch_frame_sec = int(request['sketch_timestamp'])
        if (sketch_frame_sec == -1):
            continue
        input_images, input_bboxes, frame_sec, image = processor.get_candidates_from_frame(sketch_frame_sec, video_id)
        scaled_sketches = []
        for bbox in sketches:
            scaled_bbox = [
                int(bbox['x'] / video_shape[1] * image.shape[1]),
                int(bbox['y'] / video_shape[0] * image.shape[0]),
                int(bbox['width'] / video_shape[1] * image.shape[1]),
                int(bbox['height'] / video_shape[0] * image.shape[0]),
            ]
            scaled_sketches.append(scaled_bbox)

        sketch_image = image.copy()
        for bbox in scaled_sketches:
            cv2.rectangle(sketch_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        cv2.imwrite("./images/{}_!sketch.jpg".format(filename_prefix), sketch_image)
        
        relevants_image = image.copy()
        relevant_bboxes = []
        for i, input_image in enumerate(input_images):
            count_intersect = 0
            current_image = image.copy()
            for j in range(len(scaled_sketches)):
                iou = processor.iuo(input_bboxes[i], scaled_sketches[j])
                if iou > iou_threshold:
                    cv2.rectangle(current_image, (scaled_sketches[j][0], scaled_sketches[j][1]), (scaled_sketches[j][0] + scaled_sketches[j][2], scaled_sketches[j][1] + scaled_sketches[j][3]), (0, 0, 255), 2)
                    count_intersect += 1
            if count_intersect > 0:
                cv2.rectangle(current_image, (input_bboxes[i][0], input_bboxes[i][1]), (input_bboxes[i][0] + input_bboxes[i][2], input_bboxes[i][1] + input_bboxes[i][3]), (0, 255, 0), 2)
                cv2.rectangle(relevants_image, (input_bboxes[i][0], input_bboxes[i][1]), (input_bboxes[i][0] + input_bboxes[i][2], input_bboxes[i][1] + input_bboxes[i][3]), (0, 255, 0), 2)
                cv2.imwrite("./images/{}_relevant_{}.jpg".format(filename_prefix, i), current_image)
                relevant_bboxes.append(input_bboxes[i])
        cv2.imwrite("./images/{}_!relevant.jpg".format(filename_prefix), relevants_image)

        ref_image = image.copy()

        for edit_idx, (frame_range, gt_bbox) in enumerate(zip(ground_truth['edits'], ground_truth['edits_spatial'])):
            edit_filename_prefix = "{}_edit_{}".format(filename_prefix, edit_idx)
            if gt_bbox is None:
                continue

            frame_sec = int(frame_range[0] + frame_range[1]) // 2
            # get_candidates
            input_images, input_bboxes, frame_sec, image = processor.get_candidates_from_frame(frame_sec, video_id)

            # save ground_truth
            ground_truth_image = image.copy()
            scaled_gt_bbox = [
                int(gt_bbox['x'] / video_shape[1] * image.shape[1]),
                int(gt_bbox['y'] / video_shape[0] * image.shape[0]),
                int(gt_bbox['width'] / video_shape[1] * image.shape[1]),
                int(gt_bbox['height'] / video_shape[0] * image.shape[0]),
            ]         
            cv2.rectangle(ground_truth_image, (scaled_gt_bbox[0], scaled_gt_bbox[1]), (scaled_gt_bbox[0] + scaled_gt_bbox[2], scaled_gt_bbox[1] + scaled_gt_bbox[3]), (0, 255, 0), 2)
            cv2.imwrite("./images/{}_!ground_truth.jpg".format(edit_filename_prefix), ground_truth_image)

            ### no vs-texts: relevant_rectangles x segmentations or sketches vs segmentations
            # (
            #     candidate_bbox_sketch_argmax,
            #     candidate_bbox_relevant_argmax,
            #     candidate_bbox_sketch_sum,
            #     candidate_bbox_relevant_sum
            # ) = processor.get_candidate_ref(image.copy(), input_bboxes, scaled_sketches, relevant_bboxes)

            # (
            #     candidate_bbox_sketch_argmax,
            #     candidate_bbox_relevant_argmax,
            #     candidate_bbox_sketch_sum,
            #     candidate_bbox_relevant_sum
            # ) = processor.get_candidate_ref_text(image.copy(), ref_image.copy(), input_bboxes, [request["text"]], scaled_sketches, relevant_bboxes)

            (
                candidate_bbox_sketch_argmax,
                candidate_bbox_sketch_sum,
            ) = processor.get_candidate_text(image.copy(), input_bboxes, [request["text"]])
            candidate_bbox_relevant_argmax = None
            candidate_bbox_relevant_sum = None

            # save candidate_bbox_sketch_argmax
            if candidate_bbox_sketch_argmax is not None:
                candidate_bbox_sketch_argmax_image = image.copy()
                cv2.rectangle(candidate_bbox_sketch_argmax_image, (candidate_bbox_sketch_argmax[0], candidate_bbox_sketch_argmax[1]), (candidate_bbox_sketch_argmax[0] + candidate_bbox_sketch_argmax[2], candidate_bbox_sketch_argmax[1] + candidate_bbox_sketch_argmax[3]), (0, 255, 0), 2)
                cv2.rectangle(candidate_bbox_sketch_argmax_image, (scaled_gt_bbox[0], scaled_gt_bbox[1]), (scaled_gt_bbox[0] + scaled_gt_bbox[2], scaled_gt_bbox[1] + scaled_gt_bbox[3]), (0, 0, 255), 2)
                cv2.imwrite("./images/{}_sketch_argmax.jpg".format(edit_filename_prefix), candidate_bbox_sketch_argmax_image)

            # save candidate_bbox_relevant_argmax
            if candidate_bbox_relevant_argmax is not None:
                candidate_bbox_relevant_argmax_image = image.copy()
                cv2.rectangle(candidate_bbox_relevant_argmax_image, (candidate_bbox_relevant_argmax[0], candidate_bbox_relevant_argmax[1]), (candidate_bbox_relevant_argmax[0] + candidate_bbox_relevant_argmax[2], candidate_bbox_relevant_argmax[1] + candidate_bbox_relevant_argmax[3]), (0, 255, 0), 2)
                cv2.rectangle(candidate_bbox_relevant_argmax_image, (scaled_gt_bbox[0], scaled_gt_bbox[1]), (scaled_gt_bbox[0] + scaled_gt_bbox[2], scaled_gt_bbox[1] + scaled_gt_bbox[3]), (0, 0, 255), 2)
                cv2.imwrite("./images/{}_relevant_argmax.jpg".format(edit_filename_prefix), candidate_bbox_relevant_argmax_image)

            # save candidate_bbox_sketch_sum
            if candidate_bbox_sketch_sum is not None:
                candidate_bbox_sketch_sum_image = image.copy()
                cv2.rectangle(candidate_bbox_sketch_sum_image, (candidate_bbox_sketch_sum[0], candidate_bbox_sketch_sum[1]), (candidate_bbox_sketch_sum[0] + candidate_bbox_sketch_sum[2], candidate_bbox_sketch_sum[1] + candidate_bbox_sketch_sum[3]), (0, 255, 0), 2)
                cv2.rectangle(candidate_bbox_sketch_sum_image, (scaled_gt_bbox[0], scaled_gt_bbox[1]), (scaled_gt_bbox[0] + scaled_gt_bbox[2], scaled_gt_bbox[1] + scaled_gt_bbox[3]), (0, 0, 255), 2)
                cv2.imwrite("./images/{}_sketch_sum.jpg".format(edit_filename_prefix), candidate_bbox_sketch_sum_image)

            # save candidate_bbox_relevant_sum
            if candidate_bbox_relevant_sum is not None:
                candidate_bbox_relevant_sum_image = image.copy()
                cv2.rectangle(candidate_bbox_relevant_sum_image, (candidate_bbox_relevant_sum[0], candidate_bbox_relevant_sum[1]), (candidate_bbox_relevant_sum[0] + candidate_bbox_relevant_sum[2], candidate_bbox_relevant_sum[1] + candidate_bbox_relevant_sum[3]), (0, 255, 0), 2)
                cv2.rectangle(candidate_bbox_relevant_sum_image, (scaled_gt_bbox[0], scaled_gt_bbox[1]), (scaled_gt_bbox[0] + scaled_gt_bbox[2], scaled_gt_bbox[1] + scaled_gt_bbox[3]), (0, 0, 255), 2)
                cv2.imwrite("./images/{}_relevant_sum.jpg".format(edit_filename_prefix), candidate_bbox_relevant_sum_image)

"""
Algorithm:

Segment [start, finish]

`visual-dependent` texts -> `vs_texts`

Sketch available:
    Sketch in Segment -> candidate = sketch

    Sketch not in Segment -> Find rectangles that intersect with sketch from the frame that was sketched in (iou > 0.3) -> call this `relevant_rectangles`

    Some vs_texts:
        Method 1: (vs_texts, relevant_rectangles) x segmentations -> `candidate_rectangle` -> sum_max_cos_sim or arg_max_cos_sim -> candidate
        Method 2: (vs_texts, sketch) x segmentations -> `candidate_rectangle` -> sum_max_cos_sim or arg_max_cos_sim -> candidate
    
    No vs_texts:
        relevant_rectangles x segmentations -> sum_max_cos_sim or arg_max_cos_sim -> candidate

No Sketch available:
    Some vs_texts:
        vs_texts x segmentations -> `candidate_rectangle` -> sum_max_cos_sim or arg_max_cos_sim -> candidate
    No vs_texts:
        candidate = full frame

"""