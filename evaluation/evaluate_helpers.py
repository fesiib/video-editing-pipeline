import json
import math

from copy import deepcopy

from evaluation.sentence_embedder import get_cosine_similarity_scores


VIDEO_DATABASE = {
    "2": {
        "videoUrl": "https://www.youtube.com/watch?v=kdN41iYTg3U",
        "videoId": "kdN41iYTg3U",
        "videoTitle": "At Home for the Holidays with Gordon Ramsay",
        "videoKnowledge": "procedural",
        "videoChannel": "visual",
        "expert": 9,
        "novice": 7,
        "duration": 3236.954558,
    },
    "3": {
        "videoUrl": "https://www.youtube.com/watch?v=3_nLdcHBJY4",
        "videoId": "3_nLdcHBJY4",
        "videoTitle": "Language Learning Live Stream",
        "videoKnowledge": "procedural",
        "videoChannel": "verbal",
        "expert": 1,
        "novice": 10,
        "duration": 3774.403628,
    },
    "4": {
        "videoUrl": "https://www.youtube.com/watch?v=OKQpOzEY_A4",
        "videoId": "OKQpOzEY_A4",
        "videoTitle": "Livestream: Getting Started with C++ (Episode 1)",
        "videoKnowledge": "procedural",
        "videoChannel": "visual+verbal",
        "expert": 2,
        "novice": 8,
        "duration": 3501.151202,
    },
    "5": {
        "videoUrl": "https://www.youtube.com/watch?v=sz8Lo3NY1m0",
        "videoId": "sz8Lo3NY1m0",
        "videoTitle": "Surgeon does Live Q&A | Hair Loss Awareness Month",
        "videoKnowledge": "declarative",
        "videoChannel": "verbal",
        "expert": 5,
        "novice": 6,
        "duration": 2125.740408,
    },
    "6": {
        "videoUrl": "https://www.youtube.com/live/4LdIvyfzoGY?feature=share",
        "videoId": "4LdIvyfzoGY",
        "videoTitle": "Microsoft Surface Go - Classic LIVE Unboxing",
        "videoKnowledge": "declarative",
        "videoChannel": "visual",
        "expert": 3,
        "novice": 4,
        "duration": 1218.53678,
    },
}

def __get_temporal_evaluation_margin(prediction, ground_truth, margin = 5):
    if len(ground_truth) == 0:
        return -1, -1, -1
    ground_truth_covered = [False for _ in ground_truth]
    prediction_covered = [False for _ in prediction]

    for i, prediction_segment in enumerate(prediction):
        prediction_left = prediction_segment[0]
        prediction_right = prediction_segment[1]
        for j, ground_truth_segment in enumerate(ground_truth):
            ground_truth_left = max(0, ground_truth_segment[0] - margin)
            ground_truth_right = ground_truth_segment[1] + margin
            intersection = max(0, min(prediction_right, ground_truth_right) - max(prediction_left, ground_truth_left))
            if intersection > 0:
                prediction_covered[i] = True
                ground_truth_covered[j] = True
                break
    precision = 0
    recall = 0
    f1_score = 0

    if len(prediction) > 0:
        precision = sum(prediction_covered) / len(prediction)
    if len(ground_truth) > 0:
        recall = sum(ground_truth_covered) / len(ground_truth)

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    return f1_score, precision, recall


def __get_temporal_evaluation_f1(prediction, ground_truth):
    # prediction: list [start, end], ground_truth: list [start, end]
    # start - end: float (seconds)
    # temporal
    # F1 score: 2 * intersection / (length_prediction + length_ground_truth)
    
    if len(ground_truth) == 0:
        return -1, -1, -1

    total_intersection = 0
    total_length_prediction = 0
    total_length_ground_truth = 0

    for prediction_segment in prediction:
        total_length_prediction += prediction_segment[1] - prediction_segment[0]
    
    for ground_truth_segment in ground_truth:
        total_length_ground_truth += ground_truth_segment[1] - ground_truth_segment[0]
    
    for prediction_segment in prediction:
        for ground_truth_segment in ground_truth:
            intersection = max(0, min(prediction_segment[1], ground_truth_segment[1]) - max(prediction_segment[0], ground_truth_segment[0]))
            total_intersection += intersection

    if (total_length_prediction + total_length_ground_truth == 0):
        return 0

    return ((total_intersection * 2) / (total_length_prediction + total_length_ground_truth),
        total_intersection / total_length_prediction, total_intersection / total_length_ground_truth)

def __get_temporal_evaluation_traditional(prediction, ground_truth):
    # prediction: list [start, end], ground_truth: list [start, end]
    # start - end: float (seconds)
    # temporal
    # Traditional: precision * recall / (precision + recall)

    if len(ground_truth) == 0:
        return -1, -1, -1

    precision = 0
    recall = 0

    for prediction_segment in prediction:
        for ground_truth_segment in ground_truth:
            intersection = max(0, min(prediction_segment[1], ground_truth_segment[1]) - max(prediction_segment[0], ground_truth_segment[0]))
            if prediction_segment[1] - prediction_segment[0] == 0:
                precision += 1
            else:
                precision += intersection / (prediction_segment[1] - prediction_segment[0])
            if ground_truth_segment[1] - ground_truth_segment[0] == 0:
                recall += 1
            else:
                recall += intersection / (ground_truth_segment[1] - ground_truth_segment[0])
    
    if len(prediction) > 0:
        precision /= len(prediction)
    if len(ground_truth) > 0:
        recall /= len(ground_truth)
    f1_score = 0
    if precision + recall > 0:
        f1_score = precision * recall / (precision + recall)

    return f1_score, precision, recall

def __get_single_edit_operation_evaluation(prediction, ground_truth):
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            if gt == prediction:
                return 1
    else:
        if ground_truth == prediction:
            return 1
    return 0

def __get_spatial_evaluation_miou(pairs):
    mean = 0
    all_iou = []
    for (rect1s, rect2s) in pairs:
        print("before", rect1s, rect2s)
        if isinstance(rect1s, list) == False:
            rect1s = [rect1s]
        if isinstance(rect2s, list) == False:
            rect2s = [rect2s]
        print("after", rect1s, rect2s)
        max_iou = -1
        for rect1 in rect1s:
            for rect2 in rect2s:
                if rect1 == None or rect2 == None:
                    continue
                x1 = max(rect1["x"], rect2["x"])
                y1 = max(rect1["y"], rect2["y"])
                x2 = min(rect1["x"] + rect1["width"], rect2["x"] + rect2["width"])
                y2 = min(rect1["y"] + rect1["height"], rect2["y"] + rect2["height"])
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                union = rect1["width"] * rect1["height"] + rect2["width"] * rect2["height"] - intersection
                iou = intersection / union
                max_iou = max(max_iou, iou)
        if max_iou == -1:
            continue
        mean += max_iou
        all_iou.append(max_iou)
            
    if len(all_iou) == 0:
        return 0, []

    mean /= len(all_iou)
    return mean, all_iou

def __get_spatial_evaluation_margin(prediction, ground_truth, temporal_prediction, temporal_ground_truth, margin = 5):
    pairs = []
    for i, prediction_segment in enumerate(temporal_prediction):
        prediction_left = prediction_segment[0]
        prediction_right = prediction_segment[1]
        for j, ground_truth_segment in enumerate(temporal_ground_truth):
            ground_truth_left = max(0, ground_truth_segment[0] - margin)
            ground_truth_right = ground_truth_segment[1] + margin
            intersection = max(0, min(prediction_right, ground_truth_right) - max(prediction_left, ground_truth_left))
            if intersection > 0:
                pairs.append((prediction[i], ground_truth[j]))
            
    miou, all_iou = __get_spatial_evaluation_miou(pairs)
    return miou, all_iou

def get_dataset():
    # filename = "./gt_data/parsed_gt-v0.json"
    # filename = "./gt_data/parsed_gt-v1.json"
    # filename = "./gt_data/parsed_gt-v2.json"
    filename = "./gt_data/parsed_gt-v3.json"
    with open(filename, "r") as f:
        dataset = json.load(f)
        ret_dataset = []
        for data in dataset:
            ret_dataset.append(data)
        return ret_dataset
    
def get_dataset_for_task(task_id=6):
    dataset = get_dataset()
    ret_dataset = []
    for data in dataset:
        if data["task_id"] == task_id: 
            ret_dataset.append(data)
    return ret_dataset

def get_data_point_info(dataset, index):
    if (index >= len(dataset) or index < 0):
        return None
    
    videoInfo = deepcopy(VIDEO_DATABASE[str(dataset[index]["task_id"])]),
    videoInfo = videoInfo[0]
    info = {
        "participant_id": dataset[index]["participant_id"],
        "task_id": dataset[index]["task_id"],
        "intent_id": dataset[index]["intent_id"],
        "description": dataset[index]["description"],
        "sketch": dataset[index]["sketch"],
        "sketch_timestamp": dataset[index]["sketch_timestamp"],
        "videoUrl": videoInfo["videoUrl"],
        "videoId": videoInfo["videoId"],
        "videoTitle": videoInfo["videoTitle"],
        "videoKnowledge": videoInfo["videoKnowledge"],
        "videoChannel": videoInfo["videoChannel"],
        "isExpert": videoInfo["expert"] == dataset[index]["participant_id"],
        "isNovice": videoInfo["novice"] == dataset[index]["participant_id"],
    }
    return info

def get_data_point_as_request(dataset, index):
    if (index >= len(dataset) or index < 0):
        return None
    
    duration = VIDEO_DATABASE[str(dataset[index]["task_id"])]["duration"]
    input = {
        "videoId": VIDEO_DATABASE[str(dataset[index]["task_id"])]["videoId"],
        "projectMetadata": {
            "totalIntentCnt": 2,
            "duration": duration,
            "projectId": str(dataset[index]["task_id"]),
            "height": 480,
            "fps": 25,
            "width": 854,
            "trackCnt": 1,
            "title": "test"
        },
        "curPlayPosition": 0,
        "edits": [],
        "segmentOfInterest": {
            "start": 0,
            "finish": duration
        },
        "skippedSegments": [],
        "requestParameters": {
            "processingMode": "from-scratch",
            "hasText": False,
            "hasSketch": True,
            "text": dataset[index]["description"],
            "sketchRectangles": dataset[index]["sketch"],
            "sketchFrameTimestamp": dataset[index]["sketch_timestamp"],
            "editOperation": ""
        },
    }
    ground_truth = {
        "editOperations": dataset[index]["edit"],
        "parameters": {},
        "edits_temporal": dataset[index]["temporal"],
        "edits_spatial": dataset[index]["spatial"],
        "relevant_text": {
            "temporal": dataset[index]["temporal_text"],
            "spatial": dataset[index]["spatial_text"],
            "edit": dataset[index]["edit_text"],
            "parameters": dataset[index]["params_text"],
        }
    }
    return input, ground_truth

def get_data_point_parsing(dataset, index):
    if (index >= len(dataset) or index < 0):
        return None
    input = {
        "videoId": VIDEO_DATABASE[str(dataset[index]["task_id"])]["videoId"],
        "text": dataset[index]["description"],
        "sketch": dataset[index]["sketch"],
        "sketch_timestamp": dataset[index]["sketch_timestamp"],
        "video_shape": [480, 854],
        "video_duration": VIDEO_DATABASE[str(dataset[index]["task_id"])]["duration"],
    }

    ground_truth = {
        "editOperations": dataset[index]["edit"],
        "temporal": dataset[index]["temporal_text"],
        "spatial": dataset[index]["spatial_text"],
        "edit": dataset[index]["edit_text"],
        "parameters": dataset[index]["params_text"],
    }
    return input, ground_truth

def get_data_point(dataset, index):
    if (index >= len(dataset) or index < 0):
        return None
    input = {
        "videoId": VIDEO_DATABASE[str(dataset[index]["task_id"])]["videoId"],
        "text": dataset[index]["description"],
        "sketch": dataset[index]["sketch"],
        "sketch_timestamp": dataset[index]["sketch_timestamp"],
        "video_shape": [480, 854],
        "video_duration": VIDEO_DATABASE[str(dataset[index]["task_id"])]["duration"],
    }

    ground_truth = {
        "editOperations": dataset[index]["edit"],
        "parameters": {},
        "edits_temporal": dataset[index]["temporal"],
        "edits_spatial": dataset[index]["spatial"],
        "relevant_text": {
            "temporal": dataset[index]["temporal_text"],
            "spatial": dataset[index]["spatial_text"],
            "edit": dataset[index]["edit_text"],
            "parameters": dataset[index]["params_text"],
        }
    }
    return input, ground_truth

def get_temporal_evaluation(prediction, ground_truth):
    # prediction: list [start, end], ground_truth: list [start, end]
    # start - end: float (seconds)
    # temporal
    #f1 = __get_temporal_evaluation_f1(prediction, ground_truth)
    f1_0, precision_0, recall_0 = __get_temporal_evaluation_margin(prediction, ground_truth, margin=0)
    f1_5, precision_5, recall_5 = __get_temporal_evaluation_margin(prediction, ground_truth, margin=5)
    #traditional = __get_temporal_evaluation_traditional(prediction, ground_truth)
    f1_10, precision_10, recall_10 = __get_temporal_evaluation_margin(prediction, ground_truth, margin=10)
    return (
        f1_0, precision_0, recall_0,
    ), (
        f1_5, precision_5, recall_5,
    ), (
        f1_10, precision_10, recall_10,
    )


def get_spatial_evaluation(prediction, ground_truth, temporal_prediction, temporal_ground_truth, iou_threshold = 0.5):
    
    miou_0, all_iou_0 = __get_spatial_evaluation_margin(prediction, ground_truth, temporal_prediction, temporal_ground_truth, margin=0)
    miou_5, all_iou_5 = __get_spatial_evaluation_margin(prediction, ground_truth, temporal_prediction, temporal_ground_truth, margin=5)
    miou_10, all_iou_10 = __get_spatial_evaluation_margin(prediction, ground_truth, temporal_prediction, temporal_ground_truth, margin=10)

    thresholded_0 = 0
    thresholded_5 = 0
    thresholded_10 = 0

    for iou in all_iou_0:
        if iou >= iou_threshold:
            thresholded_0 += 1
    for iou in all_iou_5:
        if iou >= iou_threshold:
            thresholded_5 += 1
    for iou in all_iou_10:
        if iou >= iou_threshold:
            thresholded_10 += 1
    
    if len(all_iou_0) > 0:
        thresholded_0 /= len(all_iou_0)
    if len(all_iou_5) > 0:
        thresholded_5 /= len(all_iou_5)
    if len(all_iou_10) > 0:
        thresholded_10 /= len(all_iou_10)

    return (miou_0, thresholded_0), (miou_5, thresholded_5), (miou_10, thresholded_10)

def get_spatial_evaluation_pairs(predictions, ground_truth, iou_threshold = 0.5):
    if len(predictions) != len(ground_truth):
        print("ERROR lengths: ", len(predictions), len(ground_truth))
        return 0, 0
    pairs = []
    for i in range(len(predictions)):
        pairs.append((predictions[i], ground_truth[i]))
    miou, all_iou = __get_spatial_evaluation_miou(pairs)

    thresholded = 0
    for iou in all_iou:
        if iou >= iou_threshold:
            thresholded += 1

    if len(all_iou) > 0:
        thresholded /= len(all_iou)
    return miou, thresholded

def get_edit_operation_evaluation(prediction, ground_truth):
    total = 0
    precision = 0
    recall = 0
    f1 = 0
    if len(ground_truth) == 0:
            return -1, -1, -1
    if isinstance(prediction, list):
        prediction = list(set(prediction))
        for single_prediction in prediction:
            result = __get_single_edit_operation_evaluation(single_prediction, ground_truth)
            total += result
    else:
        total = __get_single_edit_operation_evaluation(prediction, ground_truth)

    if len(prediction) > 0:
        precision = total / len(prediction)
    if len(ground_truth) > 0:
        recall = total / len(ground_truth)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def get_edit_params_evaluation():
    # extra params
    pass

def get_references_evaluation(prediction, ground_truth):
    if len(ground_truth) == 0:
        return (-1, -1, -1,
            -1, -1, -1,
            [], [], [], [],
        )

    # agg_score, cosine_scores_expanded, top_10_pairs_expanded, cosine_scores, top_10_pairs
    cosine_scores, top_10_pairs = get_cosine_similarity_scores(
        prediction,
        ground_truth,
    )
    # Expanded Ground Truth
    expanded_ground_truth = []
    for item in ground_truth:
        expanded_ground_truth.extend(item.split(", "))
    cosine_scores_expanded, top_10_pairs_expanded = get_cosine_similarity_scores(
        prediction,
        expanded_ground_truth,
    )
    precision_expanded = 0
    recall_expanded = 0
    f1_expanded = 0

    precision = 0
    recall = 0
    f1 = 0

    for single_cosine_scores in cosine_scores_expanded:
        if len(single_cosine_scores) > 0:
            precision_expanded += max([item for item in single_cosine_scores])
    if len(cosine_scores_expanded) > 0:
        precision_expanded = precision_expanded / len(cosine_scores_expanded)

    cosine_scores_expanded_t = [list(l) for l in zip(*cosine_scores_expanded)]
    for single_cosine_scores in cosine_scores_expanded_t:
        if len(single_cosine_scores) > 0:
            recall_expanded += max([item for item in single_cosine_scores])
    if len(cosine_scores_expanded_t) > 0:
        recall_expanded = recall_expanded / len(cosine_scores_expanded_t)
    if precision_expanded + recall_expanded > 0:
        f1_expanded = 2 * precision_expanded * recall_expanded / (precision_expanded + recall_expanded)
    
    for single_cosine_scores in cosine_scores:
        if len(single_cosine_scores) > 0:
            precision += max([item for item in single_cosine_scores])
    if len(cosine_scores) > 0:
        precision = precision / len(cosine_scores)

    cosine_scores_t = [list(l) for l in zip(*cosine_scores)]
    for single_cosine_scores in cosine_scores_t:
        if len(single_cosine_scores) > 0:
            recall += max([item for item in single_cosine_scores])
    if len(cosine_scores_t) > 0:
        recall = recall / len(cosine_scores_t)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return (
        f1,
        precision,
        recall,
        f1_expanded,
        precision_expanded,
        recall_expanded,
        cosine_scores_expanded,
        top_10_pairs_expanded,
        cosine_scores,
        top_10_pairs,
    )

def round_number(number):
    # return with trailing 3 digits
    number = math.floor(number * 1000) / 1000
    # add trailing zeros
    number = str(number) + "0" * (3 - len(str(number).split(".")[1]))
    return number

def sum(arr):
    sum = 0
    for i in arr:
        sum += i
    return sum

def avg_std(arr):
    if len(arr) == 0:
        return 0, 0

    arr = [x for x in arr if x >= 0]

    avg = sum(arr) / len(arr)
    std = 0
    for i in arr:
        std += (i - avg) ** 2
    std = (std / len(arr)) ** 0.5
    return avg, std

def append_dict(main_dict, new_dict):
    for key in new_dict:
        if key not in main_dict:
            if isinstance(new_dict[key], dict):
                main_dict[key] = {}
                main_dict[key] = append_dict(main_dict[key], new_dict[key])
            elif isinstance(new_dict[key], list):
                main_dict[key] = []
                main_dict[key].extend(new_dict[key])
            else:
                main_dict[key] = new_dict[key]
            continue

        if isinstance(new_dict[key], dict):
            main_dict[key] = append_dict(main_dict[key], new_dict[key])
        elif isinstance(new_dict[key], list):
            main_dict[key].extend(new_dict[key])
        else:
            main_dict[key] += new_dict[key]
    return main_dict

def skip_dict(d, is_skipped):
    if isinstance(d, dict):
        for key in d:
            d[key] = skip_dict(d[key], is_skipped)
    elif isinstance(d, list):
        new_list = []
        for i in range(len(d)):
            if is_skipped[i] == False:
                continue
            new_list.append(d[i])
        return new_list
    return d
            


def main():
    dataset = get_dataset_for_task(6)
    for i in range(len(dataset)):
        info = get_data_point_info(dataset, i)
        data_point = get_data_point(dataset, i)
        print(data_point)
        #print("Data: ", json.dumps(data_point, intent=2))


if __name__ == "__main__":
    main()