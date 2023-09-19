import json

from copy import deepcopy

VIDEO_DATABASE = {
    "2": {
        "videoUrl": "https://www.youtube.com/watch?v=kdN41iYTg3U",
        "videoTitle": "At Home for the Holidays with Gordon Ramsay",
        "videoKnowledge": "procedural",
        "videoChannel": "visual",
        "expert": 9,
        "novice": 7,
    },
    "3": {
        "videoUrl": "https://www.youtube.com/watch?v=3_nLdcHBJY4",
        "videoTitle": "Language Learning Live Stream",
        "videoKnowledge": "procedural",
        "videoChannel": "verbal",
        "expert": 1,
        "novice": 10,
    },
    "4": {
        "videoUrl": "https://www.youtube.com/watch?v=OKQpOzEY_A4",
        "videoTitle": "Livestream: Getting Started with C++ (Episode 1)",
        "videoKnowledge": "procedural",
        "videoChannel": "visual+verbal",
        "expert": 2,
        "novice": 8,
    },
    "5": {
        "videoUrl": "https://www.youtube.com/watch?v=sz8Lo3NY1m0",
        "videoTitle": "Surgeon does Live Q&A | Hair Loss Awareness Month",
        "videoKnowledge": "declarative",
        "videoChannel": "verbal",
        "expert": 5,
        "novice": 6,
    },
    "6": {
        "videoUrl": "https://www.youtube.com/live/4LdIvyfzoGY?feature=share",
        "videoTitle": "Microsoft Surface Go - Classic LIVE Unboxing",
        "videoKnowledge": "declarative",
        "videoChannel": "visual",
        "expert": 3,
        "novice": 4,
    },
}

def __get_temporal_evaluation_margin(prediction, groun_truth, margin = 5):
    ground_truth_covered = [False for _ in groun_truth]
    prediction_covered = [False for _ in prediction]

    for i, prediction_segment in enumerate(prediction):
        prediction_left = prediction_segment[0]
        prediction_right = prediction_segment[1]
        for j, ground_truth_segment in enumerate(groun_truth):
            ground_truth_left = max(0, ground_truth_segment[0] - margin)
            ground_truth_right = ground_truth_segment[1] + margin
            intersection = max(0, min(prediction_right, ground_truth_right) - max(prediction_left, ground_truth_left))
            if intersection > 0:
                prediction_covered[i] = True
                ground_truth_covered[j] = True
                break
    
    precision = sum(prediction_covered) / max(1, len(prediction))
    recall = sum(ground_truth_covered) / max(1, len(groun_truth))

    f1_score = 0
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    return f1_score, precision, recall


def __get_temporal_evaluation_f1(prediction, ground_truth):
    # prediction: list [start, end], ground_truth: list [start, end]
    # start - end: float (seconds)
    # temporal
    # F1 score: 2 * intersection / (length_prediction + length_ground_truth)
    
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

    if (len(prediction) == 0 or len(ground_truth) == 0):
        return 0

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
    
    precision /= len(prediction)
    recall /= len(ground_truth)
    if precision + recall == 0:
        return 0
    return precision * recall / (precision + recall), precision, recall

def __get_single_edit_operation_evaluation(prediction, ground_truth):
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            if gt == prediction:
                return 1
    else:
        if ground_truth == prediction:
            return 1
    return 0

def get_dataset():
    #filename = "./gt_data/parsed_gt-v0.json"
    filename = "./gt_data/parsed_gt-v1.json"
    with open(filename, "r") as f:
        dataset = json.load(f)
        return dataset
    
def get_dataset_for_task(task_id=6):
    dataset = get_dataset()
    output = []
    for data in dataset:
        if data["task_id"] == task_id:
            output.append(data)
    return output

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
    input = {
        "projectId": "test",
        "projectMetadata": {
            "totalIntentCnt": 2,
            "duration": 1218.53678,
            "projectId": "tutorial",
            "height": 480,
            "fps": 25,
            "width": 854,
            "trackCnt": 1,
            "title": "test"
        },
        "curPlayPosition": 100,
        "edits": [],
        "segmentOfInterest": {
            "start": 0,
            "finish": 1218.53678
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
        "editOperations": dataset[index]["edit_text"],
        "parameters": {},
        "edits": dataset[index]["temporal"],
        "edits_spatial": dataset[index]["spatial"],
        "relevant_text": {
            "temporal": dataset[index]["temporal_text"],
            "spatial": dataset[index]["spatial_text"],
            "edit": dataset[index]["edit_text"],
        }
    }
    return input, ground_truth

def get_data_point(dataset, index):
    if (index >= len(dataset) or index < 0):
        return None
    input = {
        "text": dataset[index]["description"],
        "sketch": dataset[index]["sketch"],
        "sketch_timestamp": dataset[index]["sketch_timestamp"],
    }

    ground_truth = {
        "editOperations": dataset[index]["edit_text"],
        "parameters": {},
        "edits": dataset[index]["temporal"],
        "edits_spatial": dataset[index]["spatial"],
        "relevant_text": {
            "temporal": dataset[index]["temporal_text"],
            "spatial": dataset[index]["spatial_text"],
            "edit": dataset[index]["edit_text"],
        }
    }
    return input, ground_truth


def get_temporal_text_evaluation():
    # temporal_text
    pass

def get_spatial_text_evaluation():
    # spatial_text
    pass

def get_temporal_evaluation(prediction, ground_truth):
    # prediction: list [start, end], ground_truth: list [start, end]
    # start - end: float (seconds)
    # temporal
    #f1 = __get_temporal_evaluation_f1(prediction, ground_truth)
    f1_0, precision_0, recall_0 = __get_temporal_evaluation_margin(prediction, ground_truth, margin=0)
    #traditional = __get_temporal_evaluation_traditional(prediction, ground_truth)
    f1_10, precision_10, recall_10 = __get_temporal_evaluation_margin(prediction, ground_truth, margin=10)
    return (
        f1_0, precision_0, recall_0,
    ), (
        f1_10, precision_10, recall_10,
    )


def get_spatial_evaluation():
    # spatial
    pass

def get_edit_operation_evaluation(prediction, ground_truth):
    if isinstance(prediction, list):
        if len(prediction) == 0:
            return (1 if len(ground_truth) == 0 else 0)
        total = 0
        for single_prediction in prediction:
            result = __get_single_edit_operation_evaluation(single_prediction, ground_truth)
            total += result
        return total / len(prediction)
    else:
        return __get_single_edit_operation_evaluation(prediction, ground_truth)
    # edit_text

def get_edit_params_evaluation():
    # extra params
    pass

def main():
    dataset = get_dataset_for_task(6)
    for i in range(len(dataset)):
        info = get_data_point_info(dataset, i)
        data_point = get_data_point(dataset, i)
        print(data_point)
        #print("Data: ", json.dumps(data_point, intent=2))


if __name__ == "__main__":
    main()