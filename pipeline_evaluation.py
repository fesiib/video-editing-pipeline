import json
import ast

from evaluation.evaluator import *
from evaluation.evaluate_helpers import *

PIPELINE_RESULTS_FOLDER = "pipeline-results"

PARSING_RESULTS_PREFIX = "parsing"
SPATIAL_ONLY_RESULTS_PREFIX = "spatial_only"
FULL_RESULTS_PREFIX = "full"
LOGS_PREFIX = "log"
SUMMARY_FILENAME = "combined_full_results"

def evaluate_all_tasks_parsing():
    task_ids = [2, 3, 4, 5, 6]
    for task_id in task_ids:
        dataset = get_dataset_for_task(task_id)
        result = []
        for index in range(len(dataset)):
            input, ground_truths = get_data_point_parsing(dataset, index)
            response = get_references_langchain(input)

            comparison = {}

            for key in response:
                prediction = response[key]
                ground_truth = ground_truths[key]
                if key == "editOperations":
                    f1, precision, recall = get_edit_operation_evaluation(
                        prediction,
                        ground_truth,
                    )
                    comparison[key] = {
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                    }
                    continue
                cosine_scores, top_10_pairs = get_cosine_similarity_scores(
                    prediction,
                    ground_truth,
                )
                comparison[key] = {
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "cosine_scores": cosine_scores,
                    "top_10_pairs": top_10_pairs,
                }
            result.append(comparison)
        with open(f"{PIPELINE_RESULTS_FOLDER}/{PARSING_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def evaluate_all_tasks_full():
    task_ids = [6]
    result = run_evaluation(
        task_ids = task_ids,
        data_point_getter = get_data_point_as_request,
        pipeline_runner = get_response_langchain,
        # indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[10] - video #[4] - position
        indexes=[]
    )

    with open(f"{PIPELINE_RESULTS_FOLDER}/{FULL_RESULTS_PREFIX}_{str(task_ids)}.json", "w") as f:
        json.dump(result, f, indent=2)

    if (len(result["dataset"]) == 0):
        return
    
    print("Summary:")
    print("--------------------")

    print("Temporal F1 Margin=0: ", round_number(result["temporal_f1_0"]), " --> ", [round_number(x) for x in result["all_temporal_f1_0"]])
    print("Temporal Precision Margin=0", round_number(result["temporal_precision_0"]), " --> ", [round_number(x) for x in result["all_temporal_precision_0"]])
    print("Temporal Recall Margin=0", round_number(result["temporal_recall_0"]), " --> ", [round_number(x) for x in result["all_temporal_recall_0"]])
    print("--------------------")
    print ("Temporal F1 Margin=5: ", round_number(result["temporal_f1_5"]), " --> ", [round_number(x) for x in result["all_temporal_f1_5"]])
    print("Temporal Precision Margin=5", round_number(result["temporal_precision_5"]), " --> ", [round_number(x) for x in result["all_temporal_precision_5"]])
    print("Temporal Recall Margin=5", round_number(result["temporal_recall_5"]), " --> ", [round_number(x) for x in result["all_temporal_recall_5"]])
    print("--------------------")
    print("Temporal F1 Margin=10: ", round_number(result["temporal_f1_10"]), " --> ", [round_number(x) for x in result["all_temporal_f1_10"]])
    print("Temporal Precision Margin=10", round_number(result["temporal_precision_10"]), " --> ", [round_number(x) for x in result["all_temporal_precision_10"]])
    print("Temporal Recall Margin=10", round_number(result["temporal_recall_10"]), " --> ", [round_number(x) for x in result["all_temporal_recall_10"]])
    print("--------------------")
    print("Spatial mIOU Margin=0: ", round_number(result["spatial_miou_0"]), " --> ", [round_number(x) for x in result["all_spatial_miou_0"]])
    print("Spatial Thresholded Margin=0: ", round_number(result["spatial_thresholded_0"]), " --> ", [round_number(x) for x in result["all_spatial_thresholded_0"]])
    print("--------------------")
    print("Spatial mIOU Margin=5: ", round_number(result["spatial_miou_5"]), " --> ", [round_number(x) for x in result["all_spatial_miou_5"]])
    print("Spatial Thresholded Margin=5: ", round_number(result["spatial_thresholded_5"]), " --> ", [round_number(x) for x in result["all_spatial_thresholded_5"]])
    print("--------------------")
    print("Spatial mIOU Margin=10: ", round_number(result["spatial_miou_10"]), " --> ", [round_number(x) for x in result["all_spatial_miou_10"]])
    print("Spatial Thresholded Margin=10: ", round_number(result["spatial_thresholded_10"]), " --> ", [round_number(x) for x in result["all_spatial_thresholded_10"]])
    print("--------------------")
    print("Edit Operation: ", round_number(result["edit_operation"]), " --> ", [round_number(x) for x in result["all_edit_operation"]])
    #print("Parameters: ", result["parameters"])
    pass

def evaluate_all_tasks_spatial():
    task_ids = [2, 3, 4, 5, 6]
    # task_ids = [6]
    result = run_evaluation_spatial(
        task_ids = task_ids,
        indexes = [],
        # indexes=[0, 1]
    )

    with open(f"{PIPELINE_RESULTS_FOLDER}/{SPATIAL_ONLY_RESULTS_PREFIX}_{str(task_ids)}.json", "w") as f:
        json.dump(result, f, indent=2)

    if (len(result["dataset"]) == 0):
        return
    
    print("Summary:")
    print("--------------------")
    print("Spatial mIOU: ", round_number(result["spatial_miou"]), " --> ", [round_number(x) for x in result["all_spatial_miou"]])
    print("Spatial Thresholded: ", round_number(result["spatial_thresholded"]), " --> ", [round_number(x) for x in result["all_spatial_thresholded"]])
    print("--------------------")
    print("Spatial mIOU Margin=0: ", round_number(result["spatial_miou_0"]), " --> ", [round_number(x) for x in result["all_spatial_miou_0"]])
    print("Spatial Thresholded Margin=0: ", round_number(result["spatial_thresholded_0"]), " --> ", [round_number(x) for x in result["all_spatial_thresholded_0"]])
    print("--------------------")
    print("Spatial mIOU Margin=5: ", round_number(result["spatial_miou_5"]), " --> ", [round_number(x) for x in result["all_spatial_miou_5"]])
    print("Spatial Thresholded Margin=5: ", round_number(result["spatial_thresholded_5"]), " --> ", [round_number(x) for x in result["all_spatial_thresholded_5"]])
    print("--------------------")
    print("Spatial mIOU Margin=10: ", round_number(result["spatial_miou_10"]), " --> ", [round_number(x) for x in result["all_spatial_miou_10"]])
    print("Spatial Thresholded Margin=10: ", round_number(result["spatial_thresholded_10"]), " --> ", [round_number(x) for x in result["all_spatial_thresholded_10"]])
    print("--------------------")
    pass

def evaluate_single_task(only_temporal = False):
    result = run_evaluation_for_task(
        task_id = 6,
        data_point_getter = get_data_point_as_request if only_temporal == False else get_data_point,
        pipeline_runner = get_response_langchain if only_temporal == False else get_temporal_langchain_indexed,
        # indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[10] - video #[4] - position
        indexes = [0, 2, 4, 6, 8, 10]
        # indexes=[0, 1, 2]
    )

    if (len(result["dataset"]) == 0):
        return
    info = get_data_point_info(result["dataset"], 0)
    
    print("Summary:")
    print("Video Info: ", "(" + info["videoKnowledge"], info["videoChannel"] + ")", '"' + info["videoTitle"] + '"', "-", info["videoUrl"])
    # all_cosine_similarity": all_cosine_similarity,
    # "all_top_10_cosine_similarity": all_top_10_cosine_similarity,")
    print("--------------------")

    print("Temporal F1 Margin=0: ", round_number(result["temporal_f1_0"]), " --> ", [round_number(x) for x in result["all_temporal_f1_0"]])
    print("Temporal Precision Margin=0", round_number(result["temporal_precision_0"]), " --> ", [round_number(x) for x in result["all_temporal_precision_0"]])
    print("Temporal Recall Margin=0", round_number(result["temporal_recall_0"]), " --> ", [round_number(x) for x in result["all_temporal_recall_0"]])
    print("--------------------")
    print("Temporal F1 Margin=10: ", round_number(result["temporal_f1_10"]), " --> ", [round_number(x) for x in result["all_temporal_f1_10"]])
    print("Temporal Precision Margin=10", round_number(result["temporal_precision_10"]), " --> ", [round_number(x) for x in result["all_temporal_precision_10"]])
    print("Temporal Recall Margin=10", round_number(result["temporal_recall_10"]), " --> ", [round_number(x) for x in result["all_temporal_recall_10"]])
    print("Edit Operation: ", round_number(result["edit_operation"]), " --> ", [round_number(x) for x in result["all_edit_operation"]])
    #print("Parameters: ", result["parameters"])
    pass

def summarize_pipeline_results_full():
    MIOU_THRESHOLD = 0.1
    def summarize_array(arr, key):
        if "cosine" in key:
            return
        all_len = len(arr)
        all_avg = -1
        if "spatial_miou" in key:
            all_avg = 0
            for miou in arr:
                if miou > MIOU_THRESHOLD:
                    all_avg += 1
            if all_len > 0:
                all_avg = all_avg / all_len
            print(round(all_avg, 2), "\t", f"{key[len('all_'):]}({all_len})")
        avg, std = avg_std(arr)
        print(round(avg, 2), "\t", "STD: ", round(std, 2), "\t", f"{key[len('all_'):]}({all_len})")

    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{FULL_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in range(2, 7):
        filename = file_prefix + str(task_id) + ".json"
        print("task", task_id)
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            for key in result:
                if key not in combined_results:
                    combined_results[key] = []
                combined_results[key].extend(result[key])
                summarize_array(result[key], key)
            print("----------"*5)
            print("\n")

    for key in combined_results.keys():
        summarize_array(combined_results[key], key)

def summarize_pipeline_results_parsing():
    def summarize_array(arr, key):
        all_len = len(arr)
        avg, std = avg_std(arr)
        print(round(avg, 2), "\t", "STD: ", round(std, 2), "\t", f"{key}({all_len})")

    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{PARSING_RESULTS_PREFIX}_"
    results = []
    for task_id in range(2, 7):
        filename = file_prefix + str(task_id) + ".json"
        with open(filename, "r") as f:
            result = json.load(f)
            results.append(result)
    combined_result = {}
    edit_operation_f1 = []
    edit_operation_precision = []
    edit_operation_recall = []

    for task_idx, result in enumerate(results):
        print("task", task_idx + 2)
        print("----------"*5)
        task_result = {}
        task_f1 = []
        task_precision = []
        task_recall = []
        for intent_idx, single in enumerate(result):
            for key in single.keys():
                if key == "editOperations":
                    task_f1.append(single[key]["f1"])
                    task_precision.append(single[key]["precision"])
                    task_recall.append(single[key]["recall"])
                    continue
                if key not in task_result:
                    task_result[key] = []
                final_score = 0
                if len(single[key]['ground_truth']) == 0:
                    # if len(single[key]['prediction']) > 0:
                    #     print(f"WARNING: pipeline detected non-existent reference ({task_idx}-{intent_idx}-{key})", single[key]["prediction"])
                    continue
                else:
                    # if len(single[key]['prediction']) == 0:
                    #     print(f"WARNING: pipeline did not detect reference ({task_idx}-{intent_idx}-{key})", single[key]["ground_truth"])
                    expanded_ground_truth = []
                    for ground_truth in single[key]['ground_truth']:
                        expanded_ground_truth.extend([item.strip() for item in ground_truth.split(", ")])
                    if len(single[key]['prediction']) == 0:
                        print(f"WARNING: pipeline did not detect reference ({task_idx}-{intent_idx}-{key})", single[key]["ground_truth"])
                    else:
                        cosine_scores = get_cosine_similarity_score(
                            expanded_ground_truth,
                            single[key]['prediction'],
                        )
                        for single_cosine_scores in cosine_scores:
                            final_score += max([item.item() for item in single_cosine_scores])
                        final_score = final_score / len(cosine_scores)
                    # score_per_ground_truth = [0 for i in range(len(single[key]['ground_truth']))]
                    # for i, cosine_scores in enumerate(single[key]['cosine_scores']):
                    #     for j, cosine_score in enumerate(cosine_scores):
                    #         score_per_ground_truth[j] = max(score_per_ground_truth[j], cosine_score)
                    # final_score = sum(score_per_ground_truth) / len(score_per_ground_truth)
                    # if len(single[key]['prediction']) == 0:
                    #     print(f"WARNING: pipeline did not detect reference ({task_idx}-{intent_idx}-{key})", single[key]["ground_truth"])
                    # else:
                    #     cosine_scores = get_cosine_similarity_score(
                    #         [", ".join(single[key]['ground_truth'])],
                    #         [", ".join(single[key]['prediction'])],
                    #     )
                    #     final_score = cosine_scores[0][0].item()
                task_result[key].append(final_score)  
        
        summarize_array(task_f1, "editOperations_f1")
        summarize_array(task_precision, "editOperations_precision")
        summarize_array(task_recall, "editOperations_recall")
        for key in task_result.keys():
            summarize_array(task_result[key], key)
        print("----------"*5)
        print("\n")
        for key in task_result.keys():
            if key not in combined_result:
                combined_result[key] = []
            combined_result[key].extend(task_result[key])
        edit_operation_f1.extend(task_f1)
        edit_operation_precision.extend(task_precision)
        edit_operation_recall.extend(task_recall)

    
    if len(edit_operation_f1) > 0:
        summarize_array(edit_operation_f1, "editOperations_f1")
    if len(edit_operation_precision) > 0:
        summarize_array(edit_operation_precision, "editOperations_precision")
    if len(edit_operation_recall) > 0:
        summarize_array(edit_operation_recall, "editOperations_recall")

    for key in combined_result.keys():
        summarize_array(combined_result[key], key)

def summarize_pipeline_results_spatial():
    MIOU_THRESHOLD = 0.1
    def summarize_array(arr, key):
        all_len = len(arr)
        all_avg = 0
        for miou in arr:
            if miou > MIOU_THRESHOLD:
                all_avg += 1
        if all_len > 0:
            all_avg = all_avg / all_len
        print(round(all_avg, 2), "\t", f"{key}({all_len})")
        avg, std = avg_std(arr)
        print(round(avg, 2), "\t", "STD: ", round(std, 2), "\t", f"{key}({all_len})")

    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{SPATIAL_ONLY_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in range(2, 7):
        filename = file_prefix + str(task_id) + ".json"
        print("task", task_id)
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            for key in result:
                if key not in combined_results:
                    combined_results[key] = []
                combined_results[key].extend(result[key])
                summarize_array(result[key], key)
            print("----------"*5)
            print("\n")

    for key in combined_results.keys():
        summarize_array(combined_results[key], key)

def combine_pipeline_results():
    results_prefix = f"{PIPELINE_RESULTS_FOLDER}/{FULL_RESULTS_PREFIX}_"
    logs_prefix = f"{PIPELINE_RESULTS_FOLDER}/{LOGS_PREFIX}_"
    combined_results = []

    LINE_PREFIXES = [
        "!!!input!!!:",
        "!!!prediction!!!:",
        "!!!ground_truth!!!:",
    ]

    for task_id in range(2, 7):
        results_filename = results_prefix + str([task_id]) + ".json"
        logs_filename = logs_prefix + str(task_id) + ".txt"
        result = {}
        inputs = []
        predictions = []
        ground_truths = []
        with open(results_filename, "r") as f:
            result = json.load(f)
        with open(logs_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                for line_prefix in LINE_PREFIXES:
                    if line.startswith(line_prefix) == True:
                        main_part = line[len(line_prefix):].strip()
                        json_object = ast.literal_eval(main_part)
                        if line_prefix == LINE_PREFIXES[0]:
                            inputs.append(json_object)
                        elif line_prefix == LINE_PREFIXES[1]:
                            predictions.append(json_object)
                        elif line_prefix == LINE_PREFIXES[2]:
                            ground_truths.append(json_object)
        data_points = []
        for i, (input, prediction, ground_truth) in enumerate(zip(inputs, predictions, ground_truths)):
            cur_result = {}
            for key in result:
                if key.startswith("all_") == False or "cosine" in key:
                    continue
                print(key, result[key], i)
                cur_result[key[len("all_"):]] = result[key][i]
            data_points.append({
                "input": input,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "result": cur_result,
            }) 
        combined_results.extend(data_points)
    with open(f"{PIPELINE_RESULTS_FOLDER}/{SUMMARY_FILENAME}.json", "w") as f:
        json.dump(combined_results, f, indent=4)

def summarize_pipeline_results():
    summarize_pipeline_results_parsing()
    print("!!!!!!!!!!!"*5)
    summarize_pipeline_results_full()
    print("!!!!!!!!!!!"*5)
    summarize_pipeline_results_spatial()

if __name__ == "__main__":
    evaluate_all_tasks_full()