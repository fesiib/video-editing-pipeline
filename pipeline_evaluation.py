import json
import ast

from evaluation.evaluator import *
from evaluation.evaluate_helpers import *

PIPELINE_RESULTS_FOLDER = "pipeline-results"

PARSING_RESULTS_PREFIX = "parsing"
SPATIAL_ONLY_RESULTS_PREFIX = "spatial_only"
TEMPORAL_ONLY_RESULTS_PREFIX = "temporal_only"
# FULL_RESULTS_PREFIX = "full_openai"
FULL_RESULTS_PREFIX = "full"
LOGS_PREFIX = "log"
SUMMARY_FILENAME = "combined_full_results"

MIOU_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
COSINE_SIMILARITY_THRESHOLD = 0.8

SPATIAL_SUFFIX = "_thresholded"

def evaluate_all_tasks_parsing(task_ids = [2, 3, 4, 5, 6]):
    for task_id in task_ids:
        result = run_evaulation_for_task_parsing(
            task_id = task_id,
            indexes = [],
        )
        with open(f"{PIPELINE_RESULTS_FOLDER}/{PARSING_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def evaluate_all_tasks_spatial(task_ids = [2, 3, 4, 5, 6]):
    for task_id in task_ids:
        result = run_evaluatoin_for_task_spatial(
            task_id = task_id,
            indexes = [],
            iou_threshold=IOU_THRESHOLD,
        )
        with open(f"{PIPELINE_RESULTS_FOLDER}/{SPATIAL_ONLY_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def evaluate_all_tasks_temporal(task_ids = [2, 3, 4, 5, 6]):
    for task_id in task_ids:
        result = run_evaluation_for_task_full(
            task_id = task_id,
            data_point_getter = get_data_point,
            pipeline_runner = get_temporal_langchain_indexed,
            # indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[10] - video #[4] - position
            indexes=[],
            iou_threshold=IOU_THRESHOLD,
        )
        with open(f"{PIPELINE_RESULTS_FOLDER}/{TEMPORAL_ONLY_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def evaluate_all_tasks_full(task_ids = [2, 3, 4, 5, 6]):
    for task_id in task_ids:
        result = run_evaluation_for_task_full(
            task_id = task_id,
            data_point_getter = get_data_point_as_request ,
            pipeline_runner = get_response_langchain,
            # indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[10] - video #[4] - position
            indexes=[]
        )
        with open(f"{PIPELINE_RESULTS_FOLDER}/{FULL_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def print_func_all(result, dim, metric, key):
        cur_list = []
        if key.endswith(SPATIAL_SUFFIX) and dim == "spatial":
            new_key = key.replace(SPATIAL_SUFFIX, "")
            cur_list = [*result[dim][metric][new_key]]
            cur_list = [1 if x >= MIOU_THRESHOLD else 0 for x in cur_list]
        else:
            cur_list = [*result[dim][metric][key]]
        avg, std = avg_std(cur_list)
        print("AVG:\t", round_number(avg), "STD:\t", round_number(std), f"\t{dim[0:4]}-{metric[0:4]}-{key[0:4]}:\t",  " <-- ", [round_number(x) for x in cur_list])

def print_evaluation_summary(result):
    def print_func(dim, metric, key):
        print_func_all(result, dim, metric, key)
    if "general" in result:
        print("\tGeneral:")
        print("\tTotal Requests Count:\t", len(result["general"]["predictions"]))
        print("--------------------")
    if "temporal" in result:
        print("\tTemporal:")
        print("--------------------")
        print_func("temporal", "count", "predicted")
        print_func("temporal", "count", "ground_truth")
        print("--------------------")
        print_func("temporal", "f1", "0")
        print_func("temporal", "precision", "0")
        print_func("temporal", "recall", "0")
        print("--------------------")
        print_func("temporal", "f1", "5")
        print_func("temporal", "precision", "5")
        print_func("temporal", "recall", "5")
        print("--------------------")
        print_func("temporal", "f1", "10")
        print_func("temporal", "precision", "10")
        print_func("temporal", "recall", "10")
        print("--------------------")
    if "spatial" in result:
        print(f"\tSpatial (MIOU_THRESH={MIOU_THRESHOLD} && THERSHOLDED={IOU_THRESHOLD}):")
        print("--------------------")
        print_func("spatial", "count", "predicted")
        print_func("spatial", "count", "ground_truth")
        if "pairs" in result["spatial"]["miou"] :
            print("--------------------")
            print_func("spatial", "miou", "pairs")
            print_func("spatial", "miou", "pairs" + SPATIAL_SUFFIX)
            print_func("spatial", "thresholded", "pairs")
        print("--------------------")
        print_func("spatial", "miou", "0")
        print_func("spatial", "miou", "0" + SPATIAL_SUFFIX)
        print_func("spatial", "thresholded", "0")
        print("--------------------")
        print_func("spatial", "miou", "5")
        print_func("spatial", "miou", "5" + SPATIAL_SUFFIX)
        print_func("spatial", "thresholded", "5")
        print("--------------------")
        print_func("spatial", "miou", "10")
        print_func("spatial", "miou", "10" + SPATIAL_SUFFIX)
        print_func("spatial", "thresholded", "10")
        print("--------------------")
    if "references" in result:
        print("\tEdit Operations:")
        print("--------------------")
        print_func_all(result["references"], "edit_operation", "count", "predicted")
        print_func_all(result["references"], "edit_operation", "count", "ground_truth")
        print("--------------------")
        print_func("references", "edit_operation", "f1")
        print_func("references", "edit_operation", "precision")
        print_func("references", "edit_operation", "recall")
        print("--------------------")
    if "comparison" in result:
        print("\tParsing:")
        print("--------------------")
        for key in result["comparison"]:
            local_result = {
                "f1": [],
                "precision": [],
                "recall": [],
            }
            print("\t\t", key, ":")
            for info in result["comparison"][key]:
                for metric in info:
                    if key == "editOperations":
                        if metric in ["f1", "precision", "recall"]:
                            local_result[metric].append(info[metric])
                    else:
                        # # conservative
                        # if metric in ["f1", "precision", "recall"]:
                        #     local_result[metric].append(info[metric])
                        
                        # expanded
                        if metric in ["f1_expanded", "precision_expanded", "recall_expanded"]:
                            # if info[metric] < 0:
                            #     local_result[metric.replace("_expanded", "")].append(-1)
                            #     continue
                            # score = 1
                            # if info[metric] < COSINE_SIMILARITY_THRESHOLD:
                            #     score = 0
                            # local_result[metric.replace("_expanded", "")].append(score)
                            local_result[metric.replace("_expanded", "")].append(info[metric])
            for metric in local_result:
                avg, std = avg_std(local_result[metric])
                print("AVG:\t", round_number(avg), "STD:\t", round_number(std), f"\t{metric[0:4]}:\t",  " <-- ", [round_number(x) for x in local_result[metric]])

def summarize_pipeline_results_full(
    task_ids = [2, 3, 4, 5, 6]
):
    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{FULL_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in task_ids:
        filename = file_prefix + str(task_id) + ".json"
        print(f"Task {task_id} Summary")
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)

            print_evaluation_summary(result)
            combined_results = append_dict(combined_results, result)
            print("----------"*5)
    print(f"Summary {task_ids}: ")
    print_evaluation_summary(combined_results)

def summarize_pipeline_results_parsing(
    task_ids = [2, 3, 4, 5, 6],
):

    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{PARSING_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in task_ids:
        filename = file_prefix + str(task_id) + ".json"
        print(f"Task {task_id} Summary")
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            print_evaluation_summary(result)
            combined_results = append_dict(combined_results, result)
            print("----------"*5)
    print(f"Summary {task_ids}: ")
    print_evaluation_summary(combined_results)

def summarize_pipeline_results_spatial(
    task_ids = [2, 3, 4, 5, 6],
):
    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{SPATIAL_ONLY_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in task_ids:
        filename = file_prefix + str(task_id) + ".json"
        print(f"Task {task_id} Summary")
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            print_evaluation_summary(result)
            combined_results = append_dict(combined_results, result)
            print("----------"*5)
    print(f"Summary {task_ids}: ")
    print_evaluation_summary(combined_results)

def summarize_pipeline_results():
    summarize_pipeline_results_parsing()
    print("!!!!!!!!!!!"*5)
    summarize_pipeline_results_full()
    print("!!!!!!!!!!!"*5)
    summarize_pipeline_results_spatial()

if __name__ == "__main__":
    #evaluate_all_tasks_parsing(task_ids=[2])
    # summarize_pipeline_results_parsing()
    #evaluate_all_tasks_full()
    summarize_pipeline_results_full()