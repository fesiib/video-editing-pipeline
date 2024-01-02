import json
import ast

from evaluation.evaluator import *
from evaluation.evaluate_helpers import *

PIPELINE_RESULTS_FOLDER = "pipeline-results"

PARSING_RESULTS_PREFIX = "parsing"
SPATIAL_ONLY_RESULTS_PREFIX = "spatial_only"
TEMPORAL_ONLY_RESULTS_PREFIX = "temporal_only"
FULL_RESULTS_PREFIX = "full"
LOGS_PREFIX = "log"
SUMMARY_FILENAME = "combined_full_results"

MIOU_THRESHOLD = 0.1

def evaluate_all_tasks_parsing():
    task_ids = [2, 3, 4, 5, 6]
    for task_id in task_ids:
        result = run_evaulation_for_task_parsing(
            task_id = task_id,
            indexes = [],
        )
        with open(f"{PIPELINE_RESULTS_FOLDER}/{PARSING_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def evaluate_all_tasks_spatial():
    task_ids = [2, 3, 4, 5, 6]
    for task_id in task_ids:
        result = run_evaluatoin_for_task_spatial(
            task_id = task_id,
            indexes = [],
        )
        with open(f"{PIPELINE_RESULTS_FOLDER}/{SPATIAL_ONLY_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def evaluate_all_tasks_temporal():
    task_ids = [2, 3, 4, 5, 6]
    for task_id in task_ids:
        result = run_evaluation_for_task_full(
            task_id = task_id,
            data_point_getter = get_data_point,
            pipeline_runner = get_temporal_langchain_indexed,
            # indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[10] - video #[4] - position
            indexes=[]
        )
        with open(f"{PIPELINE_RESULTS_FOLDER}/{TEMPORAL_ONLY_RESULTS_PREFIX}_{str(task_id)}.json", "w") as f:
            json.dump(result, f, indent=2)

def evaluate_all_tasks_full():
    task_ids = [2, 3, 4, 5, 6]
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
        avg, std = avg_std(result[dim][metric][key])
        print("\tAVG:\t", round_number(avg), "\tSTD:\t", round_number(std), f"\t{dim}-{metric}-{key}:\t",  " <-- ", [round_number(x) for x in result[dim][metric][key]])

def print_evaluation_summary(result):
    def print_func(dim, metric, key):
        print_func_all(result, dim, metric, key)
    if "general" in result:
        print("\tGeneral:")
        print("\tTotal Requests Count:\t", len(result["general"]["predicted"]))
        print("--------------------")
    if "temporal" in result:
        print("\tTemporal:")
        print("\tTotal Edits Count:\t")
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
        print("\tSpatial:")
        print("\tTotal Edits Count:\t")
        print("--------------------")
        print_func("spatial", "count", "predicted")
        print_func("spatial", "count", "ground_truth")
        
        ## TODO: add miou thresholded
        if "pairs" in result["spatial"]:
            print("--------------------")
            print_func("miou", "pairs")
            print_func("thresholded", "pairs")
        print("--------------------")
        print_func("spatial", "miou", "0")
        print_func("spatial", "thresholded", "0")
        print("--------------------")
        print_func("spatial", "miou", "5")
        print_func("spatial", "thresholded", "5")
        print("--------------------")
        print_func("spatial", "miou", "10")
        print_func("spatial", "thresholded", "10")
        print("--------------------")
        print_func("spatial", "miou", "10")
        print_func("spatial", "thresholded", "10")
        print("--------------------")
    if "references" in result:
        print("\tEdit Operations:")
        print("\tTotal Edits Count:\t")
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
            for metric in result["comparison"][key]:
                if key == "editOperations":
                    if metric in ["f1", "precision", "recall"]:
                        local_result[metric].append(result["comparison"][key][metric])
                else:
                    # conservative
                    # if metric in ["f1", "precision", "recall"]:
                    #     local_result[metric].append(result["comparison"][key][metric])
                    
                    # expanded
                    if metric in ["f1_expanded", "precision_expanded", "recall_expanded"]:
                        local_result[metric.replace("_expanded", "")].append(result["comparison"][key][metric])
            for metric in local_result:
                avg, std = avg_std(local_result[metric])
                print("\tAVG:\t", round_number(avg), "\tSTD:\t", round_number(std), f"\t{metric}:\t",  " <-- ", [round_number(x) for x in local_result[metric]])

def summarize_pipeline_results_full(
    task_ids = [2, 3, 4, 5, 6],
):
    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{FULL_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in task_ids:
        filename = file_prefix + str(task_id) + ".json"
        print("Task Summary:", task_id)
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            print_evaluation_summary(result)
            combined_results = append_dict(combined_results, result)
            print("----------"*5)
    print("Summary ALL: ")
    print_evaluation_summary(combined_results)

def summarize_pipeline_results_parsing(
    task_ids = [2, 3, 4, 5, 6],
):

    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{PARSING_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in task_ids:
        filename = file_prefix + str(task_id) + ".json"
        print("Task Summary:", task_id)
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            print_evaluation_summary(result)
            combined_results = append_dict(combined_results, result)
            print("----------"*5)
    print("Summary ALL: ")
    print_evaluation_summary(combined_results)

def summarize_pipeline_results_spatial(
    task_ids = [2, 3, 4, 5, 6],
):
    file_prefix = f"{PIPELINE_RESULTS_FOLDER}/{SPATIAL_ONLY_RESULTS_PREFIX}_"
    combined_results = {}
    for task_id in task_ids:
        filename = file_prefix + str(task_id) + ".json"
        print("Task Summary:", task_id)
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            print_evaluation_summary(result)
            combined_results = append_dict(combined_results, result)
            print("----------"*5)
    print("Summary ALL: ")
    print_evaluation_summary(combined_results)

def summarize_pipeline_results():
    summarize_pipeline_results_parsing()
    print("!!!!!!!!!!!"*5)
    summarize_pipeline_results_full()
    print("!!!!!!!!!!!"*5)
    summarize_pipeline_results_spatial()

if __name__ == "__main__":
    evaluate_all_tasks_full()