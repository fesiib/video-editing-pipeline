from evaluation.evaluator import *
from evaluation.evaluate_helpers import *
import math
import json

def round_number(number):
    return math.floor(number * 1000) / 1000

def main():
    # result = run_evaluation_for_task()

    result = run_evaluation_for_task(
        task_id = 6,
        data_point_getter = get_data_point,
        pipeline_runner = run_pipeline,
        indexes = [0]
    )

    if (len(result["dataset"]) == 0):
        return
    info = get_data_point_info(result["dataset"], 0)
    
    print("Summary:")
    print("Video Info: ", "(" + info["videoKnowledge"], info["videoChannel"] + ")", '"' + info["videoTitle"] + '"', "-", info["videoUrl"])

    print("--------------------")
    # all_cosine_similarity": all_cosine_similarity,
    # "all_top_10_cosine_similarity": all_top_10_cosine_similarity,")
    print("--------------------")

    print("Temporal F1: ", round_number(result["temporal_f1"]), " --> ", [round_number(x) for x in result["all_temporal_f1"]])
    print("Temporal Trad F1: ", round_number(result["temporal_traditional"]), " --> ", [round_number(x) for x in result["all_temporal_traditional"]])
    print("Edit Operation: ", round_number(result["edit_operation"]), " --> ", [round_number(x) for x in result["all_edit_operation"]])
    pass

def summarize_prompt(target_prompt_file_path):
    with open(target_prompt_file_path, 'r') as f:
        input = f.read()

    meta_promp_file_path = "./prompts/meta_prompt.txt"
    result = run_gpt4(
        meta_promp_file_path,
        input
    )
    print(result)

def test():

    dataset = get_dataset_for_task(6)
    # 0 -> transcript
    # 1 -> visual, action, transcript
    # 2 -> transcript, action
    # 3 -> timecodes
    # 4 -> video (beginning or intro) -> no metadata
    # 5 (same as 4) -> video (beginning or intro) -> no metadata
    # 6 -> transcript
    # 7 -> visual, action
    # 8 -> N/A or visual
    # 9 -> N/A or transcript & visual
    # 10 -> timecodes

    #indexes = [0, 1, 2, 6, 9] - transcript
    indexes = range(0, len(dataset))

    # implement a separate function that tests relevant text only for all data points

    for index in indexes:
        input, gt = get_data_point(dataset, index)
        result_gen = run_pipeline_test(input, "./prompts/prompt_parse_intent.txt")
        #result = run_pipeline_test(input, "./prompts/prompt.txt")
        print("predicted (gen):     ", result_gen)
        #print("predicted (normal):  ", result)
        print("ground truth:        ", gt["relevant_text"])
        print("--------------------")

if __name__ == "__main__":
    #main()
    test()

    #summarize_prompt("./prompts/prompt_parse_intent.txt")