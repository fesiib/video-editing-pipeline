from evaluation.evaluator import *
from evaluation.evaluate_helpers import *
import math
import json

def round_number(number):
    return math.floor(number * 1000) / 1000

def run_pipeline_with_file(input):
    prompt_file_path = "./prompts/prompt_parse_intent.txt"
    return run_pipeline(input, prompt_file_path)

def main():
    # result = run_evaluation_for_task()

    result = run_evaluation_for_task(
        task_id = 6,
        data_point_getter = get_data_point,
        pipeline_runner = run_pipeline_with_file,
        indexes = [0, 10]
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
    print(result["content"])

def combine_dicts(dict1, dict2):
    result = {}
    for key in dict1:
        result[key] = [dict1[key]]
    for key in dict2:
        result[key] += [dict2[key]]
    return result

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
        # result = run_pipeline_test(input, "./prompts/generated_prompt.txt")
        # print("predicted (normal):     ", json.dumps(result_gen))
        # print("predicted (gen):  ", json.dumps(result))
        # print("predicted_combined   ", json.dumps(combine_dicts(result_gen, result)))
        # print("ground truth:        ", gt["relevant_text"])
        print("predicted & gt:      ", json.dumps(combine_dicts(result_gen, gt["relevant_text"]), indent=4))
        print("--------------------")

if __name__ == "__main__":
    main()
    #test()

    #summarize_prompt("./prompts/temporal_transcript.txt")