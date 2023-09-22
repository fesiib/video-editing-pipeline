from evaluation.evaluator import *
from evaluation.evaluate_helpers import *
import math
import json

def round_number(number):
    return math.floor(number * 1000) / 1000

def main_evaluate_request():
    result = run_evaluation_for_task(
        task_id = 6,
        data_point_getter = get_data_point_as_request,
        pipeline_runner = run_langchain_pipeline_request,
        indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[10] - video #[4] - position
        # indexes = [1]
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

def main_evaluate_temporal():
    # result = run_evaluation_for_task()

    result = run_evaluation_for_task(
        task_id = 6,
        data_point_getter = get_data_point,
        pipeline_runner = run_langchain_pipeline_temporal_indexed,
        # indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[10] - video #[4] - position
        indexes = [0, 1, 2, 3]
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

    print("Temporal F1 Margin=0: ", round_number(result["temporal_f1_0"]), " --> ", [round_number(x) for x in result["all_temporal_f1_0"]])
    print("Temporal Precision Margin=0", round_number(result["temporal_precision_0"]), " --> ", [round_number(x) for x in result["all_temporal_precision_0"]])
    print("Temporal Recall Margin=0", round_number(result["temporal_recall_0"]), " --> ", [round_number(x) for x in result["all_temporal_recall_0"]])
    print("--------------------")
    print("Temporal F1 Margin=10: ", round_number(result["temporal_f1_10"]), " --> ", [round_number(x) for x in result["all_temporal_f1_10"]])
    print("Temporal Precision Margin=10", round_number(result["temporal_precision_10"]), " --> ", [round_number(x) for x in result["all_temporal_precision_10"]])
    print("Temporal Recall Margin=10", round_number(result["temporal_recall_10"]), " --> ", [round_number(x) for x in result["all_temporal_recall_10"]])
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

def combine_dicts(*dicts):
    result = {}
    for dict in dicts:
        for key in dict:
            result[key] = []
    for key in result:
        for dict in dicts:
            if key in dict:
                result[key].append([dict[key]])
            else:
                result[key].append([None])
    return result

def parsing_results(task_id):

    dataset = get_dataset_for_task(task_id)
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

    # implement a separate function that tests relevant text only for all data points
    indexes = [i for i in range(0, len(dataset))]
    for index in indexes:
        input, gt = get_data_point(dataset, index)
        print("INPUT: ", input)
        reset_intent_parser(
            prompt_parse_filename="./prompts/prompt_parse_intent.txt",
        )
        result_old = run_pipeline_test_parser(input)
        reset_intent_parser(
            prompt_parse_filename="./prompts/prompt_parse_intent_3.txt",
        )
        result_new = run_pipeline_test_parser(input)
        # print("predicted (normal):     ", json.dumps(result_gen))
        # print("predicted (gen):  ", json.dumps(result))
        # print("predicted_combined   ", json.dumps(combine_dicts(result_gen, result)))
        # print("ground truth:        ", gt["relevant_text"])
        print("new, old, gt", json.dumps(
            combine_dicts(result_new, result_old, gt["relevant_text"])
        , indent=2))
        print("--------------------")

    # indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # for index in indexes:
    #     input, gt = get_data_point(dataset, index)
    #     result = run_pipeline_test_temporal({
    #         "temporal": gt["relevant_text"]["temporal"],
    #         "temporal_labels": ["action" for _ in gt["relevant_text"]["temporal"]],
    #     })
    #     print("--------------------")
    #     print("input: ", gt["relevant_text"]["temporal"])
    #     print("prediction: ", json.dumps(result["edits"]))
    #     print("ground truth: ", json.dumps(gt["relevant_text"]["temporal"]))
    #     print("--------------------")

def summarize_captions(metadata_filename="./metadata/4LdIvyfzoGY_10.txt"):
    data = []
    with open(metadata_filename) as metadata:
        for line in metadata:
            interval = ast.literal_eval(line.rstrip())
            data.append({
                "start": interval["start"],
                "end": interval["end"],
                "dense": interval["dense_caption"],
                "synth": interval["synth_caption"],
                "action": interval["action_pred"],
                "transcript": interval["transcript"],
            })

    meta_promp_file_path = "./prompts/summarize_caption.txt"
    for i in range(0, 1):
        request_content = (
            "dense caption: " + data[i]["dense"].strip() + "\n"
            + "synthetic caption: " + data[i]["synth"].strip() + "\n"
            + "action prediction: " + data[i]["action"].strip() + "\n"
            + "transcript: " + data[i]["transcript"].strip() + "\n"
        )
        result = run_gpt4(
            meta_promp_file_path,
            request_content
        )
        summary = result["content"]
        print(request_content, "\n --> ", summary)

    #new_metadata_filename = metadata_filename.replace(".txt", "_summarized.txt")


if __name__ == "__main__":
    main_evaluate_request()
    #main_evaluate_temporal()
    
    
    # parsing_results(6)

    #summarize_prompt("./prompts/temporal_transcript.txt")
    # summarize_captions("./metadata/4LdIvyfzoGY_10.txt")