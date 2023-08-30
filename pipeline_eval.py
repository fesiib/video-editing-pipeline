from evaluation.evaluator import *
from evaluation.evaluate_helpers import *
import math

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

if __name__ == "__main__":
    main()