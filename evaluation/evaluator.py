import json

from backend.intent_parser import IntentParser
from evaluation.evaluate_helpers import *
from evaluation.sentence_embedder import get_cosine_similarity_scores


# Using all metadata
intent_parser = IntentParser(50, 0)

# ground_truth = {
#     "editOperations": dataset[index]["edit_text"],
#     "edits": dataset[index]["temporal"],
# }

def run_gpt4(prompt_file_path, input):
    with open(prompt_file_path, 'r') as f:
        context = f.read()
    return intent_parser.completion_endpoint(context, input)

def run_pipeline_test(input, prompt_file_path):
    intent_parser.reset()
    return intent_parser.predict_relevant_text(input, prompt_file_path)

def run_pipeline_request(edit_request):
    intent_parser.reset()
    edit_response = intent_parser.process_request(edit_request)
    edits_temporal = []
    for edit in edit_response["edits"]:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "editOperation": edit_response["requestParameters"]["editOperation"],
        "edits": edits_temporal,
        "relevant_text": {
            "temporal": [],
            "spatial": [],
            "edit": [edit_response["requestParameters"]["editOperation"]],
        },
    }
    return response

def run_pipeline(input):
    intent_parser.reset()
    relevant_text = intent_parser.predict_relevant_text(input)
    # run temporal parsing
    edits = intent_parser.predict_temporal_segments(relevant_text["temporal"])
    edits_temporal = []
    for edit in edits:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "editOperation": relevant_text["edit"][0],
        "edits": edits_temporal,
        "relevant_text": relevant_text,
    }
    return response

def run_evaluation_for_task(task_id = 6, data_point_getter = get_data_point_as_request, pipeline_runner = run_pipeline_request, indexes = []):
    dataset = get_dataset_for_task(task_id)

    average_temporal_f1 = 0
    average_temporal_traditional = 0
    average_edit_operation = 0

    all_temporal_f1 = []
    all_temporal_traditional = []
    all_edit_operation = []

    all_cosine_similarity_temporal = []
    all_top_10_cosine_similarity_temporal = []
    
    all_cosine_similarity_spatial = []
    all_top_10_cosine_similarity_spatial = []

    if (len(dataset) == 0):
        return {
            "temporal_f1": 1,
            "temporal_traditional": 1,
            "edit_operation": 1,
            "all_temporal_f1": all_temporal_f1,
            "all_temporal_traditional": all_temporal_traditional,
            "all_edit_operation": all_edit_operation,
            "all_cosine_similarity_temporal": all_cosine_similarity_temporal,
            "all_top_10_cosine_similarity_temporal": all_top_10_cosine_similarity_temporal,
            "all_cosine_similarity_spatial": all_cosine_similarity_spatial,
            "all_top_10_cosine_similarity_spatial": all_top_10_cosine_similarity_spatial,
            "dataset": dataset,
        }
    
    if (len(indexes) == 0):
        indexes = range(len(dataset))

    indexes = [i for i in indexes if i < len(dataset)]

    for index in indexes:
        data_point = data_point_getter(dataset, index)
        input = data_point[0]
        ground_truth = data_point[1]
        prediction = pipeline_runner(input)

        # cosine similarity if possible
        cosine_scores_temporal, top_10_pairs_temporal = get_cosine_similarity_scores(
            prediction["relevant_text"]["temporal"],
            ground_truth["relevant_text"]["temporal"]
        )
        cosine_scores_spatial, top_10_pairs_spatial = get_cosine_similarity_scores(
            prediction["relevant_text"]["spatial"],
            ground_truth["relevant_text"]["spatial"]
        )

        all_top_10_cosine_similarity_temporal.append(top_10_pairs_temporal)
        all_cosine_similarity_temporal.append(cosine_scores_temporal)

        (f1, traditional) = get_temporal_evaluation(prediction["edits"], ground_truth["edits"])
        edit_operation = get_edit_operation_evaluation(prediction["editOperation"], ground_truth["editOperations"])

        average_temporal_f1 += f1
        average_temporal_traditional += traditional
        average_edit_operation += edit_operation

        all_temporal_f1.append(f1)
        all_temporal_traditional.append(traditional)
        all_edit_operation.append(edit_operation)

        print("--------------------")
        print("!!!prediction!!!: ", prediction)
        print("!!!ground_truth!!!: ", ground_truth)
        print("!!!temporal evaluation!!!: ", "f1-score: ", f1, "traditional: ", traditional)
        print("!!!edit_op evaluation!!!: ", edit_operation)
        print("--------------------")
        print("!!!(temporal)cosine_similarity!!!: ", cosine_scores_temporal)
        print("!!!(temporal)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_temporal[0:4], indent=1))
        print("--------------------")
        print("!!!(spatial)cosine_similarity!!!: ", cosine_scores_spatial)
        print("!!!(spatial)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_spatial[0:4], indent=1))
        print("--------------------")

    average_temporal_f1 /= len(dataset)
    average_temporal_traditional /= len(dataset)
    average_edit_operation /= len(dataset)
    return {
        "temporal_f1": average_temporal_f1,
        "temporal_traditional": average_temporal_traditional,
        "edit_operation": average_edit_operation,
        "all_temporal_f1": all_temporal_f1,
        "all_temporal_traditional": all_temporal_traditional,
        "all_edit_operation": all_edit_operation,
        "all_cosine_similarity_temporal": all_cosine_similarity_temporal,
        "all_top_10_cosine_similarity_temporal": all_top_10_cosine_similarity_temporal,
        "all_cosine_similarity_spatial": all_cosine_similarity_spatial,
        "all_top_10_cosine_similarity_spatial": all_top_10_cosine_similarity_spatial,    
        "dataset": dataset,
    }

def main():
    run_evaluation_for_task()
    pass

if __name__ == "__main__":
    main()