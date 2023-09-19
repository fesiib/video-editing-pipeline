import json
import ast

from backend.intent_parser import IntentParser
from backend.pipeline import Pipeline
from LangChainPipeline import LangChainPipeline
from evaluation.evaluate_helpers import *
from evaluation.sentence_embedder import get_cosine_similarity_scores


# Using all metadata
intent_parser = IntentParser(50, 50)
pipeline = Pipeline(50, 0)
langchain_pipeline = LangChainPipeline(verbose=True)

# ground_truth = {
#     "editOperations": dataset[index]["edit_text"],
#     "edits": dataset[index]["temporal"],
# }

def reset_intent_parser(**props):
    intent_parser.reset(**props)

def run_gpt4(prompt_file_path, input):
    with open(prompt_file_path, 'r') as f:
        context = f.read()
    return intent_parser.completion_endpoint(context, input)

def run_pipeline_test_parser(input):
    return intent_parser.predict_relevant_text(input)

def run_pipeline_test_temporal(relevant_text):
    edits = intent_parser.predict_temporal_segments(
        relevant_text["temporal"], relevant_text["temporal_labels"])
    edits_temporal = []
    for edit in edits:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "edits": edits_temporal,
        "relevant_text": relevant_text,
    }
    return response

def run_pipeline_request(edit_request):
    edit_response = intent_parser.process_request(edit_request)
    edits_temporal = []
    for edit in edit_response["edits"]:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "editOperations": edit_response["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
        "edits": edits_temporal,
        "relevant_text": {
            "temporal": [],
            "spatial": [],
            "edit": edit_response["requestParameters"]["editOperations"],
        },
    }
    return response

def run_pipeline(input):
    relevant_text = intent_parser.predict_relevant_text(input)
    # run temporal parsing
    edits = intent_parser.predict_temporal_segments(
        relevant_text["temporal"], relevant_text["temporal_labels"])
    edits_temporal = []
    for edit in edits:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "editOperations": relevant_text["edit"],
        "parameters": relevant_text["parameters"],
        "edits": edits_temporal,
        "relevant_text": relevant_text,
    }
    return response

def run_pipeline_new(input):
    relevant_text = pipeline.predict_relevant_text(input)
    edits = pipeline.predict_temporal_segments(
        relevant_text["temporal"], relevant_text["temporal_labels"], [480, 854], [
            {
                "start": 0.234,
                "finish": 60*10.2345,
            },
            {
                "start": [12, 23.3],
                "finish": [0, 15, 0.3],
            }
        ]
    )
    edits_temporal = []
    for edit in edits:
        edits_temporal.append(
            [
                edit["temporalParameters"]["start"],
                edit["temporalParameters"]["finish"],
                edit["temporalParameters"]["info"],
                edit["temporalParameters"]["source"],
            ]
        )
    
    response = {
        "editOperations": relevant_text["edit"],
        "parameters": relevant_text["parameters"],
        "edits": edits_temporal,
        "relevant_text": relevant_text,
    }
    return response

def run_pipeline_request_new(edit_request):
    edit_response = pipeline.process_request(edit_request)
    edits_temporal = []
    for edit in edit_response["edits"]:
        edits_temporal.append([
            edit["temporalParameters"]["start"],
            edit["temporalParameters"]["finish"],
            edit["temporalParameters"]["info"],
            edit["temporalParameters"]["source"],
        ])
    
    response = {
        "editOperations": edit_response["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
        "edits": edits_temporal,
        "relevant_text": {
            "temporal": [],
            "spatial": [],
            "edit": edit_response["requestParameters"]["editOperations"],
        },
    }
    return response


def run_langchain_pipeline_temporal(input):
    references = langchain_pipeline.input_parser.run(input["text"])
    edits = langchain_pipeline.predict_temporal_segments(
        references.temporal, references.temporal_labels, 
        0, input["sketch_timestamp"],
        [480,854], [], 
    )
    edits_temporal = []
    edits_temporal_reasoning = []
    edits_spatial = []
    edits_spatial_reasoning = []
    for edit in edits:
        edits_temporal.append([
            edit["temporalParameters"]["start"],
            edit["temporalParameters"]["finish"],
        ])
        edits_temporal_reasoning.append([
            edit["temporalParameters"]["info"],
            edit["temporalParameters"]["source"],
        ])
        edits_spatial.append(edit["spatialParameters"])
        edits_spatial_reasoning.append([
            edit["spatialParameters"]["info"],
            edit["spatialParameters"]["source"],
        ])
    
    response = {
        "editOperations": references.edit,
        "parameters": references.get_parameters_short(),
        "edits": edits_temporal,
        "edits_temporal_reasoning": edits_temporal_reasoning,
        "edits_spatial": edits_spatial,
        "edits_spatial_reasoning": edits_spatial_reasoning,
        "relevant_text": {
            "temporal": references.temporal,
            "spatial": references.spatial,
            "edit": references.edit,
        },
    }
    return response

def run_langchain_pipeline_request(edit_request):
    edit_response = langchain_pipeline.process_request(edit_request)
    edits_temporal = []
    edits_temporal_reasoning = []
    edits_spatial = []
    edits_spatial_reasoning = []
    for edit in edit_response["edits"]:
        edits_temporal.append([
            edit["temporalParameters"]["start"],
            edit["temporalParameters"]["finish"],
        ])
        edits_temporal_reasoning.append([
            edit["temporalParameters"]["info"],
            edit["temporalParameters"]["source"],
        ])
        edits_spatial.append(edit["spatialParameters"])
        print(edit["spatialParameters"])
        edits_spatial_reasoning.append([
            edit["spatialParameters"]["info"],
            edit["spatialParameters"]["source"],
        ])
    
    response = {
        "editOperations": edit_response["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
        "edits": edits_temporal,
        "edits_temporal_reasoning": edits_temporal_reasoning,
        "edits_spatial": edits_spatial,
        "edits_spatial_reasoning": edits_spatial_reasoning,
        "relevant_text": {
            "temporal": [],
            "spatial": [],
            "edit": edit_response["requestParameters"]["editOperations"],
        },
    }
    return response

def run_evaluation_for_task(
    task_id = 6,
    data_point_getter = get_data_point_as_request,
    pipeline_runner = run_langchain_pipeline_request,
    indexes = []
):
    dataset = get_dataset_for_task(task_id)

    average_temporal_f1_0 = 0
    average_temporal_precision_0 = 0
    average_temporal_recall_0 = 0
    average_temporal_f1_10 = 0
    average_temporal_precision_10 = 0
    average_temporal_recall_10 = 0
    average_edit_operation = 0

    all_temporal_f1_0 = []
    all_temporal_precision_0 = []
    all_temporal_recall_0 = []
    all_temporal_f1_10 = []
    all_temporal_precision_10 = []
    all_temporal_recall_10 = []
    
    all_edit_operation = []

    all_cosine_similarity_temporal = []
    all_top_10_cosine_similarity_temporal = []
    
    all_cosine_similarity_spatial = []
    all_top_10_cosine_similarity_spatial = []

    if (len(dataset) == 0):
        return {
            "temporal_f1_0": 1,
            "temporal_precision_0": 1,
            "temporal_recall_0": 1,
            "temporal_f1_10": 1,
            "temporal_precision_10": 1,
            "temporal_recall_10": 1,
            "edit_operation": 1,
            "all_temporal_f1_0": all_temporal_f1_0,
            "all_temporal_precision_0": all_temporal_precision_0,
            "all_temporal_recall_0": all_temporal_recall_0,
            "all_temporal_f1_10": all_temporal_f1_10,
            "all_temporal_precision_10": all_temporal_precision_10,
            "all_temporal_recall_10": all_temporal_recall_10,
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

        (
            f1_0, precision_0, recall_0
        ), (
            f1_10, precision_10, recall_10
        ) = get_temporal_evaluation(prediction["edits"], ground_truth["edits"])
        # TODO: miou = get_spatial_evaluation(prediction["edits_spatial"], ground_truth["edits_spatial"])
        edit_operation = get_edit_operation_evaluation(prediction["editOperations"], ground_truth["editOperations"])

        average_temporal_f1_0 += f1_0
        average_temporal_precision_0 += precision_0
        average_temporal_recall_0 += recall_0
        average_temporal_f1_10 += f1_10
        average_temporal_precision_10 += precision_10
        average_temporal_recall_10 += recall_10

        average_edit_operation += edit_operation

        all_temporal_f1_0.append(f1_0)
        all_temporal_precision_0.append(precision_0)
        all_temporal_recall_0.append(recall_0)
        all_temporal_f1_10.append(f1_10)
        all_temporal_precision_10.append(precision_10)
        all_temporal_recall_10.append(recall_10)
        all_edit_operation.append(edit_operation)

        print("--------------------")
        print("!!!input!!!: ", input)
        print("!!!prediction!!!: ", prediction)
        print("!!!ground_truth!!!: ", ground_truth)
        print("!!!temporal evaluation margin=0!!!: ", "f1-margin-0: ", f1_0, "precision-margin-0: ", precision_0, "recall-margin-0: ", recall_0)
        print("!!!temporal evaluation margin=10!!!: ", "f1-margin-10: ", f1_10, "precision-margin-10: ", precision_10, "recall-margin-10: ", recall_10)
        print("!!!edit_op evaluation!!!: ", edit_operation)
        print("--------------------")
        print("!!!(temporal)cosine_similarity!!!: ", cosine_scores_temporal)
        print("!!!(temporal)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_temporal[0:4], indent=1))
        print("--------------------")
        print("!!!(spatial)cosine_similarity!!!: ", cosine_scores_spatial)
        print("!!!(spatial)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_spatial[0:4], indent=1))
        print("--------------------")

    average_temporal_f1_0 /= len(indexes)
    average_temporal_precision_0 /= len(indexes)
    average_temporal_recall_0 /= len(indexes)
    average_temporal_f1_10 /= len(indexes)
    average_temporal_precision_10 /= len(indexes)
    average_temporal_recall_10 /= len(indexes)
    average_edit_operation /= len(indexes)
    return {
        "temporal_f1_0": average_temporal_f1_0,
        "temporal_precision_0": average_temporal_precision_0,
        "temporal_recall_0": average_temporal_recall_0,
        "temporal_f1_10": average_temporal_f1_10,
        "temporal_precision_10": average_temporal_precision_10,
        "temporal_recall_10": average_temporal_recall_10,
        "edit_operation": average_edit_operation,
        "all_temporal_f1_0": all_temporal_f1_0,
        "all_temporal_precision_0": all_temporal_precision_0,
        "all_temporal_recall_0": all_temporal_recall_0,
        "all_temporal_f1_10": all_temporal_f1_10,
        "all_temporal_precision_10": all_temporal_precision_10,
        "all_temporal_recall_10": all_temporal_recall_10,
        "all_edit_operation": all_edit_operation,
        "all_cosine_similarity_temporal": all_cosine_similarity_temporal,
        "all_top_10_cosine_similarity_temporal": all_top_10_cosine_similarity_temporal,
        "all_cosine_similarity_spatial": all_cosine_similarity_spatial,
        "all_top_10_cosine_similarity_spatial": all_top_10_cosine_similarity_spatial,    
        "dataset": [item for i, item in enumerate(dataset) if i in indexes],
    }

def main():
    run_evaluation_for_task()
    pass

if __name__ == "__main__":
    main()