import json

from backend.operations import get_edit_segment
from LangChainPipeline import LangChainPipeline
from evaluation.evaluate_helpers import *
from evaluation.sentence_embedder import get_cosine_similarity_scores


# Using all metadata
langchain_pipeline = LangChainPipeline(temperature=0.0, verbose=True)


def get_references_langchain(input):
    langchain_pipeline.set_video(input["videoId"], 10)
    references = langchain_pipeline.indexed_input_parser.run(input["text"])
    simple_references = references.get_simple_references()
    parameters = simple_references.get_parameters_short()
    flattened_parameters = set()
    for key in parameters.keys():
        flattened_parameters = flattened_parameters.union(parameters[key])
    flattened_parameters = list(flattened_parameters)

    response = {
        "editOperations": simple_references.edit,
        "temporal": simple_references.temporal,
        "spatial": simple_references.spatial,
        "edit": [item.reference for item in references.edit_references],
        "parameters": flattened_parameters,
    }
    return response

def get_temporal_langchain_indexed(input):
    return get_temporal_langchain(input, indexed=True)

def get_temporal_langchain(input, indexed=False):
    langchain_pipeline.set_video(input["videoId"], 10)
    references = None
    temporal = []
    temporal_labels = []
    temporal_offsets = []
    if indexed == True:
        references = langchain_pipeline.indexed_input_parser.run(input["text"])
        temporal = [item.reference for item in references.temporal_references]
        temporal_labels = references.temporal_labels
        temporal_offsets = [item.offset for item in references.temporal_references]
    else:
        references = langchain_pipeline.input_parser.run(input["text"])
        temporal = references.temporal
        temporal_labels = references.temporal_labels
        temporal_offsets = [-1 for _ in temporal]

    edits = langchain_pipeline.predict_temporal_segments(
        input["text"],
        temporal, temporal_labels, temporal_offsets,
        0, input["sketch_timestamp"],
        input["video_shape"], [],
        input["video_duration"],
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
            edit["temporalParameters"]["offsets"],
        ])
        edits_spatial.append(edit["spatialParameters"])
        edits_spatial_reasoning.append([
            edit["spatialParameters"]["info"],
            edit["spatialParameters"]["source"],
            edit["temporalParameters"]["offsets"],
        ])
    
    response = {
        "editOperations": references.edit,
        "parameters": references.get_parameters_short(),
        "edits_temporal": edits_temporal,
        "edits_temporal_reasoning": edits_temporal_reasoning,
        "edits_spatial": edits_spatial,
        "edits_spatial_reasoning": edits_spatial_reasoning,
        "relevant_text": {
            "temporal": temporal,
            "spatial": [item.reference for item in references.spatial_references],
            "edit": [item.reference for item in references.edit_references],
            "indexed_temporal": [[item.offset, item.reference] for item in references.temporal_references],
            "indexed_spatial": [[item.offset, item.reference] for item in references.spatial_references],
            "indexed_edit": [[item.offset, item.reference] for item in references.edit_references],
        },
    }
    return response

def get_response_langchain(edit_request):
    langchain_pipeline.set_video(edit_request["videoId"], 10)
    edit_response = langchain_pipeline.process_request_indexed(edit_request)
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
            edit["temporalParameters"]["offsets"],
        ])
        edits_spatial.append(edit["spatialParameters"])
        print(edit["spatialParameters"])
        edits_spatial_reasoning.append([
            edit["spatialParameters"]["info"],
            edit["spatialParameters"]["source"],
            edit["temporalParameters"]["offsets"],
        ])
    
    relevant_text = edit_response["requestParameters"]["relevantText"]

    response = {
        "editOperations": edit_response["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
        "edits_temporal": edits_temporal,
        "edits_temporal_reasoning": edits_temporal_reasoning,
        "edits_spatial": edits_spatial,
        "edits_spatial_reasoning": edits_spatial_reasoning,
        "relevant_text": {
            "temporal": relevant_text["temporal"],
            "spatial": relevant_text["spatial"],
            "edit": relevant_text["edit"],
        },
    }
    return response

### Evaluate the parsing on a single task (i.e data points from a single video)
def run_evaulation_for_task_parsing(
    task_id = 6,
    indexes = [],
    data_point_getter = get_data_point_parsing,
):
    general = {
        "inputs": [],
        "predictions": [],
        "ground_truths": [],
    }
    comparison = {
        "editOperations": [],
        "temporal": [],
        "spatial": [],
        "edit": [],
        "parameters": [],
    }

    evaluated_dataset = []
    dataset = get_dataset_for_task(task_id)

    if len(indexes) == 0:
        indexes = range(len(dataset))

    for index in indexes:
        data_point = data_point_getter(dataset, index)
        if data_point == None:
            continue
        input = data_point[0]
        ground_truth = data_point[1]
        prediction = get_references_langchain(input)
        
        general["inputs"].append(input)
        general["predictions"].append(prediction)
        general["ground_truths"].append(ground_truth)

        evaluated_dataset.append(dataset[index])

        print("--------------------")
        print("!!!input!!!: ", input)
        print("--------------------")

        for key in prediction.keys():
            if key not in comparison:
                comparison[key] = []
            print(f"{key}:")
            print("\t!!!prediction!!!:\t", prediction[key])
            print("\t!!!ground_truth!!!:\t", ground_truth[key])
            
            if key == "editOperations":
                f1, precision, recall = get_edit_operation_evaluation(
                    prediction[key],
                    ground_truth[key],
                )
                comparison[key].append({
                    "prediction": prediction[key],
                    "ground_truth": ground_truth[key],
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                })
                print("\t\tf1:\t", f1)
                print("\t\tprecision:\t", precision)
                print("\t\trecall:\t", recall)
            else:
                # expand the ground truth and calculate a final score
                (
                    f1,
                    precision,
                    recall,
                    f1_expanded,
                    precision_expanded,
                    recall_expanded,
                    cosine_scores_expanded,
                    top_10_pairs_expanded,
                    cosine_scores,
                    top_10_pairs,
                ) = get_references_evaluation(
                    prediction[key],
                    ground_truth[key],
                )
                comparison[key].append({
                    "prediction": prediction[key],
                    "ground_truth": ground_truth[key],
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "f1_expanded": f1_expanded,
                    "precision_expanded": precision_expanded,
                    "recall_expanded": recall_expanded,
                    "cosine_scores_expanded": cosine_scores_expanded,
                    "top_10_pairs_expanded": top_10_pairs_expanded,
                    "cosine_scores": cosine_scores,
                    "top_10_pairs": top_10_pairs,
                })
                print("\t\tcosine_similarity:\t", cosine_scores)
                print("\t\ttop_4_cosine_similarity:\t", json.dumps(top_10_pairs[0:4], indent=1))
            print("--------------------")
    return {
        "general": general,
        "comparison": comparison,
        "dataset": evaluated_dataset,
    }
        
### Evaluate the pipeline on a single task (i.e data points from a single video)
def run_evaluation_for_task_full(
    task_id = 6,
    data_point_getter = get_data_point_as_request,
    pipeline_runner = get_response_langchain,
    indexes = [],
    iou_threshold = 0.5,
):
    
    general = {
        "inputs": [],
        "predictions": [],
        "ground_truths": [],
    }

    temporal = {
        "count": {
            "predicted": [],
            "ground_truth": [],
        },
        "f1": {
            "0": [],
            "5": [],
            "10": [],
        },
        "precision": {
            "0": [],
            "5": [],
            "10": [],
        },
        "recall": {
            "0": [],
            "5": [],
            "10": [],
        },
    }
    spatial = {
        "count": {
            "predicted": [],
            "ground_truth": [],
        },
        "miou": {
            "0": [],
            "5": [],
            "10": [],
        },
        "thresholded": {
            "0": [],
            "5": [],
            "10": [],
        },
    }

    references = {
        "edit_operation": {
            "count": {
                "predicted": [],
                "ground_truth": [],
            },
            "f1": [],
            "precision": [],
            "recall": [],
        },
        "temporal": {
            "cos_sim": [],
            "top_10": [],
        },
        "spatial": {
            "cos_sim": [],
            "top_10": [],
        }
    }
    evaluated_dataset = []

    dataset = get_dataset_for_task(task_id)

    if (len(indexes) == 0):
        indexes = range(len(dataset))

    for index in indexes:
        data_point = data_point_getter(dataset, index)
        if data_point == None:
            continue
        input = data_point[0]
        ground_truth = data_point[1]
        prediction = pipeline_runner(input)

        general["inputs"].append(input)
        general["predictions"].append(prediction)
        general["ground_truths"].append(ground_truth)

        evaluated_dataset.append(dataset[index])

        # temporal references evaluation
        cosine_scores_temporal, top_10_pairs_temporal = get_cosine_similarity_scores(
            prediction["relevant_text"]["temporal"],
            ground_truth["relevant_text"]["temporal"]
        )
        references["temporal"]["cos_sim"].append(cosine_scores_temporal)
        references["temporal"]["top_10"].append(top_10_pairs_temporal)

        # spatial references evaluation
        cosine_scores_spatial, top_10_pairs_spatial = get_cosine_similarity_scores(
            prediction["relevant_text"]["spatial"],
            ground_truth["relevant_text"]["spatial"]
        )
        references["spatial"]["cos_sim"].append(cosine_scores_spatial)
        references["spatial"]["top_10"].append(top_10_pairs_spatial)
        
        (f1, precision, recall) = get_edit_operation_evaluation(prediction["editOperations"], ground_truth["editOperations"])
        references["edit_operation"]["f1"].append(f1)
        references["edit_operation"]["precision"].append(precision)
        references["edit_operation"]["recall"].append(recall)
        references["edit_operation"]["count"]["predicted"].append(len(prediction["editOperations"]))
        references["edit_operation"]["count"]["ground_truth"].append(len(ground_truth["editOperations"]))

        # temporal evaluation
        (
            ( f1_0, precision_0, recall_0 ),
            ( f1_5, precision_5, recall_5 ),
            ( f1_10, precision_10, recall_10),
        ) = get_temporal_evaluation(prediction["edits_temporal"], ground_truth["edits_temporal"])
        temporal["f1"]["0"].append(f1_0)
        temporal["precision"]["0"].append(precision_0)
        temporal["recall"]["0"].append(recall_0)
        temporal["f1"]["5"].append(f1_5)
        temporal["precision"]["5"].append(precision_5)
        temporal["recall"]["5"].append(recall_5)
        temporal["f1"]["10"].append(f1_10)
        temporal["precision"]["10"].append(precision_10)
        temporal["recall"]["10"].append(recall_10)
        temporal["count"]["predicted"].append(len(prediction["edits_temporal"]))
        temporal["count"]["ground_truth"].append(len(ground_truth["edits_temporal"]))
        
        # spatial evaluation
        (
            (miou_0, thresholded_0),
            (miou_5, thresholded_5),
            (miou_10, thresholded_10),
        ) = get_spatial_evaluation(
            prediction["edits_spatial"],
            ground_truth["edits_spatial"],
            prediction["edits_temporal"],
            ground_truth["edits_temporal"],
            iou_threshold,
        )
        spatial["miou"]["0"].append(miou_0)
        spatial["thresholded"]["0"].append(thresholded_0)
        spatial["miou"]["5"].append(miou_5)
        spatial["thresholded"]["5"].append(thresholded_5)
        spatial["miou"]["10"].append(miou_10)
        spatial["thresholded"]["10"].append(thresholded_10)
        spatial["count"]["predicted"].append(len(prediction["edits_spatial"]))
        spatial["count"]["ground_truth"].append(len(ground_truth["edits_spatial"]))

        print("--------------------")
        print("!!!input!!!: ", input)
        print("!!!prediction!!!: ", prediction)
        print("!!!ground_truth!!!: ", ground_truth)
        print("--------------------")
        print("\ttemporal evaluation count: ", "prediction: ", len(prediction["edits_temporal"]), "ground_truth: ", len(ground_truth["edits_temporal"]))
        print("\t\tmargin=0:\t", "f1: ", f1_0, "precision: ", precision_0, "recall: ", recall_0)
        print("\t\tmargin=5:\t", "f1: ", f1_5, "precision: ", precision_5, "recall: ", recall_5)
        print("\t\tmargin=10:\t", "f1: ", f1_10, "precision: ", precision_10, "recall: ", recall_10)
        print("--------------------")
        print("\tspatial evaluation count: ", "prediction: ", len(prediction["edits_spatial"]), "ground_truth: ", len(ground_truth["edits_spatial"]))
        print("\t\tmargin=0:\t", "miou: ", miou_0, "thresholded: ", thresholded_0)
        print("\t\tmargin=5:\t", "miou: ", miou_5, "thresholded: ", thresholded_5)
        print("\t\tmargin=10:\t", "miou: ", miou_10, "thresholded: ", thresholded_10)
        print("--------------------")
        print("\tedit_op evaluation count: ", "prediction: ", len(prediction["editOperations"]), "ground_truth: ", len(ground_truth["editOperations"]))
        print("\t\tsummary:\t", "f1", f1, "precision", precision, "recall", recall)
        print("--------------------")
        print("\t(temporal)cosine_similarity:\t", cosine_scores_temporal)
        print("\t(temporal)top_4_cosine_similarity:\t", json.dumps(top_10_pairs_temporal[0:4], indent=1))
        print("\t(spatial)cosine_similarity:\t", cosine_scores_spatial)
        print("\t(spatial)top_4_cosine_similarity:\t", json.dumps(top_10_pairs_spatial[0:4], indent=1))
        print("--------------------")

    return {
        "general": general,
        "temporal": temporal,
        "spatial": spatial,
        "references": references,
        "dataset": evaluated_dataset,
    }

### spatial evaluation in isolation (given ground_truth segments)
def run_evaluatoin_for_task_spatial(
    task_id = 6,
    indexes = [],
    data_point_getter = get_data_point,
    iou_threshold = 0.5, 
):
    general = {
        "inputs": [],
        "predictions": [],
        "ground_truths": [],
    }
    spatial = {
        "count": {
            "predicted": [],
            "ground_truth": [],
        },
        "miou": {
            "pairs": [],
            "0": [],
            "5": [],
            "10": [],
        },
        "thresholded": {
            "pairs": [],
            "0": [],
            "5": [],
            "10": [],
        },
    }

    evaluated_dataset = []

    dataset = get_dataset_for_task(task_id)
    
    if (len(indexes) == 0):
        indexes = range(len(dataset))

    for index in indexes:
        data_point = data_point_getter(dataset, index)
        if data_point == None:
            continue
        input = data_point[0]
        ground_truth = data_point[1]

        general["inputs"].append(input)
        general["ground_truths"].append(ground_truth)

        count_spatial_gt = 0
        for spatial_gts in ground_truth["edits_spatial"]:
            if len(spatial_gts) > 0:
                count_spatial_gt += 1

        if count_spatial_gt == 0:
            continue
        
        evaluated_dataset.append(dataset[index])
        langchain_pipeline.set_video(input["videoId"], 10)

        sketches = input["sketch"]
        sketch_timestamp = input["sketch_timestamp"]
        for sketch in sketches:
            sketch["timestamp"] = sketch_timestamp
        video_shape = input["video_shape"]

        edits = []
        for edit in ground_truth["edits_temporal"]:
            start = edit[0]
            finish = edit[1]
            explanation = ["ground_truth"]
            source = ["ground_truth"]
            offsets = [-1]
            edit = get_edit_segment(start, finish, explanation, source, offsets, video_shape)
            edits.append(edit)       

        references = langchain_pipeline.indexed_input_parser.run(input["text"])
        simple_references = references.get_simple_references()
        edits = langchain_pipeline.predict_spatial_locations_new(
            input["text"],
            simple_references.spatial, simple_references.spatial_labels,
            [item.offset for item in references.spatial_references],
            edits, sketches, video_shape,
            sketch_timestamp
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
                edit["temporalParameters"]["offsets"],
            ])
            edits_spatial.append(edit["spatialParameters"])
            edits_spatial_reasoning.append([
                edit["spatialParameters"]["info"],
                edit["spatialParameters"]["source"],
                edit["temporalParameters"]["offsets"],
            ])
        prediction = {
            "editOperations": simple_references.edit,
            "parameters": simple_references.get_parameters_short(),
            "edits_temporal": edits_temporal,
            "edits_temporal_reasoning": edits_temporal_reasoning,
            "edits_spatial": edits_spatial,
            "edits_spatial_reasoning": edits_spatial_reasoning,
            "relevant_text": {
                "temporal": simple_references.temporal,
                "spatial": simple_references.spatial,
                "edit": [item.reference for item in references.edit_references],
                "parameters": simple_references.get_parameters(),
            },
        }

        (
            (miou_0, thresholded_0),
            (miou_5, thresholded_5),
            (miou_10, thresholded_10),
        ) = get_spatial_evaluation(
            prediction["edits_spatial"],
            ground_truth["edits_spatial"],
            prediction["edits_temporal"],
            ground_truth["edits_temporal"],
            iou_threshold,
        )

        (miou, thresholded) = get_spatial_evaluation_pairs(
            prediction["edits_spatial"],
            ground_truth["edits_spatial"],
            iou_threshold,
        )


        general["predictions"].append(prediction)

        spatial["miou"]["0"].append(miou_0)
        spatial["thresholded"]["0"].append(thresholded_0)
        spatial["miou"]["5"].append(miou_5)
        spatial["thresholded"]["5"].append(thresholded_5)
        spatial["miou"]["10"].append(miou_10)
        spatial["thresholded"]["10"].append(thresholded_10)

        spatial["miou"]["pairs"].append(miou)
        spatial["thresholded"]["pairs"].append(thresholded)

        spatial["count"]["predicted"].append(len(prediction["edits_spatial"]))
        spatial["count"]["ground_truth"].append(len(ground_truth["edits_spatial"]))

        print("--------------------")
        print("!!!input!!!:\t", input)
        print("!!!prediction!!!:\t", prediction)
        print("!!!ground_truth!!!:\t", ground_truth)
        print("--------------------")
        print("\tspatial evaluation count:\t", "prediction: ", len(prediction["edits_spatial"]), "ground_truth: ", len(ground_truth["edits_spatial"]))
        print("\t\tmargin=0:\t", "miou: ", miou_0, "thresholded: ", thresholded_0)
        print("\t\tmargin=5:\t", "miou: ", miou_5, "thresholded: ", thresholded_5)
        print("\t\tmargin=10:\t", "miou: ", miou_10, "thresholded: ", thresholded_10)
        print("\t\tpairs:\t", "miou: ", miou, "thresholded: ", thresholded)
        print("--------------------")
    return {
        "general": general,
        "spatial": spatial,
        "dataset": evaluated_dataset,
    }

def main():
    run_evaluation_for_task_full()
    pass

if __name__ == "__main__":
    main()