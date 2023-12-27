import json

from backend.operations import get_edit_segment
from LangChainPipeline import LangChainPipeline
from evaluation.evaluate_helpers import *
from evaluation.sentence_embedder import get_cosine_similarity_scores


# Using all metadata
langchain_pipeline = LangChainPipeline(verbose=True)


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

# Evaluate the pipeline on a single task (i.e data points from a single video)
def run_evaluation_for_task(
    task_id = 6,
    data_point_getter = get_data_point_as_request,
    pipeline_runner = get_response_langchain,
    indexes = [],
    iou_threshold = 0.5,
):
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

    if (len(dataset) == 0):
        return {
            "temporal": temporal,
            "spatial": spatial,
            "references": references,
            "dataset": evaluated_dataset,
        }
    
    if (len(indexes) == 0):
        indexes = range(len(dataset))

    indexes = [i for i in indexes if i < len(dataset)]

    for index in indexes:
        data_point = data_point_getter(dataset, index)
        input = data_point[0]
        ground_truth = data_point[1]
        prediction = pipeline_runner(input)

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
        print("\ttemporal evaluation count: ", "prediction: ", len(prediction["edits_temporal"]), "ground_truth: ", len(ground_truth["edits_temporal"]))
        print("\t\tmargin=0:\t", "f1: ", f1_0, "precision: ", precision_0, "recall: ", recall_0)
        print("\t\tmargin=5:\t", "f1: ", f1_5, "precision: ", precision_5, "recall: ", recall_5)
        print("\t\tmargin=10:\t", "f1: ", f1_10, "precision: ", precision_10, "recall: ", recall_10)
        print("\tspatial evaluation count: ", "prediction: ", len(prediction["edits_spatial"]), "ground_truth: ", len(ground_truth["edits_spatial"]))
        print("\t\tmargin=0:\t", "miou: ", miou_0, "thresholded: ", thresholded_0)
        print("\t\tmargin=5:\t", "miou: ", miou_5, "thresholded: ", thresholded_5)
        print("\t\tmargin=10:\t", "miou: ", miou_10, "thresholded: ", thresholded_10)

        print("\tedit_op evaluation count: ", "prediction: ", len(prediction["editOperations"]), "ground_truth: ", len(ground_truth["editOperations"]))
        print("\t\tsummary:\t", "f1", f1, "precision", precision, "recall", recall)
        print("--------------------")
        print("\t(temporal)cosine_similarity:\t", cosine_scores_temporal)
        print("\t(temporal)top_4_cosine_similarity:\t", json.dumps(top_10_pairs_temporal[0:4], indent=1))
        print("--------------------")
        print("\t(spatial)cosine_similarity:\t", cosine_scores_spatial)
        print("\t(spatial)top_4_cosine_similarity:\t", json.dumps(top_10_pairs_spatial[0:4], indent=1))
        print("--------------------")

    return {
        "temporal": temporal,
        "spatial": spatial,
        "references": references,
        "dataset": evaluated_dataset,
    }

# Evaluate the pipeline on `task_ids` tasks
def run_evaluation( 
    task_ids,
    data_point_getter = get_data_point_as_request,
    pipeline_runner = get_response_langchain,
    indexes = [],
    iou_threshold = 0.5,
):
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
    
    for task_id in task_ids:
        task_results = run_evaluation_for_task(
            task_id,
            data_point_getter,
            pipeline_runner,
            indexes,
            iou_threshold,
        )

        print("Statistics for task:\t", task_id)
        print("\tdata points count:\t", len(task_results["dataset"]))
        print("--------------------")
        print("\ttemporal evaluation count:\t",
                "prediction_avg_std", avg_std(task_results["temporal"]["count"]["prediction"]),
                "ground_truth_avg_std: ", avg_std(task_results["temporal"]["count"]["ground_truth"]),
        )
        for evaluation_parameters in ["0", "5", "10"]:
            print(f"\t\tmargin={evaluation_parameters}:\t", "f1: ", avg_std(task_results["temporal"]["f1"][evaluation_parameters]),
                "precision: ", avg_std(task_results["temporal"]["precision"][evaluation_parameters]),
                "recall: ", avg_std(task_results["temporal"]["recall"][evaluation_parameters])
            )
        
        print("\tspatial evaluation count:\t",
                "prediction_avg_std", avg_std(task_results["spatial"]["count"]["prediction"]),
                "ground_truth_avg_std: ", avg_std(task_results["spatial"]["count"]["ground_truth"]),
        )
        for evaluation_parameters in ["0", "5", "10"]:
            print(f"\t\tmargin={evaluation_parameters}:\t", "miou: ", avg_std(task_results["spatial"]["miou"][evaluation_parameters]),
                "thresholded: ", avg_std(task_results["spatial"]["thresholded"][evaluation_parameters])
            )
        
        print("\tedit_op evaluation count:\t",
                "prediction_avg_std", avg_std(task_results["references"]["edit_operation"]["count"]["prediction"]),
                "ground_truth_avg_std: ", avg_std(task_results["references"]["edit_operation"]["count"]["ground_truth"]),
        )
        print("\t\tsummary:\t", "f1", avg_std(task_results["edit"]["operation"]["f1"]),
            "precision", avg_std(task_results["edit"]["operation"]["precision"]),
            "recall", avg_std(task_results["edit"]["operation"]["recall"])
        )
        print("--------------------")

        temporal = append_dict(temporal, task_results["temporal"])
        spatial = append_dict(spatial, task_results["spatial"])
        references = append_dict(references, task_results["references"])
        evaluated_dataset.extend(task_results["dataset"])

    return {
        "temporal": temporal,
        "spatial": spatial,
        "references": references,    
        "dataset": evaluated_dataset,
    }

### spatial evaluation in isolation (given ground_truth segments)
def run_evaluation_spatial(
    task_ids,
    indexes = [],
    iou_threshold = 0.5, 
):
    results = {
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

    for task_id in task_ids:
        dataset = get_dataset_for_task(task_id)
        if (len(dataset) == 0):
            continue
        cur_indexes = []
        if (len(indexes) == 0):
            cur_indexes = range(len(dataset))
        else:
            for index in indexes:
                if index < len(dataset):
                    cur_indexes.append(index)

        if (len(cur_indexes) == 0):
            continue

        task_results = {
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

        for index in cur_indexes:
            data_point = get_data_point(dataset, index)
            input = data_point[0]
            ground_truth = data_point[1]
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
            for edit in ground_truth["edits"]:
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

            task_results["miou"]["0"].append(miou_0)
            task_results["thresholded"]["0"].append(thresholded_0)
            task_results["miou"]["5"].append(miou_5)
            task_results["thresholded"]["5"].append(thresholded_5)
            task_results["miou"]["10"].append(miou_10)
            task_results["thresholded"]["10"].append(thresholded_10)

            task_results["miou"]["pairs"].append(miou)
            task_results["thresholded"]["pairs"].append(thresholded)

            task_results["count"]["predicted"].append(len(prediction["edits_spatial"]))
            task_results["count"]["ground_truth"].append(len(ground_truth["edits_spatial"]))

            print("--------------------")
            print("!!!input!!!:\t", input)
            print("!!!prediction!!!:\t", prediction)
            print("!!!ground_truth!!!:\t", ground_truth)
            print("\tspatial evaluation count:\t", "prediction: ", len(prediction["edits_spatial"]), "ground_truth: ", len(ground_truth["edits_spatial"]))
            print("\t\tmargin=0:\t", "miou: ", miou_0, "thresholded: ", thresholded_0)
            print("\t\tmargin=5:\t", "miou: ", miou_5, "thresholded: ", thresholded_5)
            print("\t\tmargin=10:\t", "miou: ", miou_10, "thresholded: ", thresholded_10)
            print("\t\tpairs:\t", "miou: ", miou, "thresholded: ", thresholded)
            print("--------------------")

        print("Spatial Statistics for task: ", task_id)
        print("\tdata points count:\t", len(evaluated_dataset))
        print("--------------------")
        print("\tspatial evaluation count:\t",
                "prediction_avg_std", avg_std(task_results["count"]["predicted"]),
                "ground_truth_avg_std: ", avg_std(task_results["count"]["ground_truth"]),
        )
        for evaluation_parameters in ["0", "5", "10", "pairs"]:
            label = ("margin=" + evaluation_parameters) if evaluation_parameters != "pairs" else "pairs"
            print(f"\t\t{label}:\t", "miou: ", avg_std(task_results["miou"][evaluation_parameters]),
                "thresholded: ", avg_std(task_results["threshold"][evaluation_parameters])
            )
        print("--------------------")
        # append results for task
        results = append_dict(results, task_results)
        
    return {
        "results": results,
        "dataset": evaluated_dataset,
    }

def main():
    run_evaluation_for_task()
    pass

if __name__ == "__main__":
    main()