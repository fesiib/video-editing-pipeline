import os
import json
import re
import ast

import csv

from analysis.participant_data import Participant
from analysis.participant_data import PARTICIPANT_DATABASE, EDIT_TYPES
from analysis.analysis_helpers import combine_dicts

from video_host.processor import get_alternative_transcript

from backend.helpers import Timecode

from analysis.plotter import timeline_plot, pie_plot, bar_plot

from evaluation.sentence_embedder import get_cosine_similarity_score

def get_simple_numbers(
    participants_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
):
    analysis_per_video = {
        "video-1": {},
        "video-2": {},
    }
    data_points = []

    sorted_participants = []
    for i, email in enumerate(PARTICIPANT_DATABASE):
        sorted_participants.append({
            "email": email,
            "id": PARTICIPANT_DATABASE[email]["id"],
        })

    sorted_participants.sort(key=lambda x: x["id"])

    for s_p in sorted_participants:
        email = s_p["email"]
        participant = Participant(email, **PARTICIPANT_DATABASE[email])
        if participant.exists():
            if participant.study_id != 2 or participant.order not in participants_of_interest:
                continue
            results = {
                "email": participant.email,
                "id": participant.id,
                "video": participant.video_name(),
                "total_logs": len(participant.logs()),
                "count_unique_msgs": len(participant.get_unique_msgs()),
                "count_unique_msg_types": len(participant.get_unqiue_msg_types()),
                "processing_logs": participant.count_processing_logs(),
                "decision_logs": participant.count_decision(),
                "task_summary": participant.get_task_summary(participant.video_id()),
            }
            for edit_type in EDIT_TYPES:
                if edit_type not in results["task_summary"]["edit_types_count"]:
                    results["task_summary"]["edit_types_count"][edit_type] = {
                        "count_intents": 0,
                        "count_edits": 0,
                        "count_history_points": 0,
                        "count_edits_from_suggestion": 0,
                        "edit_type": edit_type,
                    }
            data_point = {
                "id": participant.id,
                "video_id": participant.video_id(),
                "processing_requests": results["processing_logs"]["total"],
                "processing_requests_from_scratch": results["processing_logs"]["count_from_scratch"],
                "processing_requests_add_more": results["processing_logs"]["count_add_more"],
                "processing_requests_suggested": results["processing_logs"]["total_count_suggested"],
                "processing_requests_iterations": results["processing_logs"]["total_count_iterations"],
                "processing_requests_iterations_list": results["processing_logs"]["iterations_list"].copy(),
                "processing_requests_has_text": results["processing_logs"]["count_has_text"],
                "processing_requests_has_sketch": results["processing_logs"]["count_has_sketch"],
                "processing_requests_has_text_and_sketch": results["processing_logs"]["count_has_text_and_sketch"],
                "decisions_accept": results["decision_logs"]["accept"],
                "decisions_reject": results["decision_logs"]["reject"],
                "tasks_intents": results["task_summary"]["count_intents"],
                "tasks_edits": results["task_summary"]["count_edits"],
                "tasks_history_points": results["task_summary"]["count_history_points"],
                "tasks_edit_from_suggestion": results["task_summary"]["count_edits_from_suggestion"],
                "tasks_edit_types_text_intents": results["task_summary"]["edit_types_count"]["text"]["count_intents"],
                "tasks_edit_types_text_edits": results["task_summary"]["edit_types_count"]["text"]["count_edits"],
                "tasks_edit_types_text_from_suggestion": results["task_summary"]["edit_types_count"]["text"]["count_edits_from_suggestion"],
                "tasks_edit_types_image_intents": results["task_summary"]["edit_types_count"]["image"]["count_intents"],
                "tasks_edit_types_image_edits": results["task_summary"]["edit_types_count"]["image"]["count_edits"],
                "tasks_edit_types_image_from_suggestion": results["task_summary"]["edit_types_count"]["image"]["count_edits_from_suggestion"],
                "tasks_edit_types_shape_intents": results["task_summary"]["edit_types_count"]["shape"]["count_intents"],
                "tasks_edit_types_shape_edits": results["task_summary"]["edit_types_count"]["shape"]["count_edits"],
                "tasks_edit_types_shape_from_suggestion": results["task_summary"]["edit_types_count"]["shape"]["count_edits_from_suggestion"],
                "tasks_edit_types_blur_intents": results["task_summary"]["edit_types_count"]["blur"]["count_intents"],
                "tasks_edit_types_blur_edits": results["task_summary"]["edit_types_count"]["blur"]["count_edits"],
                "tasks_edit_types_blur_from_suggestion": results["task_summary"]["edit_types_count"]["blur"]["count_edits_from_suggestion"],
                "tasks_edit_types_zoom_intents": results["task_summary"]["edit_types_count"]["zoom"]["count_intents"],
                "tasks_edit_types_zoom_edits": results["task_summary"]["edit_types_count"]["zoom"]["count_edits"],
                "tasks_edit_types_zoom_from_suggestion": results["task_summary"]["edit_types_count"]["zoom"]["count_edits_from_suggestion"],
                "tasks_edit_types_crop_intents": results["task_summary"]["edit_types_count"]["crop"]["count_intents"],
                "tasks_edit_types_crop_edits": results["task_summary"]["edit_types_count"]["crop"]["count_edits"],
                "tasks_edit_types_crop_from_suggestion": results["task_summary"]["edit_types_count"]["crop"]["count_edits_from_suggestion"],
                "tasks_edit_types_cut_intents": results["task_summary"]["edit_types_count"]["cut"]["count_intents"],
                "tasks_edit_types_cut_edits": results["task_summary"]["edit_types_count"]["cut"]["count_edits"],
                "tasks_edit_types_cut_from_suggestion": results["task_summary"]["edit_types_count"]["cut"]["count_edits_from_suggestion"],
                "tasks_edit_types_empty_intents": results["task_summary"]["edit_types_count"][""]["count_intents"],
                "tasks_edit_types_empty_edits": results["task_summary"]["edit_types_count"][""]["count_edits"],
                "tasks_edit_types_empty_from_suggestion": results["task_summary"]["edit_types_count"][""]["count_edits_from_suggestion"],
            }
            data_points.append(data_point)
            analysis_per_video[participant.video_id()] = combine_dicts(
                analysis_per_video[participant.video_id()],
                results
            )
    
    with open("analysis_results/summary.csv", "w") as f:
        csv.writer(f).writerow(data_points[0].keys())
        for data_point in data_points:
            csv.writer(f).writerow(data_point.values())
    
    for video_id in analysis_per_video:
        print(video_id)
        results = analysis_per_video[video_id]
        print("participant Ids: ", "`" + results["id_str"] + "`")
        
        print("\tprocessed descriptions")
        print("\t\ttotal: ", "`" + results["processing_logs"]["total_str"] + " = " + str(results["processing_logs"]["total"]) + "`")
        print("\t\ttotal iterations:", "`" + results["processing_logs"]["total_count_iterations_str"] + " = " + str(results["processing_logs"]["total_count_iterations"]) + "`")
        print("\t\ttotal suggested edits: ", "`" + results["processing_logs"]["total_count_suggested_str"] + " = " + str(results["processing_logs"]["total_count_suggested"]) + "`")
        print("\t\tprocesing requests (from scratch): ", "`" + results["processing_logs"]["count_from_scratch_str"] + " = " + str(results["processing_logs"]["count_from_scratch"]) + "`")
        print("\t\tprocesing requests (add more): ", "`" + results["processing_logs"]["count_add_more_str"] + " = " + str(results["processing_logs"]["count_add_more"]) + "`")
        
        print("\tdescriptions")
        print("\t\tdescriptions (has text): ", "`" + results["processing_logs"]["count_has_text_str"] + " = " + str(results["processing_logs"]["count_has_text"]) + "`")
        print("\t\tdescriptions (has sketch): ", "`" + results["processing_logs"]["count_has_sketch_str"] + " = " + str(results["processing_logs"]["count_has_sketch"]) + "`")
        print("\t\tdescriptions (has text and sketch): ", "`" + results["processing_logs"]["count_has_text_and_sketch_str"] + " = " + str(results["processing_logs"]["count_has_text_and_sketch"]) + "`")

        print("\tdecisions")
        print("\t\tdecisions (accept): ", "`" + results["decision_logs"]["accept_str"] + " = " + str(results["decision_logs"]["accept"]) + "`")
        print("\t\tdecisions (reject): ", "`" + results["decision_logs"]["reject_str"] + " = " + str(results["decision_logs"]["reject"]) + "`")

        print("\ttasks")
        print("\t\ttotal intents: ", "`" + results["task_summary"]["count_intents_str"] + " = " + str(results["task_summary"]["count_intents"]) + "`")
        print("\t\ttotal edits: ", "`" + results["task_summary"]["count_edits_str"] + " = " + str(results["task_summary"]["count_edits"]) + "`")
        print("\t\ttotal history points: ", "`" + results["task_summary"]["count_history_points_str"] + " = " + str(results["task_summary"]["count_history_points"]) + "`")
        
        print("\t\tedit types")

        print("\t\t\ttext-intents", "`" + results["task_summary"]["edit_types_count"]["text"]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"]["text"]["count_intents"]) + "`")
        print("\t\t\ttext-edits", "`" + results["task_summary"]["edit_types_count"]["text"]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"]["text"]["count_edits"]) + "`")

       
        print("\t\t\timage-intents", "`" + results["task_summary"]["edit_types_count"]["image"]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"]["image"]["count_intents"]) + "`")
        print("\t\t\timage-edits", "`" + results["task_summary"]["edit_types_count"]["image"]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"]["image"]["count_edits"]) + "`")

        print("\t\t\tshape-intents", "`" + results["task_summary"]["edit_types_count"]["shape"]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"]["shape"]["count_intents"]) + "`")
        print("\t\t\tshape-edits", "`" + results["task_summary"]["edit_types_count"]["shape"]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"]["shape"]["count_edits"]) + "`")

        print("\t\t\tblur-intents", "`" + results["task_summary"]["edit_types_count"]["blur"]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"]["blur"]["count_intents"]) + "`")
        print("\t\t\tblur-edits", "`" + results["task_summary"]["edit_types_count"]["blur"]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"]["blur"]["count_edits"]) + "`")

        print("\t\t\tzoom-intents", "`" + results["task_summary"]["edit_types_count"]["zoom"]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"]["zoom"]["count_intents"]) + "`")
        print("\t\t\tzoom-edits", "`" + results["task_summary"]["edit_types_count"]["zoom"]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"]["zoom"]["count_edits"]) + "`")

        print("\t\t\tcrop-intents", "`" + results["task_summary"]["edit_types_count"]["crop"]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"]["crop"]["count_intents"]) + "`")
        print("\t\t\tcrop-edits", "`" + results["task_summary"]["edit_types_count"]["crop"]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"]["crop"]["count_edits"]) + "`")

        print("\t\t\tcut-intents", "`" + results["task_summary"]["edit_types_count"]["cut"]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"]["cut"]["count_intents"]) + "`")
        print("\t\t\tcut-edits", "`" + results["task_summary"]["edit_types_count"]["cut"]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"]["cut"]["count_edits"]) + "`")

        print("\t\t\tempty-intents", "`" + results["task_summary"]["edit_types_count"][""]["count_intents_str"] + " = " + str(results["task_summary"]["edit_types_count"][""]["count_intents"]) + "`")
        print("\t\t\tempty-edits", "`" + results["task_summary"]["edit_types_count"][""]["count_edits_str"] + " = " + str(results["task_summary"]["edit_types_count"][""]["count_edits"]) + "`")

def get_commands(
    participants_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
):
    analysis_per_video = {
        "video-1": {},
        "video-2": {},
    }

    for email in PARTICIPANT_DATABASE:
        participant = Participant(email, **PARTICIPANT_DATABASE[email])
        if participant.exists():
            if participant.study_id != 2 or participant.order not in participants_of_interest:
                continue
            results = {
                "email": participant.email,
                "id": participant.id,
                "video_id": participant.video_id(),
                "processing_logs": participant.get_processing_logs(),
                "task_summary": participant.get_task_summary(participant.video_id()),
                #"process": participant.get_high_level_process(),
            }
            print(json.dumps(results, indent=4))

def get_process(
    participants_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
):
    analysis_per_video = {
        "video-1": [],
        "video-2": [],
    }
    unique_msgs = set()
    for email in PARTICIPANT_DATABASE:
        participant = Participant(email, **PARTICIPANT_DATABASE[email])
        if participant.exists():
            if participant.study_id != 2 or participant.order not in participants_of_interest:
                continue
            unique_msgs.update(participant.get_unique_msgs())
            results = {
                "email": participant.email,
                "id": participant.id,
                "video_id": participant.video_id(),
                "process": participant.get_edit_process(high_level=False),
            }
            if results["video_id"] in analysis_per_video == False:
                analysis_per_video[results["video_id"]] = []
            
            analysis_per_video[results["video_id"]].append(results)

    # unique_msgs = list(unique_msgs)
    # unique_msgs.sort()
    # print(json.dumps(unique_msgs, indent=4))

    all_events = []
    all_durations = []
    for video_id in analysis_per_video:
        video_durations = []
        video_events = []
        print("\t\t", video_id, "-----"*5)
        for participant in analysis_per_video[video_id]:
            print("\t", participant["email"])
            csv_filename = f'analysis_results/P{participant["id"]}_process.csv'
            with open(csv_filename, "w") as f:
                print(f'date, msg', file=f)
                for log in participant["process"]:
                    print(f'{log["date"]}, {log["msg"]}', file=f)
            events = [""]
            start_times = [0]
            durations = []
            prev_time = 0
            for log in participant["process"]:
                if log["msg"] == "admin":
                    continue
                events.append(log["msg"])
                start_times.append(log["date"])
                cur_time = log["time"] / 1000
                durations.append(cur_time - prev_time)
                prev_time = cur_time
            
            sum_duration = sum(durations[1:])
            scaled_durations = [duration / sum_duration * 100 for duration in durations[1:]]
            video_durations.extend(scaled_durations)
            #video_events.extend([f"{event}_{participant['id']}" for event in events[1:]])
            video_events.extend(events[1:])

            timeline_plot_filename = f'analysis_results/P{participant["id"]}_timeline.png'
            timeline_plot(timeline_plot_filename, events[1:], start_times[1:])
            pie_plot_filename = f'analysis_results/P{participant["id"]}_pie.png'
            pie_plot(pie_plot_filename, durations[1:], events[1:])
            bar_plot_filename = f'analysis_results/P{participant["id"]}_bar.png'
            bar_plot(bar_plot_filename, durations[1:], events[1:])

        pie_plot_filename = f'analysis_results/all_{video_id}_pie.png'
        pie_plot(pie_plot_filename, video_durations, video_events)
        bar_plot_filename = f'analysis_results/all_{video_id}_bar.png'
        bar_plot(bar_plot_filename, video_durations, video_events)
        #all_events.extend([f"{event}_{video_id}" for event in video_events])
        all_events.extend(video_events)
        all_durations.extend(video_durations)
    pie_plot_filename = f'analysis_results/all_pie.png'
    pie_plot(pie_plot_filename, all_durations, all_events)
    bar_plot_filename = f'analysis_results/all_bar.png'
    bar_plot(bar_plot_filename, all_durations, all_events)

def sum(arr):
    sum = 0
    for i in arr:
        sum += i
    return sum

def avg_std(arr):
    if len(arr) == 0:
        return 0, 0

    avg = sum(arr) / len(arr)
    std = 0
    for i in arr:
        std += (i - avg) ** 2
    std = (std / len(arr)) ** 0.5
    return avg, std

def calculate_dataset_requests_with_visuals():
    frame = [4, 0, 7, 8, 15, 9, 0, 5, 10, 20]
    asset = [1, 1, 0, 9, 0, 1, 6, 0, 1, 0]
    sum_frame = sum(frame)
    sum_asset = sum(asset)
    print(sum_frame, sum_frame + sum_asset)

def transcribe_study_recordings():
    STUDY_RECORDINGS = "study-recordings"
    for filename in os.listdir(STUDY_RECORDINGS):
        if filename.endswith(".mp3"):
            studyname = filename.split(".")[0]
            print(studyname)
            # extract mp3
            filepath = os.path.join(STUDY_RECORDINGS, filename)
            # transcribe
            transcript = get_alternative_transcript(filepath)
            # save
            transcript_filepath = os.path.join(STUDY_RECORDINGS, studyname + ".txt")

            transcript_dict = []
            for segment in transcript["segments"]:
                transcript_dict.append({
                    "text": segment["text"],
                    "start": Timecode.convert_sec_to_timecode(segment["start"]),
                    "end": Timecode.convert_sec_to_timecode(segment["end"]),
                })
            with open(transcript_filepath, "w") as f:
                for segment in transcript_dict:
                    f.write(segment["start"] + " - " + segment["end"] + ": " + segment["text"] + "\n")

def summarize_evaluation_interpretation():
    MIOU_THRESHOLD = 0.1
    def summarize_array(arr, key):
        if "cosine" in key:
            return
        all_len = len(arr)
        all_avg = -1
        if "spatial_miou" in key:
            all_avg = 0
            for miou in arr:
                if miou > MIOU_THRESHOLD:
                    all_avg += 1
            if all_len > 0:
                all_avg = all_avg / all_len
            print(round(all_avg, 2), "\t", f"{key[len('all_'):]}({all_len})")
        avg, std = avg_std(arr)
        print(round(avg, 2), "\t", "STD: ", round(std, 2), "\t", f"{key[len('all_'):]}({all_len})")

    file_prefix = "results/evaluation_result_"
    combined_results = {}
    for task_id in range(2, 7):
        filename = file_prefix + str([task_id]) + ".json"
        print("task", task_id)
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            for key in result:
                if key.startswith("all_") == False:
                    continue
                if key not in combined_results:
                    combined_results[key] = []
                combined_results[key].extend(result[key])
                summarize_array(result[key], key)
            print("----------"*5)
            print("\n")

    for key in combined_results.keys():
        summarize_array(combined_results[key], key)
        

def summarize_evaluation_parsing():
    def summarize_array(arr, key):
        all_len = len(arr)
        avg, std = avg_std(arr)
        print(round(avg, 2), "\t", "STD: ", round(std, 2), "\t", f"{key}({all_len})")

    file_prefix = "results/evaluation_parsing_"
    results = []
    for task_id in range(2, 7):
        filename = file_prefix + str(task_id) + ".json"
        with open(filename, "r") as f:
            result = json.load(f)
            results.append(result)
    combined_result = {}
    edit_operation_f1 = []
    edit_operation_precision = []
    edit_operation_recall = []

    for task_idx, result in enumerate(results):
        print("task", task_idx + 2)
        print("----------"*5)
        task_result = {}
        task_f1 = []
        task_precision = []
        task_recall = []
        for intent_idx, single in enumerate(result):
            for key in single.keys():
                if key == "editOperations":
                    task_f1.append(single[key]["f1"])
                    task_precision.append(single[key]["precision"])
                    task_recall.append(single[key]["recall"])
                    continue
                if key not in task_result:
                    task_result[key] = []
                final_score = 0
                if len(single[key]['ground_truth']) == 0:
                    # if len(single[key]['prediction']) > 0:
                    #     print(f"WARNING: pipeline detected non-existent reference ({task_idx}-{intent_idx}-{key})", single[key]["prediction"])
                    continue
                else:
                    # if len(single[key]['prediction']) == 0:
                    #     print(f"WARNING: pipeline did not detect reference ({task_idx}-{intent_idx}-{key})", single[key]["ground_truth"])
                    expanded_ground_truth = []
                    for ground_truth in single[key]['ground_truth']:
                        expanded_ground_truth.extend([item.strip() for item in ground_truth.split(", ")])
                    if len(single[key]['prediction']) == 0:
                        print(f"WARNING: pipeline did not detect reference ({task_idx}-{intent_idx}-{key})", single[key]["ground_truth"])
                    else:
                        cosine_scores = get_cosine_similarity_score(
                            expanded_ground_truth,
                            single[key]['prediction'],
                        )
                        for single_cosine_scores in cosine_scores:
                            final_score += max([item.item() for item in single_cosine_scores])
                        final_score = final_score / len(cosine_scores)
                    # score_per_ground_truth = [0 for i in range(len(single[key]['ground_truth']))]
                    # for i, cosine_scores in enumerate(single[key]['cosine_scores']):
                    #     for j, cosine_score in enumerate(cosine_scores):
                    #         score_per_ground_truth[j] = max(score_per_ground_truth[j], cosine_score)
                    # final_score = sum(score_per_ground_truth) / len(score_per_ground_truth)
                    # if len(single[key]['prediction']) == 0:
                    #     print(f"WARNING: pipeline did not detect reference ({task_idx}-{intent_idx}-{key})", single[key]["ground_truth"])
                    # else:
                    #     cosine_scores = get_cosine_similarity_score(
                    #         [", ".join(single[key]['ground_truth'])],
                    #         [", ".join(single[key]['prediction'])],
                    #     )
                    #     final_score = cosine_scores[0][0].item()
                task_result[key].append(final_score)  
        
        summarize_array(task_f1, "editOperations_f1")
        summarize_array(task_precision, "editOperations_precision")
        summarize_array(task_recall, "editOperations_recall")
        for key in task_result.keys():
            summarize_array(task_result[key], key)
        print("----------"*5)
        print("\n")
        for key in task_result.keys():
            if key not in combined_result:
                combined_result[key] = []
            combined_result[key].extend(task_result[key])
        edit_operation_f1.extend(task_f1)
        edit_operation_precision.extend(task_precision)
        edit_operation_recall.extend(task_recall)

    
    if len(edit_operation_f1) > 0:
        summarize_array(edit_operation_f1, "editOperations_f1")
    if len(edit_operation_precision) > 0:
        summarize_array(edit_operation_precision, "editOperations_precision")
    if len(edit_operation_recall) > 0:
        summarize_array(edit_operation_recall, "editOperations_recall")

    for key in combined_result.keys():
        summarize_array(combined_result[key], key)

def summarize_evaluation_spatial():
    MIOU_THRESHOLD = 0.1
    def summarize_array(arr, key):
        all_len = len(arr)
        all_avg = 0
        for miou in arr:
            if miou > MIOU_THRESHOLD:
                all_avg += 1
        if all_len > 0:
            all_avg = all_avg / all_len
        print(round(all_avg, 2), "\t", f"{key}({all_len})")
        avg, std = avg_std(arr)
        print(round(avg, 2), "\t", "STD: ", round(std, 2), "\t", f"{key}({all_len})")

    file_prefix = "results/spatial_evaluation_"
    combined_results = {}
    for task_id in range(2, 7):
        filename = file_prefix + str(task_id) + ".json"
        print("task", task_id)
        print("----------"*5)
        with open(filename, "r") as f:
            result = json.load(f)
            for key in result:
                if key not in combined_results:
                    combined_results[key] = []
                combined_results[key].extend(result[key])
                summarize_array(result[key], key)
            print("----------"*5)
            print("\n")

    for key in combined_results.keys():
        summarize_array(combined_results[key], key)

def combine_evaluation_results():
    results_prefix = "results/evaluation_result_"
    logs_prefix = "results/evaluation_all_"
    combined_results = []

    LINE_PREFIXES = [
        "!!!input!!!:",
        "!!!prediction!!!:",
        "!!!ground_truth!!!:",
    ]

    for task_id in range(2, 7):
        results_filename = results_prefix + str([task_id]) + ".json"
        logs_filename = logs_prefix + str(task_id) + ".txt"
        result = {}
        inputs = []
        predictions = []
        ground_truths = []
        with open(results_filename, "r") as f:
            result = json.load(f)
        with open(logs_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                for line_prefix in LINE_PREFIXES:
                    if line.startswith(line_prefix) == True:
                        main_part = line[len(line_prefix):].strip()
                        json_object = ast.literal_eval(main_part)
                        if line_prefix == LINE_PREFIXES[0]:
                            inputs.append(json_object)
                        elif line_prefix == LINE_PREFIXES[1]:
                            predictions.append(json_object)
                        elif line_prefix == LINE_PREFIXES[2]:
                            ground_truths.append(json_object)
        data_points = []
        for i, (input, prediction, ground_truth) in enumerate(zip(inputs, predictions, ground_truths)):
            cur_result = {}
            for key in result:
                if key.startswith("all_") == False or "cosine" in key:
                    continue
                print(key, result[key], i)
                cur_result[key[len("all_"):]] = result[key][i]
            data_points.append({
                "input": input,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "result": cur_result,
            }) 
        combined_results.extend(data_points)
    with open("results/evaluation_combined.json", "w") as f:
        json.dump(combined_results, f, indent=4)

def summarize_evaluation():
    summarize_evaluation_parsing()
    print("!!!!!!!!!!!"*5)
    summarize_evaluation_interpretation()
    print("!!!!!!!!!!!"*5)
    summarize_evaluation_spatial()

def main():
    # summarize_evaluation_parsing()
    # combine_evaluation_results()
    # get_simple_numbers()
    # get_commands()
    get_process()

if __name__ == '__main__':
    main()