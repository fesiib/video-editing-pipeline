import os
import json
import re
import ast

import csv

from analysis.participant_data import Participant
from analysis.participant_data import PARTICIPANT_DATABASE, EDIT_TYPES
from analysis.analysis_helpers import combine_dicts, avg_std, sum

from video_host.processor import get_alternative_transcript

from backend.helpers import Timecode

from analysis.plotter import timeline_plot, pie_plot, bar_plot

STUDY_RESULTS_FOLDER = "study-results"

# Transcribe study recordings
STUDY_RECORDINGS_FOLDER = "study-recordings"
STUDY_TRANSCRIPTS_FOLDER = "study-transcripts"

def transcribe_study_recordings():
    for filename in os.listdir(STUDY_RECORDINGS_FOLDER):
        if filename.endswith(".mp3"):
            studyname = filename.split(".")[0]
            print(studyname)
            # extract mp3
            filepath = os.path.join(STUDY_RECORDINGS_FOLDER, filename)
            # transcribe
            transcript = get_alternative_transcript(filepath)
            # save
            transcript_filepath = os.path.join(STUDY_TRANSCRIPTS_FOLDER, studyname + ".txt")

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

def analyze_comparative_study_results(
    participants_of_interest=[],
    study_id=3,
):
    treatment = {}
    control = {}
    sorted_participants = []
    for i, file_ref in enumerate(PARTICIPANT_DATABASE):
        sorted_participants.append({
            "file_ref": file_ref,
            "id": PARTICIPANT_DATABASE[file_ref]["id"],
        })

    sorted_participants.sort(key=lambda x: x["id"])

    # treatment
    for s_p in sorted_participants:
        file_ref = s_p["file_ref"]
        participant = Participant(
            file_ref,
            False,
            **PARTICIPANT_DATABASE[file_ref])
        if participant.exists():
            if participants_of_interest != [] and participant.order not in participants_of_interest:
                continue
            if participant.study_id != study_id:
                continue
            results = {
                "study_order": participant.study_order(),
                "file_ref": participant.file_ref,
                "id": participant.id,
                "video": participant.video_name(),
                "total_logs": len(participant.logs()),
                "count_unique_msgs": len(participant.get_unique_msgs()),
                "count_unique_msg_types": len(participant.get_unqiue_msg_types()),
                "processing_logs": participant.count_processing_logs(),
                "decision_logs": participant.count_decision(),
                "task_summary": participant.get_task_summary(participant.video_id()),
                "process": participant.get_edit_process("navigation"),
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
        treatment[participant.id] = results
    
    # control
    for s_p in sorted_participants:
        file_ref = s_p["file_ref"]
        participant = Participant(
            file_ref,
            True,
            **PARTICIPANT_DATABASE[file_ref])
        if participant.exists():
            if participants_of_interest != [] and participant.order not in participants_of_interest:
                continue
            if participant.study_id != study_id:
                continue
            results = {
                "study_order": participant.study_order(),
                "file_ref": participant.file_ref,
                "id": participant.id,
                "video": participant.video_name(),
                "total_logs": len(participant.logs()),
                "count_unique_msgs": len(participant.get_unique_msgs()),
                "count_unique_msg_types": len(participant.get_unqiue_msg_types()),
                "processing_logs": participant.count_processing_logs(),
                "decision_logs": participant.count_decision(),
                "task_summary": participant.get_task_summary(participant.video_id()),
                "process": participant.get_edit_process("navigation"),
                
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
        control[participant.id] = results    

    # print important info
    for s_p in sorted_participants:
        id = s_p["id"]
        print("participant Id: ", id)
        if id not in treatment or id not in control:
            print("\tNOTHING")
            continue
        print("\tstudy order: ", treatment[id]["study_order"], control[id]["study_order"])
        print("\tprocessing_requests: ", treatment[id]["processing_logs"]["total"], control[id]["processing_logs"]["total"])
        print("\t\ttext: ", treatment[id]["processing_logs"]["count_has_text"], control[id]["processing_logs"]["count_has_text"])
        print("\t\tskecth: ", treatment[id]["processing_logs"]["count_has_sketch"], control[id]["processing_logs"]["count_has_sketch"])
        print("\t\ttotal_suggested: ", treatment[id]["processing_logs"]["total_count_suggested"], control[id]["processing_logs"]["total_count_suggested"])
        print("\t\tdecision_accept: ", treatment[id]["decision_logs"]["accept"], control[id]["decision_logs"]["accept"])
        print("\t\tdecision_reject: ", treatment[id]["decision_logs"]["reject"], control[id]["decision_logs"]["reject"])
        print("\tintents: ", treatment[id]["task_summary"]["count_intents"], control[id]["task_summary"]["count_intents"])
        print("\t\tedits: ", treatment[id]["task_summary"]["count_edits"], control[id]["task_summary"]["count_edits"])
        print("\t\t\tfrom_suggestion: ", treatment[id]["task_summary"]["count_edits_from_suggestion"], control[id]["task_summary"]["count_edits_from_suggestion"])
    
    # save as csv
    with open("{STUDY_RESULTS_FOLDER}/comparative_study_for_starlab.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "participant_id",
            "condition",
            "study_order",
            "video",
            "processing_requests",
            "processing_requests_text",
            "processing_requests_sketch",
            "processing_requests_suggested",
            "processing_requests_decision_accept",
            "processing_requests_decision_reject",
            "intents",
            "edits",
            "edits_from_suggestion",
        ])
        for s_p in sorted_participants:
            id = s_p["id"]
            if id not in treatment or id not in control:
                continue
            writer.writerow([
                id,
                "treatment",
                treatment[id]["study_order"],
                treatment[id]["video"],
                treatment[id]["processing_logs"]["total"],
                treatment[id]["processing_logs"]["count_has_text"],
                treatment[id]["processing_logs"]["count_has_sketch"],
                treatment[id]["processing_logs"]["total_count_suggested"],
                treatment[id]["decision_logs"]["accept"],
                treatment[id]["decision_logs"]["reject"],
                treatment[id]["task_summary"]["count_intents"],
                treatment[id]["task_summary"]["count_edits"],
                treatment[id]["task_summary"]["count_edits_from_suggestion"],
            ])
            writer.writerow([
                id,
                "control",
                control[id]["study_order"],
                control[id]["video"],
                control[id]["processing_logs"]["total"],
                control[id]["processing_logs"]["count_has_text"],
                control[id]["processing_logs"]["count_has_sketch"],
                control[id]["processing_logs"]["total_count_suggested"],
                control[id]["decision_logs"]["accept"],
                control[id]["decision_logs"]["reject"],
                control[id]["task_summary"]["count_intents"],
                control[id]["task_summary"]["count_edits"],
                control[id]["task_summary"]["count_edits_from_suggestion"],
            ])

    results = {
        "treatment": {
            "durations": [],
            "events": [],
            "navigation_durations": [],
            "not_navigation_durations": [],
        },
        "control": {
            "durations": [],
            "events": [],
            "navigation_durations": [],
            "not_navigation_durations": [],
        }
    }
    for s_p in sorted_participants:
        id = s_p["id"]
        if id not in treatment or id not in control:
            continue
        for condition, result in zip(["treatment", "control"], [treatment[id], control[id]]):
            csv_filename = f'{STUDY_RESULTS_FOLDER}/P{id}_{condition}_process.csv'
            with open(csv_filename, "w") as f:
                print(f'date, msg', file=f)
                for log in result["process"]:
                    print(f'{log["date"]}, {log["msg"]}', file=f)
            events = [""]
            start_times = [0]
            durations = []
            prev_time = 0
            for log in result["process"]:
                if log["msg"] == "admin":
                    continue
                events.append(log["msg"])
                start_times.append(log["date"])
                cur_time = log["time"] / 1000
                durations.append(cur_time - prev_time)
                prev_time = cur_time
            
            event_duration = {
                "navigation": 0,
                "not_navigation": 0,
            }
            for event, duration in zip(events[1:], durations[1:]):
                if event not in event_duration:
                    continue
                event_duration[event] += duration

            results[condition]["navigation_durations"].append(event_duration["navigation"])
            results[condition]["not_navigation_durations"].append(event_duration["not_navigation"])
            sum_duration = sum(durations[1:])
            scaled_durations = [duration / sum_duration * 100 for duration in durations[1:]]
            results[condition]["durations"].extend(scaled_durations)
            #video_events.extend([f"{event}_{participant['id']}" for event in events[1:]])
            results[condition]["events"].extend(events[1:])

            timeline_plot_filename = f'{STUDY_RESULTS_FOLDER}/P{id}_{condition}_timeline.png'
            timeline_plot(timeline_plot_filename, events[1:], start_times[1:])
            pie_plot_filename = f'{STUDY_RESULTS_FOLDER}/P{id}_{condition}_pie.png'
            pie_plot(pie_plot_filename, durations[1:], events[1:])
            bar_plot_filename = f'{STUDY_RESULTS_FOLDER}/P{id}_{condition}_bar.png'
            bar_plot(bar_plot_filename, durations[1:], events[1:])
    for condition in ["treatment", "control"]:
        pie_plot_filename = f'{STUDY_RESULTS_FOLDER}/summary_{condition}_pie.png'
        pie_plot(pie_plot_filename, results[condition]["durations"], results[condition]["events"])
        bar_plot_filename = f'{STUDY_RESULTS_FOLDER}/summary_{condition}_bar.png'
        bar_plot(bar_plot_filename, results[condition]["durations"], results[condition]["events"])
        print(condition, "navigation")
        for i in results[condition]["navigation_durations"]:
            print(i / 60)
        print(condition, "not navigation")
        for i in results[condition]["not_navigation_durations"]:
            print(i / 60)

def anayze_study_results_summary(
    participants_of_interest=[],
    study_id=3,
    is_baseline=False,
):
    analysis_per_video = {
        "video-1": {},
        "video-2": {},
    }
    data_points = []

    sorted_participants = []
    for i, file_ref in enumerate(PARTICIPANT_DATABASE):
        sorted_participants.append({
            "file_ref": file_ref,
            "id": PARTICIPANT_DATABASE[file_ref]["id"],
        })

    sorted_participants.sort(key=lambda x: x["id"])

    for s_p in sorted_participants:
        file_ref = s_p["file_ref"]
        participant = Participant(
            file_ref,
            is_baseline,
            **PARTICIPANT_DATABASE[file_ref])
        if participant.exists():
            if participants_of_interest != [] and participant.order not in participants_of_interest:
                continue
            if participant.study_id != study_id:
                continue
            results = {
                "file_ref": participant.file_ref,
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
    
    with open("{STUDY_RESULTS_FOLDER}/summary.csv", "w") as f:
        csv.writer(f).writerow(data_points[0].keys())
        for data_point in data_points:
            csv.writer(f).writerow(data_point.values())

    for video_id in analysis_per_video:
        print(video_id)
        results = analysis_per_video[video_id]
        if "id_str" not in results:
            print("NOTHING")
            continue
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

def get_study_results_commands(
    participants_of_interest=[],
    study_id=3,
    is_baseline=False,
):

    for file_ref in PARTICIPANT_DATABASE:
        participant = Participant(
            file_ref,
            is_baseline,
            **PARTICIPANT_DATABASE[file_ref])
        if participant.exists():
            if participants_of_interest != [] and participant.order not in participants_of_interest:
                continue
            if participant.study_id != study_id:
                continue
            results = {
                "file_ref": participant.file_ref,
                "id": participant.id,
                "video_id": participant.video_id(),
                "processing_logs": participant.get_processing_logs(),
                "task_summary": participant.get_task_summary(participant.video_id()),
            }
            print(json.dumps(results, indent=4))

def analyze_study_results_process(
    participants_of_interest=[],
    study_id=3,
    is_baseline=False,
    process_type="low_level"
):
    analysis_per_video = {
        "video-1": [],
        "video-2": [],
    }
    unique_msgs = set()
    for file_ref in PARTICIPANT_DATABASE:
        participant = Participant(
            file_ref,
            is_baseline,
            **PARTICIPANT_DATABASE[file_ref])
        if participant.exists():
            if participants_of_interest != [] and participant.order not in participants_of_interest:
                continue
            if participant.study_id != study_id:
                continue
            unique_msgs.update(participant.get_unique_msgs())
            results = {
                "file_ref": participant.file_ref,
                "id": participant.id,
                "video_id": participant.video_id(),
                "process": participant.get_edit_process(process_type),
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
            print("\t", participant["file_ref"])
            csv_filename = f'{STUDY_RESULTS_FOLDER}/P{participant["id"]}_process.csv'
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

            timeline_plot_filename = f'{STUDY_RESULTS_FOLDER}/P{participant["id"]}_timeline.png'
            timeline_plot(timeline_plot_filename, events[1:], start_times[1:])
            pie_plot_filename = f'{STUDY_RESULTS_FOLDER}/P{participant["id"]}_pie.png'
            pie_plot(pie_plot_filename, durations[1:], events[1:])
            bar_plot_filename = f'{STUDY_RESULTS_FOLDER}/P{participant["id"]}_bar.png'
            bar_plot(bar_plot_filename, durations[1:], events[1:])

        pie_plot_filename = f'{STUDY_RESULTS_FOLDER}/all_{video_id}_pie.png'
        pie_plot(pie_plot_filename, video_durations, video_events)
        bar_plot_filename = f'{STUDY_RESULTS_FOLDER}/all_{video_id}_bar.png'
        bar_plot(bar_plot_filename, video_durations, video_events)
        #all_events.extend([f"{event}_{video_id}" for event in video_events])
        all_events.extend(video_events)
        all_durations.extend(video_durations)
    pie_plot_filename = f'{STUDY_RESULTS_FOLDER}/all_pie.png'
    pie_plot(pie_plot_filename, all_durations, all_events)
    bar_plot_filename = f'{STUDY_RESULTS_FOLDER}/all_bar.png'
    bar_plot(bar_plot_filename, all_durations, all_events)

def calculate_dataset_requests_with_visuals():
    frame = [4, 0, 7, 8, 15, 9, 0, 5, 10, 20]
    asset = [1, 1, 0, 9, 0, 1, 6, 0, 1, 0]
    sum_frame = sum(frame)
    sum_asset = sum(asset)
    print(sum_frame, sum_frame + sum_asset)

def main():
    anayze_study_results_summary()

if __name__ == '__main__':
    main()