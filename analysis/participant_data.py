import json
import os
import datetime

from analysis.analysis_constants import (
    REPLACE_LIST, USER_DATA_PATH, EDIT_TYPES,
    MSG_TYPES, HIGH_LEVEL_PROCESS, EDITING_PROCESS,
    PARTICIPANT_DATABASE, EDIT_TYPES,
    NAVIGATION_LOGS,
)

class Participant:
    def __init__(self, file_ref, is_baseline, id, order, study, videos, baseline):
        self.file_ref = file_ref
        self.id = id
        self.order = order
        self.study_id = study
        self.videos = videos
        self.baseline = baseline
        self.json_files = []
        self.found_file = False
        self.is_baseline = is_baseline
        for filename in os.listdir(USER_DATA_PATH):
            if filename.endswith(".json"):
                if file_ref in filename:
                    self.json_files.append(os.path.join(USER_DATA_PATH, filename))

        self.data = {}
        if len(self.json_files) == 0:
            self.found_file = False
        else:
            self.found_file = True
            for json_file in self.json_files:
                with open(json_file, "r") as f:
                    cur_data = json.load(f)
                    log_length = len(cur_data.get("logs", []))
                    cur_log_length = len(self.data.get("logs", []))
                    if log_length > cur_log_length:
                        # print("!!!", json_file, log_length)
                        self.data = cur_data

            self.adjust_data()

    def exists(self):
        return self.found_file
    
    def video_id(self):
        # check the baseline first
        if self.is_baseline:
            return self.baseline
        for video in self.videos:
            if video != self.baseline:
                return video
        return None
    
    def study_order(self):
        for i, video in enumerate(self.videos):
            if video == self.video_id():
                return i + 1
    
    def video_name(self):
        if self.video_id() == "video-1":
            return "cooking"
        elif self.video_id() == "video-2":
            return "talking-head"
        return "unknown"

    ### user logs
    def logs(self):
        return self.data.get("logs", [])
    
    def get_specific_logs(self, msg):
        logs = self.logs()
        processing_logs = []
        for log in logs:
            if log["videoId"] != self.video_id():
                continue
            if log["msg"] == msg:
                processing_logs.append(log)
        return processing_logs

    def get_unique_msgs(self):
        logs = self.logs()
        unique_msgs = []
        for log in logs:
            if log["videoId"] != self.video_id():
                continue
            if log["msg"] not in unique_msgs:
                unique_msgs.append(log["msg"])
        return unique_msgs
    
    def get_unqiue_msg_types(self):
        unique_msgs = self.get_unique_msgs()
        unique_msg_types = {}
        for msg in unique_msgs:
            msg_type = msg[0:4]
            if msg_type not in unique_msg_types:
                unique_msg_types[msg_type] = []
            unique_msg_types[msg_type].append(msg)
        return unique_msg_types
    
    def adjust_data(self, task_idx = 0):
        logs = self.logs()
        new_logs = []
        for log in logs:
            if self.id == 22:
                ## skip logs after 12:00:00
                date = datetime.datetime.fromtimestamp(log["time"] / 1000)
                if date.hour >= 12:
                    continue
            if self.id == 6:
                date = datetime.datetime.fromtimestamp(log["time"] / 1000)
                if date.hour >= 18 and date.minute >= 30:
                    continue
            if self.id == 14:
                date = datetime.datetime.fromtimestamp(log["time"] / 1000)
                if date.month < 10 or date.day < 1 or date.day > 3 or date.hour < 15 or date.hour > 17:
                    continue
            if self.id == 25:
                date = datetime.datetime.fromtimestamp(log["time"] / 1000)
                if date.minute >= 30:
                    continue
            if self.id == 27:
                date = datetime.datetime.fromtimestamp(log["time"] / 1000)
                if date.month != 12 or date.day != 20 or date.hour != 12:
                    continue
            if self.id == 28:
                date = datetime.datetime.fromtimestamp(log["time"] / 1000)
                if date.minute > 8 and date.hour >= 12:
                    log["time"] = log["time"] - 1000*(20*60+40 - 8*60)
            if self.id == 29:
                date = datetime.datetime.fromtimestamp(log["time"] / 1000)
                if date.month != 12 or date.day != 20 or date.hour != 15:
                    continue
            if log["msg"] in REPLACE_LIST:
                if REPLACE_LIST[log["msg"]] == None:
                    continue
                else:
                    log["msg"] = REPLACE_LIST[log["msg"]]
            new_logs.append(log)
        self.data["logs"] = new_logs

    def count_msg(self, msg):
        logs = self.logs()
        count = 0
        for log in logs:
            if log["videoId"] != self.video_id():
                continue
            if log["msg"] == msg:
                count += 1
        return count
    
    def count_msg_type(self, msg_type):
        logs = self.logs()
        count = 0
        for log in logs:
            if log["videoId"] != self.video_id():
                continue
            if log["msg"].startswith(msg_type):
                count += 1
        return count
    
    def get_processing_logs(self):
        processing_logs = self.get_specific_logs("processingComplete")
        formatted_logs = []
        for log in processing_logs:
            date = datetime.datetime.fromtimestamp(log["time"] / 1000)
            formatted_date = date.strftime("%H:%M:%S")
            formatted_logs.append({
                "intent_id": log["data"]["intentId"],
                "text": log["data"]["text"],
                "sketch": log["data"]["sketch"],
                "mode": log["data"]["mode"],
                "explanation": json.dumps(log["data"]["explanation"], separators=(',', ': ')),
                "formatted_time": formatted_date,
                "time": log["time"],
            })
        formatted_logs.sort(key=lambda x: x["time"])
        return formatted_logs

    def count_processing_logs(self):
        PROCESSING_COMPLETE_MSG = "processingComplete"
        logs = self.logs()
        count = 0
        count_iterations = 0
        count_processings = {}

        combined_data = []
        for log in logs:
            if log["videoId"] != self.video_id():
                continue
            if log["msg"] == PROCESSING_COMPLETE_MSG:
                data = log["data"]
                if data["intentId"] not in count_processings.keys():
                    count_processings[data["intentId"]] = 0
                count_processings[data["intentId"]] += 1
                combined_data.append(data)
        
        iterations_list = [num - 1 for num in count_processings.values()]
        count_iterations = sum(iterations_list)
        count = sum(count_processings.values())

        processed_data = []
        total_count_edits = 0
        count_from_scratch = 0
        count_add_more = 0
        count_has_text = 0
        count_has_sketch = 0
        count_has_text_and_sketch = 0
        for data in combined_data:
            point = {
                "text": data["text"],
                "hasSketch": len(data["sketch"]) > 0,
                "count_edits": len(data["edits"]),
                "start": data["start"],
                "finish": data["finish"],
                "mode": data["mode"],
            }
            processed_data.append(point)
            total_count_edits += len(data["edits"])
            if data["mode"] == "from-scratch":
                count_from_scratch += 1
            elif data["mode"] == "add-more":
                count_add_more += 1
            
            if len(data["text"]) > 0:
                count_has_text += 1
            if len(data["sketch"]) > 0:
                count_has_sketch += 1
            if len(data["text"]) > 0 and len(data["sketch"]) > 0:
                count_has_text_and_sketch += 1
        return {
            "total": count,
            "total_count_suggested": total_count_edits,
            "total_count_iterations": count_iterations,
            "iterations_list": iterations_list,
            "count_from_scratch": count_from_scratch,
            "count_add_more": count_add_more,
            "count_has_text": count_has_text,
            "count_has_sketch": count_has_sketch,
            "count_has_text_and_sketch": count_has_text_and_sketch,
        }


    def count_decision(self):
        REJECT_MSG = "timelineDecisionReject"
        ACCEPT_MSG = "timelineDecisionAccept"
        logs = self.logs()
        reject_count = 0
        accept_count = 0
        for log in logs:
            if log["videoId"] != self.video_id():
                continue
            if log["msg"] == REJECT_MSG:
                reject_count += 1
            elif log["msg"] == ACCEPT_MSG:
                accept_count += 1
        return {
            "reject": reject_count,
            "accept": accept_count,
        }


    ### user tasks
    def tasks(self):
        return self.data.get("tasks", [])
    
    def get_task(self, task_id):
        tasks = self.tasks()
        for task in tasks:
            if task["projectMetadata"]["projectId"] == task_id:
                return task
        return None

    def get_all_intents(self):
        return self.data.get("intents", [])

    def get_intent(self, intent_id):
        all_intents = self.get_all_intents()
        for intent in all_intents:
            if intent["id"] == intent_id:
                return intent

    def get_task_summary(self, task_id):
        accepted_suggestion_ids = []
        logs = self.logs()
        for log in logs:
            if log["msg"] == "timelineDecisionAccept" and log["videoId"] == self.video_id():
                data = log["data"]
                if "addedEdits" in data:
                    accepted_suggestion_ids.extend(data["addedEdits"])

        task = self.get_task(task_id)
        intents = task["intents"]
        
        count_intents = len(intents)
        count_edits = 0
        count_history_points = 0
        count_edits_from_suggestion = 0
        edit_types_count = {}

        intents_data = []
        for intent_id in intents:
            intent = self.get_intent(intent_id)
            intents_data.append({
                "intent_id": intent_id,
                "edit_type": intent["editOperationKey"],
                "count_edits": len(intent["activeEdits"]),
                "history_points": len(intent["history"]),
                "count_suggestions": len(intent["suggestedEdits"]),
            })
            count_edits += len(intent["activeEdits"])
            count_history_points += len(intent["history"]) - 1
            
            edits_from_suggested = []
            for edit_id in intent["activeEdits"]:
                if edit_id in accepted_suggestion_ids:
                    edits_from_suggested.append(edit_id)

            count_edits_from_suggestion += len(edits_from_suggested)

            edit_type = intent["editOperationKey"]
            if edit_type not in edit_types_count:
                edit_types_count[edit_type] = {
                    "count_intents": 0,
                    "count_edits": 0,
                    "count_history_points": 0,
                    "count_edits_from_suggestion": 0,
                    "edit_type": edit_type,
                }
            edit_types_count[edit_type]["count_intents"] += 1
            edit_types_count[edit_type]["count_edits"] += len(intent["activeEdits"])
            edit_types_count[edit_type]["count_history_points"] += len(intent["history"]) - 1
            edit_types_count[edit_type]["count_edits_from_suggestion"] += len(edits_from_suggested)

        return {
            "count_intents": count_intents,
            "count_edits": count_edits,
            "count_history_points": count_history_points,
            "count_edits_from_suggestion": count_edits_from_suggestion,
            "edit_types_count": edit_types_count,
            #"intents": intents_data,
        }
    
    def get_edit_process(self, based_on="low_level"):
        logs = self.logs()
        process = []
        for log in logs:
            if log["videoId"] != self.video_id():
                continue
            process.append({
                "msg": log["msg"].strip(),
                "data": log["data"],
                "time": log["time"],
            })
        process.sort(key=lambda x: x["time"])
        
        # combine the same logs near each other
        for log in process:
            if based_on == "navigation":
                if log["msg"] in EDITING_PROCESS["admin"]:
                    log["msg"] = "admin"
                elif log["msg"] in NAVIGATION_LOGS:
                    log["msg"] = "navigation"
                else:
                    log["msg"] = "not_navigation"
                continue
            if based_on.endswith("_level"):
                found_edit_types = []
                for edit_type in EDITING_PROCESS.keys():
                    if log["msg"] in EDITING_PROCESS[edit_type]:
                        found_edit_types.append(edit_type)
                if len(found_edit_types) == 0:
                    log["msg"] = "unknown"
                else:
                    found_edit_type = ""
                    if len(found_edit_types) > 1:
                        if "suggested" in log["data"] and log["data"]["suggested"] == True:
                            found_edit_type = "examine"
                        else:
                            found_edit_type = "manual"
                        #print("!!!", log["msg"], found_edit_types)
                    else:
                        found_edit_type = found_edit_types[0]
                    if based_on == "high_level":
                        log["msg"] = HIGH_LEVEL_PROCESS[found_edit_type]
                    else:
                        log["msg"] = found_edit_type
            else:
                log["msg"] = log["msg"].strip()
        
        # filter out anything or unknown
        new_process = []
        for log in process:
            if log["msg"] == "anything" or log["msg"] == "unknown":
               continue
            new_process.append(log)
        process = new_process

        combined_process = []
        for log in process:
            if len(combined_process) == 0:
                combined_process.append(log)
                continue
            last_log = combined_process[-1]
            if last_log["msg"] == log["msg"]:
                last_log["time"] = log["time"]
            else:
                combined_process.append(log)

        # add duration
        for log in combined_process:
            log["duration"] = 0
        for i in range(1, len(combined_process)):
            log = combined_process[i]
            prev_log = combined_process[i - 1]
            log["duration"] = log["time"] - prev_log["time"]

        formatted_process = []
        for log in combined_process:
            date = datetime.datetime.fromtimestamp(log["time"] / 1000)
            duration = datetime.datetime.fromtimestamp(log["duration"] / 1000)
            formatted_date = date.strftime("%H:%M:%S")
            formatted_duration = duration.strftime("%M:%S")

            formatted_process.append({
                "msg": log["msg"],
                "date": formatted_date,
                "duration": formatted_duration,
                "time": log["time"],
            })
        return formatted_process