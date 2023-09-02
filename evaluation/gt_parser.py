import os
from typing import Any
import pandas as pd
import numpy as np
import json

COLUMN_PARTICIPANT_ID = "Participant"
COLUMN_TASK_ID = "Task"
COLUMN_INTENT_ID = "Intent"
COLUMN_DESCRIPTION = "Notes"
COLUMN_TIMESTAMP = "Timestamp"
COLUMN_TEMPORAL_TEXT = "Temporal"
COLUMN_SPATIAL_TEXT = "Spatial"
COLUMN_EDIT_TEXT = "Edit"
COLUMN_EXTRA_PARAMS = "Extra params"
COLUMN_TEMPORAL = "Temporal Disambiguation"
COLUMN_SPATIAL = "Spatial Disambiguation"

UNNAMED_HEADER = "Unnamed"

EDIT_OPERATION_TEXT = "adding text"
EDIT_OPERATION_SHAPE = "adding shape"
EDIT_OPERATION_IMAGE = "adding image"
EDIT_OPERATION_BLUR = "blurring"
EDIT_OPERATION_CUT = "cutting/trimming"
EDIT_OPERATION_ZOOM = "zooming"
EDIT_OPERATION_CROP = "cropping"
EDIT_OPERATION_FRAME = "adding frames"

edit_operation_mapping = {
    EDIT_OPERATION_TEXT: "text",
    EDIT_OPERATION_SHAPE: "shape",
    EDIT_OPERATION_IMAGE: "image",
    EDIT_OPERATION_BLUR: "blur",
    EDIT_OPERATION_CUT: "cut",
    EDIT_OPERATION_ZOOM: "zoom",
    EDIT_OPERATION_CROP: "crop",
    EDIT_OPERATION_FRAME: "frame",
}

column_mapping = {
    COLUMN_PARTICIPANT_ID: "participant_id",
    COLUMN_TASK_ID: "task_id",
    COLUMN_INTENT_ID: "intent_id",
    COLUMN_DESCRIPTION: "description",
    COLUMN_TEMPORAL_TEXT: "temporal_text",
    COLUMN_SPATIAL_TEXT: "spatial_text",
    COLUMN_EDIT_TEXT: "edit_text", # TODO: should be mapped to edit_operations
    COLUMN_EXTRA_PARAMS: "params",
    COLUMN_TEMPORAL: "temporal",
    COLUMN_SPATIAL: "spatial",
}

def csv_parser(csv_file):
    with open(csv_file, 'r') as f:
        df = pd.read_csv(f, sep=',', header=0)
        headers = df.columns.values.tolist()
        lines = df.values.tolist()

        headers = [header.strip() for header in headers]
        return headers, lines

def csv_to_dict(csv_file):
    headers, lines = csv_parser(csv_file)
    # last_valid_header = "Unnamed"
    # for i in range(len(headers)):
    #     header = headers[i]
    #     if len(headers) == 0 or header.strip().startswith("Unnamed") == True:
    #         headers[i] = last_valid_header
    #     else:
    #         last_valid_header = header
    data = []
    for line in lines:
        data.append(dict(zip(headers, line)))
    return data

def is_timestamp(string):
    return string.count('-') == 0

def is_range(string):
    return string.count('-') == 1

def parse_timestamp(string):
    parts = [part for part in reversed(string.split(':'))]
    while len(parts) < 3:
        parts.append('0')
    seconds = float(parts[0])
    minutes = float(parts[1])
    hours = float(parts[2])
    total = seconds + minutes * 60 + hours * 3600
    return total

class DataPoint():
    participant_id = -1
    task_id = -1
    intent_id = -1
    description = ""
    temporal_text = []
    spatial_text = []
    edit_text = []
    extra_params = []
    temporal = []
    spatial = []

    def __init__(self,
        participant_id = -1,
        task_id = -1,
        intent_id = -1,
        description = "",
        temporal_text = [],
        spatial_text = [],
        edit_text = [],
        extra_params = [],
        temporal = [],
        spatial = [],
    ):
        self.participant_id = participant_id
        self.task_id = task_id
        self.intent_id = intent_id
        self.description = description
        self.temporal_text = temporal_text
        self.spatial_text = spatial_text
        self.edit_text = edit_text
        self.extra_params = extra_params
        self.temporal = temporal
        self.spatial = spatial

    def reset(self):
        self.participant_id = -1
        self.task_id = -1
        self.intent_id = -1
        self.description = ""
        self.temporal_text = []
        self.spatial_text = []
        self.edit_text = []
        self.extra_params = []
        self.temporal = []
        self.spatial = []


    def __str__(self):
        return f"{self.participant_id}, {self.task_id}, {self.intent_id}, {self.description}, {self.temporal_text}, {self.spatial_text}, {self.edit_text}, {self.extra_params}, {self.temporal}, {self.spatial}"
    
    def set_attr(self, __name: str, __value: Any) -> None:
        #print("->>", __name, __value)
        if (__name == column_mapping.get(COLUMN_PARTICIPANT_ID)):
            if (isinstance(__value, float) == True
                and pd.isnull(__value) == False
                and str(__value) != "nan"
            ):
                self.participant_id = int(__value)
            elif (isinstance(__value, int) == True):
                self.participant_id = __value
        if (__name == column_mapping.get(COLUMN_TASK_ID)):
            if (isinstance(__value, float) == True
                and pd.isnull(__value) == False
                and str(__value) != "nan"
            ):
                self.task_id = int(__value)
            elif (isinstance(__value, int) == True):
                self.task_id = __value
        if (__name == column_mapping.get(COLUMN_INTENT_ID)):
            if (isinstance(__value, float) == True
                and pd.isnull(__value) == False
                and str(__value) != "nan"
            ):
                self.intent_id = int(__value)
            elif (isinstance(__value, int) == True):
                self.intent_id = __value
        if (__name == column_mapping.get(COLUMN_DESCRIPTION)):
            if (isinstance(__value, str) == True):
                self.description = __value.strip()
        if (__name == column_mapping.get(COLUMN_TEMPORAL_TEXT)):
            if (isinstance(__value, str) == True):
                # parts = __value.split(',')
                # self.temporal_text += [part.strip() for part in parts]
                self.temporal_text.append(__value)
            elif (isinstance(__value, list) == True):
                self.temporal_text += __value
        if (__name == column_mapping.get(COLUMN_SPATIAL_TEXT)):
            if (isinstance(__value, str) == True):
                # parts = __value.split(',')
                # self.spatial_text += [part.strip() for part in parts]
                self.spatial_text.append(__value)
            elif (isinstance(__value, list) == True):
                self.spatial_text += __value
        if (__name == column_mapping.get(COLUMN_EDIT_TEXT)):
            if (isinstance(__value, str) == True):
                parts = __value.lower().split(',')
                parts = [part.strip()[0:10] for part in parts]
                for part in parts:
                    edit_operation = part
                    for operation in edit_operation_mapping:
                        if operation.startswith(part) == True:
                            edit_operation = edit_operation_mapping[operation]
                            break
                    self.edit_text.append(edit_operation)
            elif (isinstance(__value, list) == True):
                self.edit_text += __value
        if (__name == column_mapping.get(COLUMN_EXTRA_PARAMS)):
            if (isinstance(__value, str) == True):
                # parts = __value.split(',')
                # self.extra_params += [part.strip() for part in parts]
                self.extra_params.append(__value)
            elif (isinstance(__value, list) == True):
                self.extra_params += __value
        if (__name == column_mapping.get(COLUMN_TEMPORAL)):
            if (isinstance(__value, str) == True):
                if (is_timestamp(__value)):
                    timestamp = parse_timestamp(__value)
                    self.temporal.append([timestamp - 2, timestamp + 2])
                elif (is_range(__value)):
                    timestamps_str = __value.split('-')
                    timestamp_1 = parse_timestamp(timestamps_str[0])
                    timestamp_2 = parse_timestamp(timestamps_str[1])
                    if (timestamp_1 > timestamp_2):
                        timestamp_1, timestamp_2 = timestamp_2, timestamp_1
                    self.temporal.append([timestamp_1, timestamp_2])
            elif (isinstance(__value, list) == True):
                self.temporal.append(__value)
        if (__name == column_mapping.get(COLUMN_SPATIAL)):
            if (isinstance(__value, str) == True):
                coordinates_str = __value.split(',')
                coordinates = [int(float(coord.strip())) for coord in coordinates_str]
                self.spatial.append(coordinates)
            elif (isinstance(__value, list) == True):
                self.spatial.append(__value)

    def consume_dict(self, dict):
        self.reset()
        last_valid_header = UNNAMED_HEADER
        for key in dict:
            cur_key = key.strip()
            if (cur_key.startswith(UNNAMED_HEADER) == True or cur_key == ""):
                cur_key = last_valid_header
            else:
                last_valid_header = cur_key
            if (cur_key in column_mapping):
                val = dict[key]
                str_val = str(val).strip().lower()
                if str_val == "nan" or str_val == "NaN" or pd.isnull(val) == True or str_val == "":
                    continue
                if isinstance(val, str) == True:
                    val = val.strip()
                attr = column_mapping[cur_key]
                self.set_attr(attr, val)
            # else:
            #     print(f"Warning: {key} is not a valid column name and its value is {dict[key]}")
        self.temporal = sorted(self.temporal, key=lambda x: x[0])
        new_temporal = [[0, 0]]
        for segement in self.temporal:
            if segement[0] <= new_temporal[-1][1]:
                new_temporal[-1][1] = max(segement[1], new_temporal[-1][1])
            else:
                new_temporal.append(segement)
        self.temporal = [segment for segment in new_temporal if segment[1] - segment[0] > 0]

    def evaluate(self, response):
        scores = {
            "temporal_text_score": 0,
            "spatial_text_score": 0,
            "edit_text_score": 0,
            "extra_params_score": 0,
            "temporal_score": 0,
            "spatial_score": 0,
        }   
        return scores
    

def main(args):
    csv_file = args.csv
    data = csv_to_dict(csv_file)
    data_points = []
    for dict in data:
        #print(json.dumps(dict, indent=4))
        data_point = DataPoint()
        data_point.consume_dict(dict)
        data_points.append(data_point)

    with open("./data/parsed_gt_v0.json", 'w') as f:
        f.write(json.dumps([data_point.__dict__ for data_point in data_points], indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv', type=str, default="./data/gt-v0.csv",
                        help='path to csv file')
    args = parser.parse_args()
    main(args)