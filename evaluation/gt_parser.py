import os
from typing import Any
import pandas as pd
import numpy as np
import json

COLUMN_PARTICIPANT_ID = "Participant"
COLUMN_TASK_ID = "Task"
COLUMN_INTENT_ID = "Intent"
COLUMN_DESCRIPTION = "Notes"
COLUMN_SKETCH = "Sketch Coordinates"
COLUMN_SKETCH_TIMESTAMP = "Time of sketch coordinate"
COLUMN_TIMESTAMP = "Timestamp"
COLUMN_TEMPORAL_TEXT = "Temporal"
COLUMN_SPATIAL_TEXT = "Spatial"
COLUMN_EDIT_TEXT = "Edit"

COLUMN_SEGMENTS = "Segments"
COLUMN_PARAMETERS = "Parameters"
COLUMN_EDIT_OPERATION = "Edit Operation"

UNNAMED_HEADER = "Unnamed"

EDIT_OPERATION_TEXT = "adding text"
EDIT_OPERATION_SHAPE = "adding shape"
EDIT_OPERATION_IMAGE = "adding image"
EDIT_OPERATION_BLUR = "blurring"
EDIT_OPERATION_CUT = "cutting/trimming"
EDIT_OPERATION_ZOOM = "zooming"
EDIT_OPERATION_CROP = "cropping"

edit_operation_mapping = {
    EDIT_OPERATION_TEXT: "text",
    EDIT_OPERATION_SHAPE: "shape",
    EDIT_OPERATION_IMAGE: "image",
    EDIT_OPERATION_BLUR: "blur",
    EDIT_OPERATION_CUT: "cut",
    EDIT_OPERATION_ZOOM: "zoom",
    EDIT_OPERATION_CROP: "crop",
}

column_mapping = {
    COLUMN_PARTICIPANT_ID: "participant_id",
    COLUMN_TASK_ID: "task_id",
    COLUMN_INTENT_ID: "intent_id",
    COLUMN_DESCRIPTION: "description",
    COLUMN_SKETCH: "sketch",
    COLUMN_SKETCH_TIMESTAMP: "sketch_timestamp",
    COLUMN_TEMPORAL_TEXT: "temporal_text",
    COLUMN_SPATIAL_TEXT: "spatial_text",
    COLUMN_EDIT_TEXT: "edit_text",
    COLUMN_PARAMETERS: "params_text",
    
    COLUMN_SEGMENTS: "segments",
    COLUMN_EDIT_OPERATION: "edit",
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
    sketch = []
    sketch_timestamp = -1
    temporal_text = []
    spatial_text = []
    edit_text = []
    params_text = []

    temporal = []
    spatial = []
    edit = []

    def __init__(self,
        participant_id = -1,
        task_id = -1,
        intent_id = -1,
        description = "",
        sketch = [],
        sketch_timestamp = -1,
        temporal_text = [],
        spatial_text = [],
        edit_text = [],
        params_text = [],
        temporal = [],
        spatial = [],
        edit = [],
    ):
        self.participant_id = participant_id
        self.task_id = task_id
        self.intent_id = intent_id
        self.description = description
        self.sketch = sketch
        self.sketch_timestamp = sketch_timestamp
        self.temporal_text = temporal_text
        self.spatial_text = spatial_text
        self.edit_text = edit_text
        self.params_text = params_text

        self.temporal = temporal
        self.spatial = spatial
        self.edit = edit

    def reset(self):
        self.participant_id = -1
        self.task_id = -1
        self.intent_id = -1
        self.description = ""
        self.sketch = []
        self.sketch_timestamp = -1
        self.temporal_text = []
        self.spatial_text = []
        self.edit_text = []
        self.params_text = []
        
        self.temporal = []
        self.spatial = []
        self.edit = []


    def __str__(self):
        return f"""
            {self.participant_id}, {self.task_id}, {self.intent_id},\n
            {self.description}, {self.sketch}, {self.sketch_timestamp},\n
            {self.temporal_text}, {self.spatial_text}, {self.edit_text}, {self.params_text},\n
            {self.temporal}, {self.spatial}, {self.edit}, {self.params_text}
        """
    
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
        if (__name == column_mapping.get(COLUMN_SKETCH)):
            if (__value == "x"):
                    self.sketch = []
            elif (isinstance(__value, str) == True):
                sketches = __value.strip().split('\n')
                for sketch in sketches:
                    #coordinates_str = json.loads(sketch)
                    #coordinates = [int(float(coord.strip())) for coord in coordinates_str]
                    coordinates = json.loads(sketch)
                    self.sketch.append({
                        "x": coordinates[0],
                        "y": coordinates[1],
                        "width": coordinates[2],
                        "height": coordinates[3],
                        "rotation": 0,
                    })
            elif (isinstance(__value, list) == True):
                self.sketch += __value
        if (__name == column_mapping.get(COLUMN_SKETCH_TIMESTAMP)):

            if (isinstance(__value, str) == True):
                if (__value == "x"):
                    self.sketch_timestamp = -1
                elif (is_timestamp(__value)):
                    timestamp = parse_timestamp(__value)
                    self.sketch_timestamp = timestamp
            elif (isinstance(__value, float) == True):
                self.sketch_timestamp = __value

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
        
        """ FROM V3"""
        if (__name == column_mapping.get(COLUMN_PARAMETERS)):
            if (isinstance(__value, str) == True):
                # parts = __value.split(',')
                # self.extra_params += [part.strip() for part in parts]
                self.params_text.append(__value)
            elif (isinstance(__value, list) == True):
                self.params_text += __value

        if (__name == column_mapping.get(COLUMN_EDIT_TEXT)):
            if (isinstance(__value, str) == True):
                # parts = __value.split(',')
                # self.spatial_text += [part.strip() for part in parts]
                self.edit_text.append(__value)
            elif (isinstance(__value, list) == True):
                self.edit_text += __value

        if (__name == column_mapping.get(COLUMN_EDIT_OPERATION)):
            if (isinstance(__value, str) == True):
                parts = __value.lower().split(',')
                parts = [part.strip()[0:10] for part in parts]
                for part in parts:
                    edit_operation = part
                    for operation in edit_operation_mapping:
                        if operation.startswith(part) == True:
                            edit_operation = edit_operation_mapping[operation]
                            break
                    self.edit.append(edit_operation)
            elif (isinstance(__value, list) == True):
                self.edit += __value

        if (__name == column_mapping.get(COLUMN_SEGMENTS)):
            if (isinstance(__value, str) == True):
                parts = __value.strip().split(',', 1)
                timestamp_str = parts[0].strip()
                rect_str = None
                if (len(parts) > 1):
                    rect_str = ",".join(parts[1:]).strip()
                if (is_timestamp(timestamp_str)):
                    timestamp = parse_timestamp(timestamp_str)
                    self.temporal.append([max(0, timestamp - 2), timestamp + 2])
                elif (is_range(timestamp_str)):
                    timestamps_str = timestamp_str.split('-')
                    timestamp_1 = parse_timestamp(timestamps_str[0])
                    timestamp_2 = parse_timestamp(timestamps_str[1])
                    if (timestamp_1 > timestamp_2):
                        timestamp_1, timestamp_2 = timestamp_2, timestamp_1
                    self.temporal.append([timestamp_1, timestamp_2])
                
                if (rect_str != None):
                    new_spatial = []
                    coordinates = json.loads(rect_str)
                    if isinstance(coordinates[0], list) == True:
                        for coordinate in coordinates:
                            new_spatial.append({
                                "x": coordinate[0],
                                "y": coordinate[1],
                                "width": coordinate[2],
                                "height": coordinate[3],
                                "rotation": 0,
                            })
                    else:
                        new_spatial.append({
                            "x": coordinates[0],
                            "y": coordinates[1],
                            "width": coordinates[2],
                            "height": coordinates[3],
                            "rotation": 0,
                        })
                    self.spatial.append(new_spatial)
                    #print("Spatial", new_spatial)
                else:
                    self.spatial.append(None)

                # if (is_timestamp(__value)):
                #     timestamp = parse_timestamp(__value)
                #     self.temporal.append([timestamp - 2, timestamp + 2])
                # elif (is_range(__value)):
                #     timestamps_str = __value.split('-')
                #     print(timestamps_str)
                #     timestamp_1 = parse_timestamp(timestamps_str[0])
                #     timestamp_2 = parse_timestamp(timestamps_str[1])
                #     if (timestamp_1 > timestamp_2):
                #         timestamp_1, timestamp_2 = timestamp_2, timestamp_1
                #     self.temporal.append([timestamp_1, timestamp_2])
            elif (isinstance(__value, list) == True):
                self.temporal.append(__value)
                self.spatial.append(None)

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
        new_temporal_spatial = [[0, 0, []]]
        for segment, rects in zip(self.temporal, self.spatial):
            if segment[0] <= new_temporal_spatial[-1][1]:
                new_temporal_spatial[-1][1] = max(segment[1], new_temporal_spatial[-1][1])
                if rects != None:
                    # union_rect = {}
                    # if new_temporal_spatial[-1][2] != None:
                    #     union_rect['x'] = min(rect['x'], new_temporal_spatial[-1][2]['x'])
                    #     union_rect['y'] = min(rect['y'], new_temporal_spatial[-1][2]['y'])
                    #     union_rect['width'] = round(max(rect['x'] + rect['width'], new_temporal_spatial[-1][2]['x'] + new_temporal_spatial[-1][2]['width']) - union_rect['x'], 2)
                    #     union_rect['height'] = round(max(rect['y'] + rect['height'], new_temporal_spatial[-1][2]['y'] + new_temporal_spatial[-1][2]['height']) - union_rect['y'], 2)
                    #     union_rect['rotation'] = 0
                    # else:
                    #     union_rect = rects.copy()
                    # #get union of two rects
                    # print("Warning: competing spatial", rect, new_temporal_spatial[-1][2], union_rect, dict['Participant'], dict['Task'], dict['Intent'])
                    
                    new_temporal_spatial[-1][2].extend(rects.copy())
            else:
                if rects != None:
                    new_temporal_spatial.append(segment.copy() + [rects.copy()])
                else:
                    new_temporal_spatial.append(segment.copy() + [[]])
        self.spatial = []
        self.temporal = []
        for segment_rect in new_temporal_spatial:
            if segment_rect[1] <= segment_rect[0]:
                continue
            self.temporal.append(segment_rect[0:2])
            self.spatial.append(segment_rect[2])

def main(args):
    csv_file = args.csv
    data = csv_to_dict(csv_file)
    data_points = []
    for dict in data:
        #print(json.dumps(dict, indent=4))
        data_point = DataPoint()
        data_point.consume_dict(dict)
        data_points.append(data_point)

    csv_filename = os.path.basename(csv_file).split(".")[0]

    with open("./gt_data/parsed_" + csv_filename + ".json", 'w') as f:
        f.write(json.dumps([data_point.__dict__ for data_point in data_points], indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv', type=str, default="./gt_data/gt-v0.csv",
                        help='path to csv file')
    args = parser.parse_args()
    main(args)