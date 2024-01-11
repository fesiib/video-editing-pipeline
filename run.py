import argparse
import os
import sys
import time
import json

from LangChainPipeline import LangChainPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--text",
        type=str,
        required=True,
        help="The text description of the edit")
    parser.add_argument(
        "-s", "--sketch", 
        type=str,
        default=None,
        help="The sketch rectangle description of the edit"
    )
    parser.add_argument(
        "-sf", "--sketchTimestamp",
        type=int,
        default=-1,
        help="The timestamp of the sketch rectangle (in seconds)"
    )
    parser.add_argument(
        "-v", "--videoId",
        type=str,
        required=True,
        help="The video-id of the video"
    )
    parser.add_argument(
        "-vd", "--videoDuration",
        type=int,
        required=True,
        help="The duration of the video (in seconds)"
    )
    args = parser.parse_args()
    return args

def sketchToSketchRectangles(sketch):
    sketchRectangles = []
    if sketch is None:
        return sketchRectangles
    rectangle = sketch.split(",")
    rectangle = [int(x.strip()) for x in rectangle]
    sketchRectangles.append({
        "x": rectangle[0],
        "y": rectangle[1],
        "width": rectangle[2],
        "height": rectangle[3],
        "stroke": "red",
        "strokeWidth": 2,
        "lineCap": "round",
        "lineJoin": "round"
    })
    return sketchRectangles

def format_output(edit_response):
    edits = []
    for edit in edit_response["edits"]:
        edits.append({
            "start": edit["temporalParameters"]["start"],
            "finish": edit["temporalParameters"]["finish"],
            "temporal_reasoning": edit["temporalParameters"]["info"],
            "temporal_source": edit["temporalParameters"]["source"],
            "spatial": edit["spatialParameters"],
            "spatial_reasoning": edit["spatialParameters"]["info"],
            "spatial_source": edit["spatialParameters"]["source"],
            "edit_parameters": {
                "text": edit["textParameters"],
                "image": edit["imageParameters"],
                "shape": edit["shapeParameters"],
                "blur": edit["blurParameters"],
                "cut": edit["cutParameters"],
                "crop": edit["cropParameters"],
                "zoom": edit["zoomParameters"],
            },
        })
    return {
        "parsing_results": {
            **edit_response["requestParameters"]["relevantText"],
            "parameters": edit_response["requestParameters"]["parameters"],
        },
        "edits": edits,
        "edit_operations": edit_response["requestParameters"]["editOperations"],
    }

def main(args):
    text = args.text
    sketch = args.sketch
    sketchTimestamp = args.sketchTimestamp
    videoId = args.videoId
    videoDuration = args.videoDuration
    
    edit_request = {
        "videoId": videoId,
        "projectMetadata": {
            "totalIntentCnt": 1,
            "duration": videoDuration,
            "projectId": videoId,
            "height": 480,
            "fps": 25,
            "width": 854,
            "trackCnt": 1,
            "title": "test"
        },
        "curPlayPosition": 0,
        "edits": [],
        "segmentOfInterest": {
            "start": 0,
            "finish": videoDuration,
        },
        "skippedSegments": [],
        "requestParameters": {
            "processingMode": "from-scratch",
            "hasText": True,
            "hasSketch": sketch is not None,
            "text": text,
            "sketchRectangles": sketchToSketchRectangles(sketch),
            "sketchFrameTimestamp": sketchTimestamp,
            "editOperation": ""
        },
    }
    pipeline = LangChainPipeline(temperature=0.7, verbose=False)
    pipeline.set_video(edit_request["videoId"], 10)
    edit_response = pipeline.process_request_indexed(
        edit_request
    )
    
    output = format_output(edit_response)

    with open("output.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
