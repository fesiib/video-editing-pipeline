import argparse
import os
import sys
import time
import json

from LangChainPipeline import LangChainPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", type=str, default="add text whenever chicken is mentioned")
    parser.add_argument("-s", "--sketch", type=str, default="0,0,854,480")
    parser.add_argument("-sf", "--sketchFrame", type=int, default=10)
    parser.add_argument("-v", "--videoId", type=str, default="kdN41iYTg3U")
    parser.add_argument("-vd", "--videoDuration", type=int, default=100)
    args = parser.parse_args()
    return args

def sketchToSketchRectangles(sketch):
    sketchRectangles = []
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
        })
    return {
        "edits": edits,
        "edit_operations": edit["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
    }

def main(args):
    text = args.text
    sketch = args.sketch
    sketchFrame = args.sketchFrame
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
            "hasSketch": True,
            "text": text,
            "sketchRectangles": sketchToSketchRectangles(sketch),
            "sketchFrameTimestamp": sketchFrame,
            "editOperation": ""
        },
    }
    pipeline = LangChainPipeline(verbose=True)
    pipeline.set_video(edit_request["videoId"], 10)
    edit_response = pipeline.process_request_indexed(
        edit_request
    )
    
    output = format_output(edit_response)
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)
