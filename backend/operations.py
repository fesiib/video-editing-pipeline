from helpers import *

textParameters = {
    "content": "HELLO",
    "style": {
        "fill": "#000000",
        "fontSize": 50,
        "fontFamily": "Arial",
        "align": "center",
        "verticalAlign": "middle"
    },
    "background": {
        "fill": "#ffffff",
        "alpha": 1
    }
}
imageParameters = {
    "source": "/placeholder.jpg"
}

shapeParameters = {
    "type": "rectangle",
    "background": {
        "fill": "#ffffff",
        "alpha": 1
    },
    "stroke": {
        "width": 2,
        "fill": "#000000",
        "alpha": 1
    }
}

zoomParameters = {
    "zoomDurationStart": 0,
    "zoomDurationEnd": 0
}

blurParameters = {
    "blur": 6
}

cutParameters = {}

cropParameters = {
    "x": 0,
    "y": 0,
    "width": 0,
    "height": 0,
    "cropX": 0,
    "cropY": 0,
    "cropWidth": 0,
    "cropHeight": 0
}
parseData = {
    "projectId": "test",
    "edits": [{
        "textParameters": textParameters,
        "imageParameters": imageParameters,
        "shapeParameters": shapeParameters,
        "zoomParameters": zoomParameters,
        "cropParameters": cropParameters,
        "cutParameters": cutParameters,
        "blurParameters": blurParameters,
        "spatialParameters": {
            "x": 0,
            "y": 0,
            "width": 200,
            "height": 200,
            "rotation": 0
        },
        "temporalParameters": {
            "start": 0,
            "finish": 10,
            "duration": 10
        }
    }],
    "requestParameters": {
        "considerEdits": True,
        "hasText": True,
        "hasSketch": True,
        "editOperation": "text"
    }
}
edit_instance = {
        "textParameters": textParameters,
        "imageParameters": imageParameters,
        "shapeParameters": shapeParameters,
        "zoomParameters": zoomParameters,
        "cropParameters": cropParameters,
        "cutParameters": cutParameters,
        "blurParameters": blurParameters,
        "spatialParameters": {
            "x": 0,
            "y": 0,
            "width": 200,
            "height": 200,
            "rotation": 0
        },
        "temporalParameters": {
            "start": 0,
            "finish": 10,
            "duration": 10
        }
    }

def get_timecoded_edit_instance(interval)
    start = interval["start"].convert_timecode_to_sec()
    end = interval["end"].convert_timecode_to_sec()
    duration = end - start
    edit_instance["temporalParameters"]["start"] = start
    edit_instance["temporalParameters"]["end"] = end
    edit_instance["temporalParameters"]["duration"] = duration
    return edit_instance

