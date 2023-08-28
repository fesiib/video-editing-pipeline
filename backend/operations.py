from .helpers import *
class EditInstance:
    def __init__(self):
        # Text Parameters
        self.textParameters = {
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

        # Image Parameters
        self.imageParameters = {
            "source": "/placeholder.jpg"
        }

        # Shape Parameters
        self.shapeParameters = {
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

        # Zoom, Blur, Cut, and Crop Parameters
        self.zoomParameters = {"zoomDurationStart": 0, "zoomDurationEnd": 0}
        self.blurParameters = {"blur": 6}
        self.cutParameters = {}
        self.cropParameters = {
            "x": 0, "y": 0, "width": 0, "height": 0,
            "cropX": 0, "cropY": 0, "cropWidth": 0, "cropHeight": 0
        }

        # Spatial and Temporal Parameters
        self.spatialParameters = {
            "x": 0, "y": 0, "width": 200, "height": 200, "rotation": 0
        }
        self.temporalParameters = {
            "start": 0, "finish": 10, "duration": 10
        }

        # Edit and Parse Data
        self.edit = {
            "textParameters": self.textParameters,
            "imageParameters": self.imageParameters,
            "shapeParameters": self.shapeParameters,
            "zoomParameters": self.zoomParameters,
            "cropParameters": self.cropParameters,
            "cutParameters": self.cutParameters,
            "blurParameters": self.blurParameters,
            "spatialParameters": self.spatialParameters,
            "temporalParameters": self.temporalParameters
        }

        self.parseData = {
            "projectId": "test",
            "edits": [self.edit],
            "requestParameters": {
                "considerEdits": True,
                "hasText": True,
                "hasSketch": True,
                "editOperation": "text"
            }
        }

    def get_edit_params(self):
        return self.edit

def get_timecoded_edit_instance(interval):
    edit = EditInstance()
    edit_instance = edit.get_edit_params()
    start = interval["start"].convert_timecode_to_sec()
    end = interval["end"].convert_timecode_to_sec()
    duration = end - start
    edit_instance["temporalParameters"]["start"] = start
    edit_instance["temporalParameters"]["end"] = end
    edit_instance["temporalParameters"]["duration"] = duration
    return edit_instance