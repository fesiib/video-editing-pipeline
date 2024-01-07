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
            "source": "/placeholder.jpg",
            "searchQuery": "",
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
            },
            "star": {
                "numPoints": 6,
                "innerRadius": 100
            }
        }

        # Zoom, Blur, Cut, and Crop Parameters
        self.zoomParameters = {"zoomDurationStart": 0, "zoomDurationEnd": 0}
        self.blurParameters = {"blur": 6}
        self.cutParameters = {}
        self.cropParameters = {
            "x": 0, "y": 0, "width": 720, "height": 1280,
            "cropX": 0, "cropY": 0, "cropWidth": 200, "cropHeight": 200
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
                "processingMode": "from-scratch",
                "hasText": True,
                "hasSketch": True,
                "editOperation": "text",
                "editOperations": [],
                "parameters": {},
            }
        }

    def get_edit_params(self):
        return self.edit

def get_timecoded_edit_instance(interval, video_shape):
    edit = EditInstance()
    edit_instance = edit.get_edit_params()
    start = interval["start"].convert_timecode_to_sec()
    finish = interval["end"].convert_timecode_to_sec()
    duration = finish - start
    edit_instance["temporalParameters"]["start"] = start
    edit_instance["temporalParameters"]["finish"] = finish
    edit_instance["temporalParameters"]["duration"] = duration
    edit_instance["cropParameters"] = {
        "x": 0, "y": 0, "width": video_shape[1], "height": video_shape[0],
        "cropX": 0, "cropY": 0, "cropWidth": video_shape[1], "cropHeight": video_shape[0]
    }
    return edit_instance

def get_edit_segment(start, finish, explanation, source, offsets, video_shape):
    edit = EditInstance()
    edit_instance = edit.get_edit_params()
    
    duration = finish - start
    edit_instance["temporalParameters"]["start"] = start
    edit_instance["temporalParameters"]["finish"] = finish
    edit_instance["temporalParameters"]["duration"] = duration
    edit_instance["temporalParameters"]["info"] = explanation
    edit_instance["temporalParameters"]["source"] = source
    edit_instance["temporalParameters"]["offsets"] = offsets
    
    edit_instance["spatialParameters"]["info"] = ["Default location"]
    edit_instance["spatialParameters"]["source"] = ["default"]
    edit_instance["spatialParameters"]["offsets"] = [-1]
    
    edit_instance["cropParameters"] = {
        "x": 0, "y": 0, "width": video_shape[1], "height": video_shape[0],
        "cropX": 0, "cropY": 0, "cropWidth": video_shape[1], "cropHeight": video_shape[0]
    }
    return edit_instance
