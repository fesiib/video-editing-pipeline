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

def map_operation_to_parameters(operation):
    editOperations = ["text","image","shape","cut","crop","zoom","blur"]
    if operation == "text":
    elif operation == "image":
    # elif operation == "shape":
    # elif operation == "cut":
    # elif operation == "crop":
    # elif operation == "zoom":
    # elif operation == "blur":
    # else:
    #     return "Invalid operation"