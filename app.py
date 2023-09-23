import json

from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS

#from backend.intent_parser import *
from backend.pipeline import Pipeline
from LangChainPipeline import LangChainPipeline
from backend.quick_parser import extract_adverbs_of_space, extract_adverbs_of_space_gpt3

from video_host.processor import process_video, get_video_by_filename


app = Flask(__name__)

CORS(app, origins=["http://localhost:3000", "http://localhost:7777", "http://internal.kixlab.org:7777"])

'''
Server side for video editing pipline 
'''
#intent_parser = IntentParser(40, 80)
pipeline = Pipeline(50, 0)
langchain_pipeline = LangChainPipeline()

'''
Input:
    Video Reference: youtube link
    State of intent:
'''
@app.route("/intent", methods=['GET', 'POST'])
def parse_intent():
    edit_request = request.json
    
    #pipeline.reset()
    edit_response = langchain_pipeline.process_request_indexed(edit_request)

    return jsonify(edit_response)

'''
Summary of the edit description
'''
@app.route("/summary", methods=['GET', 'POST'])
def fetch_summary():
    data = request.json

    pipeline.reset()
    response = pipeline.get_summary(data.get("input"))

    return jsonify({"summary": response})

@app.route("/ambiguous", methods=['POST'])
def fetch_ambiguous():
    data = request.json
    input = data.get("input")
    return jsonify({"ambiguousParts": extract_adverbs_of_space(input)})

@app.route("/ambiguous-gpt", methods=['POST'])
def fetch_ambiguous_gpt():
    data = request.json
    input = data.get("input")
    return jsonify({"ambiguousParts": extract_adverbs_of_space_gpt3(input)})

def fail_with(msg):
    return {
        "status": "failed",
        "message": msg,
    }

@app.route("/process_youtube_link", methods=["POST"])
def process_youtube_link():
    decoded = request.data.decode('utf-8')
    request_json = json.loads(decoded)
    video_link = request_json["videoLink"]

    # Extract transcript
    # Extract OCR results per frame
    # Group OCR results based on similarity
    # Map transcript to OCR
    # moment: {
    #   "start": 0,
    #   "finish": 0,
    #   "transcriptStart": 0,
    #   "transcriptFinish": 0,
    #   "title": "text caption",
    # }

    # result = {
    #   "moments": list[moments]
    #   "transcript": list[transcript]
    # }

    transcript, moments, metadata = process_video(video_link)
    filename = f'{metadata["id"]}.mp4'

    responseJSON = {
        "request": {
            "videoLink": video_link,
        },
        "moments": moments,
        "metadata": metadata,
        "transcript": transcript,
        #"source": url_for("display_video", filename=filename),
        "source": str(get_video_by_filename(filename)),
        "status": "success"
    }
    return json.dumps(responseJSON)

@app.route('/display_video/<filename>', methods=["GET"])
def display_video(filename):
    video_path = get_video_by_filename(filename)
    print(filename, video_path)
    return redirect(video_path, code=301)

def test(edit_request):
    pipeline.reset()
    print("!!!response!!!", json.dumps(pipeline.process_request({
        "requestParameters": {
            "text": edit_request,
            "editOperation": "text",
            "processingMode": "from-scratch",
        },
        "edits": [{
            "temporalParameters": {
                "start": "0:00",
                "finish": "5:00",
            }
        }],
    }), indent=1))

def launch_server():
    app.run(host="0.0.0.0", port=7778)

def test_processor(video_link):
    transcript, moments, metadata = process_video(video_link)
    with open("transcript.json", "w") as f:
        json.dump(transcript, f, indent=2)

def test_langchain_pipeline():
    pipeline = LangChainPipeline(verbose=True)
    
    EXAMPLE_REQUEST = {
        "projectId": "test",
        "projectMetadata": {
            "totalIntentCnt": 2,
            "duration": 1218.53678,
            "projectId": "tutorial",
            "height": 480,
            "fps": 25,
            "width": 854,
            "trackCnt": 1,
            "title": "test"
        },
        "curPlayPosition": 100,
        "edits": [],
        "segmentOfInterest": {
            "start": 0,
            "finish": 1218.53678
        },
        "skippedSegments": [],
        "requestParameters": {
            "processingMode": "from-scratch",
            "hasText": False,
            "hasSketch": True,
            "text": "at 1:00, put a transparent star around the laptop",
            "sketchRectangles": [
                # {
                #     "x": 222.14218431771891,
                #     "y": 83.49621553954651,
                #     "width": 231.3279022403259,
                #     "height": 189.60598945438687,
                #     "stroke": "red",
                #     "strokeWidth": 2,
                #     "lineCap": "round",
                #     "lineJoin": "round"
                # },
                # {
                #     "x": 597.8326120162933,
                #     "y": 226.13558375293846,
                #     "width": 200.020366598778,
                #     "height": 182.64797149275802,
                #     "stroke": "red",
                #     "strokeWidth": 2,
                #     "lineCap": "round",
                #     "lineJoin": "round"
                # }
            ],
            "sketchFrameTimestamp": 63.48,
            "editOperation": ""
        },
    }

    result = pipeline.process_request(
        EXAMPLE_REQUEST
    )
    # result = pipeline.predict_temporal_segments(
    #     ["talking about the chicken again"], ["transcript"],
    #     102, [480, 854],
    #     [{"start": 0, "finish": 100}, {"start": 107, "finish": 300}]
    # )
    print(result)

if __name__ == "__main__":
    # test_processor("https://www.youtube.com/live/4LdIvyfzoGY?feature=share")
    launch_server()
    # test("whenever the person mentions the surface go, emphasize the screen response time")
    # test_langchain_pipeline()
