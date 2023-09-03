import json

from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS

#from backend.intent_parser import *
from backend.pipeline import Pipeline

from video_host.processor import process_video, get_video_by_filename


app = Flask(__name__)

CORS(app, origins=["http://localhost:7777", "http://internal.kixlab.org:7777"])

'''
Server side for video editing pipline 
'''
#intent_parser = IntentParser(40, 80)
pipeline = Pipeline(50, 0)
'''
Input:
    Video Reference: youtube link
    State of intent:
'''
@app.route("/intent", methods=['GET', 'POST'])
def parse_intent():
    edit_request = request.json
    
    pipeline.reset()
    edit_response = pipeline.process_request(edit_request)
    
    response = "User Request: {} \n \n Edit Disambiguation: {}".format(edit_request, edit_response)

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
if __name__ == "__main__":
    launch_server()
    # test("whenever the person mentions the surface go, emphasize the screen response time")
