from flask import Flask, request, jsonify  
from flask_cors import CORS
from backend.intent_parser import *
import json

app = Flask(__name__)

CORS(app, origins = ["http://localhost:3000"])

'''
Server side for video editing pipline 
'''
intent_parser = IntentParser(40, 80)
'''
Input:
    Video Reference: youtube link
    State of intent:
'''
@app.route("/intent", methods=['GET', 'POST'])
def parse_intent():
    edit_request = request.json
    
    intent_parser.reset()
    edit_response = intent_parser.process_request(edit_request)
    
    response = "User Request: {} \n \n Edit Disambiguation: {}".format(edit_request, edit_response)

    return jsonify(edit_response)

'''
Summary of the edit description
'''
@app.route("/summary", methods=['GET', 'POST'])
def fetch_summary():
    data = request.json

    intent_parser.reset()
    response = intent_parser.get_summary(data.get("input"))

    return jsonify({"summary": response})

def launch_server():
    app.run(host="0.0.0.0", port=9888)

def test(edit_request):
    intent_parser.reset()
    print("!!!response!!!", json.dumps(intent_parser.process_request({
        "requestParameters": {
            "text": edit_request,
            "editOperation": "text",
        },
        "edits": [],
    }), indent=1))

if __name__ == "__main__":
    #launch_server()
    test("whenever the person mentions the surface go, emphasize the screen response time")
