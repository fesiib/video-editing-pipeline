from flask import Flask, request, jsonify  
from flask_cors import CORS
from backend.intent_parser import IntentParser
import json
app = Flask(__name__)
CORS(app)
'''
Server side for video editing pipline 
'''
intent_parser = IntentParser()
'''
Input:
    Video Reference: youtube link
    State of intent:
'''
@app.route("/intent", methods=['GET', 'POST'])
def parse_intent():
    data = request.json
    data = json.loads()
    
    edit_response = intent_parser.process_message(data)

    response = "User Request: {} \n \n Edit Disambiguation: {}".format(edit_request, edit_response)
    return jsonify(msg)

'''
Summary of the edit description
'''
@app.route("/summary", methods=['GET', 'POST'])
def fetch_summary():
    data = request.json
    summary_request = "Generate a several word caption to summarize the purpose of the following video edit request."
    response = intent_parser.completion_endpoint(summary_request, data.get("input"))
    return jsonify({"summary": response})
