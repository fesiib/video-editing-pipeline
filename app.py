from flask import Flask, request, jsonify  
from flask_cors import CORS
from backend.intent_parser import IntentParser
app = Flask(__name__)
CORS(app)
'''
Server side for video editing pipline 
'''

'''
Input:
    Video Reference: youtube link
    State of intent:
'''
@app.route("/intent", methods=['GET', 'POST'])
def parse_intent():
    data = request.json 
    edit_request = data.get("input")
    
    intent_parser = IntentParser()
    edit_response = intent_parser.process_message(edit_request)

    response = "User Request: {} \n \n Edit Disambiguation: {}".format(edit_request, edit_response)
    return jsonify({"temporal": 200, "spatial" : response, "edit": })

'''
Summary of the edit description
'''
@app.route("/summary", methods=['GET', 'POST'])
def fetch_summary():
