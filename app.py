from flask import Flask, request, jsonify  
from flask_cors import CORS
from intent_parser import IntentParser
app = Flask(__name__)
CORS(app)
@app.route("/intent", methods=['GET', 'POST'])
def hello_world():
    data = request.json 
    edit_request = data.get("input")
    
    intent_parser = IntentParser()
    edit_response = intent_parser.process_message(edit_request)
    response = "User Request: {} \n \n Edit Disambiguation: {}".format(edit_request, edit_response)
    return jsonify({"response_code": 200, "results" : response})