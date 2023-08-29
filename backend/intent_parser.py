import os
import json
import sys
import argparse
import openai
import ast
from .transcript_parser import *
from .helpers import Timecode
from .operations import *

class IntentParser():
    def __init__(self, chunk_size = 20) -> None:
        self.chunk_size = chunk_size
        self.ranges = []
        self.input = "" 
        self.outputs = []
        openai.organization = "org-z6QgACarPepyUdyAq45DMSiB"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.temporal = {"Time Code": "00:00:00", "Transcript": "None", "Action": "None"}
        self.spatial = {"Frame": "None", "Object": "None"}
        self.edit_operation = {}

    def reset(self):
        self.ranges = []
        self.input = "" 
        self.outputs = []

    def completion_endpoint(self, prompt, msg, model="gpt-4"):
        completion = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": msg}
            ]
        ) 
        return completion.choices[0].message

    def process_message(self, msg):
        # edit_request = msg
        edit_request = msg["requestParameters"]["text"]

        file_path = "./prompts/prompt.txt"
        
        with open(file_path, 'r') as f:
            context = f.read()
        completion = self.completion_endpoint(context, edit_request) 

        # with open("outputs.txt", 'w') as f:
        #     f.write(str(completion.choices[0].message))
        completion = ast.literal_eval(completion["content"])
        
        self.temporal = completion["temporal"]
        self.spatial = completion["spatial"]
        self.edit_operation = completion["edit"]
        
        self.process_temporal()
        # self.process_spatial()
        # self.process_edit_operation()
        edits = []
        for interval in self.ranges:
            edits.append(get_timecoded_edit_instance(interval))
        
        msg["edits"] = edits
        msg["requestParameters"]["editOperation"] = self.edit_operation[0]
        return msg

    def convert_json_list_to_text(self, json_list):
        output = ""
        for item in json_list:
            output += json.dumps(item) + '\n'
        return output

    def process_temporal(self, metadata_filename="./prompts/metadata_split.txt", temporal_disambiguation_prompt="./prompts/temporal.txt"):
        '''
            Process video metadata retrieved from BLIP2 image captioning + InternVideo Action recognition
        '''
        with open(temporal_disambiguation_prompt, 'r') as f:
            prompt = f.read()
        
        video_data = []
        with open(metadata_filename) as metadata:
            for line in metadata:
                interval = ast.literal_eval(line.rstrip())
                interval['dense_caption'] = ""
                video_data.append(interval)
        
        # split video data into chunks
        CHUNK_SIZE = self.chunk_size 
        for i in range(0, len(video_data[0:CHUNK_SIZE * 2]), CHUNK_SIZE):
            for item in self.temporal:
                print(item)
                request = self.convert_json_list_to_text(video_data[i:i+CHUNK_SIZE]) + "\n User Request: " + item
                response = self.completion_endpoint(prompt, request)
                print(response)
                try:
                    self.ranges += ast.literal_eval(response["content"])
                except:
                    print("Incorrect format returned by GPT")
        self.ranges = merge_ranges(self.ranges)
        return 

    def process_spatial(self):
        # image_processor = ImageProcessor()
        return
    def process_edit_operation(self):
        return

    # If any fields results in N/A, ask clarifying question to resolve ambiguity
    def clarify_message():
        return
    
    def get_summary(self, input):
        summary_request = "Generate a several word caption to summarize the purpose of the following video edit request."
        response = self.completion_endpoint(summary_request, input)
        completion = ast.literal_eval(response["content"])
        return completion

def main():
    intent_parser = IntentParser()
    intent_parser.process_message("whenever the person mentions the surface go, emphasize the screen response time") 

if __name__ == "__main__":
    main()
