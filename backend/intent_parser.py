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
    def __init__(self, chunk_size = 20, limit = 40) -> None:
        self.chunk_size = chunk_size
        self.limit = limit
        openai.organization = "org-z6QgACarPepyUdyAq45DMSiB"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # self.temporal = {"Time Code": "00:00:00", "Transcript": "None", "Action": "None"}
        # self.spatial = {"Frame": "None", "Object": "None"}
        # self.edit_operation = {}

    def reset(self):
        # self.temporal = {"Time Code": "00:00:00", "Transcript": "None", "Action": "None"}
        # self.spatial = {"Frame": "None", "Object": "None"}
        # self.edit_operation = {}
        pass

    def completion_endpoint(self, prompt, msg, model="gpt-4"):
        completion = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": msg}
            ]
        ) 
        return completion.choices[0].message
    
    def predict_relevant_text(self, input):
        file_path = "./prompts/prompt.txt"
        
        with open(file_path, 'r') as f:
            context = f.read()
        completion = self.completion_endpoint(context, input) 

        # with open("outputs.txt", 'w') as f:
        #     f.write(str(completion.choices[0].message))
        completion = ast.literal_eval(completion["content"])
        
        return {
            "temporal": completion["temporal"],
            "spatial": completion["spatial"],
            "edit": completion["edit"],
        }

    def process_request(self, msg):
        # edit_request = msg
        edit_request = msg["requestParameters"]["text"]
        relevant_text = self.predict_relevant_text(edit_request)
        edits = self.predict_temporal_segments(relevant_text["temporal"])
        msg["edits"] = edits
        msg["requestParameters"]["editOperation"] = relevant_text["edit"][0]
        return msg
    
    def predict_temporal_segments(self, input_texts):
        ranges = self.process_temporal(input_texts)
        edits = []
        for interval in ranges:
            edits.append(get_timecoded_edit_instance(interval))
        
        return edits

    def convert_json_list_to_text(self, json_list):
        output = ""
        for item in json_list:
            output += json.dumps(item) + '\n'
        return output

    def process_temporal(self, input_texts, metadata_filename="./prompts/metadata_split.txt", temporal_disambiguation_prompt="./prompts/temporal.txt"):
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
        LIMIT = min(self.limit, len(video_data))
        if (LIMIT == 0):
            LIMIT = len(video_data)

        ranges = []

        for i in range(0, len(video_data[0:LIMIT]), CHUNK_SIZE):
            for item in input_texts:
                print(item)
                if (Timecode.is_timecode(item)):
                    timecode = Timecode(item)
                    start_timecode = Timecode(video_data[i]["start"])
                    end_timecode = Timecode(video_data[min(i+CHUNK_SIZE-1, len(video_data) - 1)]["end"])
                    if (start_timecode > timecode or end_timecode < timecode):
                        continue
                    cur_time = int(timecode.convert_timecode_to_sec())
                    cur_range = [{
                        "start": Timecode.convert_sec_to_timecode(max(cur_time - 2, 0)),
                        "end": Timecode.convert_sec_to_timecode(cur_time + 2),
                    }]
                    ranges += cur_range
                    continue
                request = self.convert_json_list_to_text(video_data[i:i+CHUNK_SIZE]) + "\n User Request: " + item
                response = self.completion_endpoint(prompt, request)
                #print(response)
                try:
                    ranges += ast.literal_eval(response["content"])
                except:
                    print("Incorrect format returned by GPT")
        return merge_ranges(ranges)

    def process_spatial(self, input_texts, temporal_segments):
        # image_processor = ImageProcessor()
        return
    def process_edit_operation(self, input_texts):
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
    intent_parser.process_request({
         "requestParameters": {
            "text": "whenever the person mentions the surface go, emphasize the screen response time",
            "editOperation": "",
        },
        "edits": [],
    }) 

if __name__ == "__main__":
    main()
