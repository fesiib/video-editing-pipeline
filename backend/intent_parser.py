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
    
    def predict_relevant_text(self, input, file_path="./prompts/prompt.txt"):
        
        with open(file_path, 'r') as f:
            context = f.read()
        completion = self.completion_endpoint(context, input) 

        # with open("outputs.txt", 'w') as f:
        #     f.write(str(completion.choices[0].message))
        completion = ast.literal_eval(completion["content"])
        
        return {
            "temporal": completion["temporal"],
            "temporal_labels": completion["temporal_labels"],
            "spatial": completion["spatial"],
            "edit": completion["edit"],
            "parameters": completion["parameters"],
        }

    def process_request(self, msg):
        # edit_request = msg
        edit_request = msg["requestParameters"]["text"]
        relevant_text = self.predict_relevant_text(edit_request)
        edits = self.predict_temporal_segments(relevant_text["temporal"], relevant_text["temporal_labels"])
        msg["edits"] = edits
        msg["requestParameters"]["editOperation"] = relevant_text["edit"][0]
        return msg
    
    def predict_temporal_segments(self, temporal_segments, temporal_labels):
        ranges = []
        all_context = ["Duration of the video: 00:00:00 - 00:20:20"]
        all_position = []
        all_transcript = []
        all_action = []
        all_visual = []

        for (segment, label) in zip(temporal_segments, temporal_labels):
            if label == "context":
                all_context.append(segment)
            if label == "position":
                all_position.append(segment)
            if label == "transcript":
                all_transcript.append(segment)
            if label == "action":
                all_action.append(segment)
            if label == "visual":
                all_visual.append(segment)
        
        # ranges_position = self.process_temporal_position(all_position, all_context, "./prompts/temporal_position.txt")
        ranges_transcript = self.process_temporal_transcript(all_transcript, all_context, "./prompts/temporal_transcript.txt", "./metadata/metadata.txt")
        # ranges_action = self.process_temporal_action(all_action, all_context, "./prompts/temporal_action.txt", "./metadata/metadata.txt")
        # ranges_visual = self.process_temporal_visual(all_visual, all_context, "./prompts/temporal_visual.txt", "./metadata/metadata.txt")
        # ranges = merge_ranges(ranges_position + ranges_transcript + ranges_action + ranges_visual)

        #ranges = self.process_temporal_metadata(temporal_segments, "./metadata/metadata_split.txt", "./prompts/temporal.txt")
        ranges = ranges_transcript
        edits = []
        for interval in ranges:
            edits.append(get_timecoded_edit_instance(interval))
        
        return edits

    def convert_list_to_text(self, input_list, separator='\n'):
        if (len(input_list) == 0):
            return ""
        output = ""
        for item in input_list[0:-1]:
            output += item + separator
        output += input_list[-1]
        return output

    def convert_json_list_to_text(self, json_list, separator='\n'):
        if (len(json_list) == 0):
            return ""
        output = ""
        for item in json_list[0:-1]:
            output += json.dumps(item) + separator
        output += json.dumps(json_list[-1])
        return output

    def process_temporal_metadata(self, input_texts, metadata_filename="./prompts/metadata_split.txt", temporal_disambiguation_prompt="./prompts/temporal.txt"):
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
                request = self.convert_json_list_to_text(video_data[i:i+CHUNK_SIZE]) + "\nUser Request: " + item
                response = self.completion_endpoint(prompt, request)
                #print(response)
                try:
                    ranges += ast.literal_eval(response["content"])
                except:
                    print("Incorrect format returned by GPT")
        return merge_ranges(ranges)
    
    def process_temporal_position(self, input_texts, context_texts, prompt_filename):
        return self.__process_temporal_specific(input_texts, context_texts, prompt_filename, [])
    
    def process_temporal_transcript(self, input_texts, context_texts, prompt_filename, metadata_filename):
        # TODO: merge input_texts and context_texts
        with open(prompt_filename, 'r') as f:
            prompt = f.read()
        timecoded_transcript = []
        transcript = []
        with open(metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                transcript.append(interval["transcript"])
                timecoded_transcript.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "transcript": interval["transcript"],
                })
        relevant_transcript = []
        # for i, item in enumerate(transcript):
        #     print(json.dumps(i), json.dumps(item))
        
        text_transcript = self.convert_json_list_to_text(transcript, ", ") 
        
        request = ("Context: [" + self.convert_json_list_to_text(context_texts, ", ") + "]"
                    + "\nTranscript: [" + text_transcript + "]"
                    + "\nUser Request: [" + self.convert_json_list_to_text(input_texts, ", ") + "]")
        response = self.completion_endpoint(prompt, request)

        try:
            relevant_transcript = ast.literal_eval(response["content"])
        except:
            print("Incorrect format returned by GPT")

        ranges = []
        for transcript_segment in relevant_transcript:
            found = False
            for item in timecoded_transcript:
                if (transcript_segment == item["transcript"]):
                    ranges.append({
                        "start": item["start"],
                        "end": item["end"],
                    })
                    found = True
                    break
            if (not found):
                print("ERROR: Transcript segment not found in metadata", transcript_segment)
        return merge_ranges(ranges)
        #return self.__process_temporal_specific(input_texts, context_texts, prompt_filename, metadata)
    
    def process_temporal_action(self, input_texts, context_texts, prompt_filename, metadata_filename):
        metadata = []
        with open(metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "action": interval["action_pred"],
                    "caption": interval["synth_caption"],
                })
        return self.__process_temporal_specific(input_texts, context_texts, prompt_filename, metadata)
    
    def process_temporal_visual(self, input_texts, context_texts, prompt_filename, metadata_filename):
        metadata = []
        with open(metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    #"visual_description": interval["dense_caption"],
                    "caption": interval["synth_caption"],
                })
        return self.__process_temporal_specific(input_texts, context_texts, prompt_filename, metadata)
    
    def __process_temporal_specific(self, input_texts, context_texts, prompt_filename, metadata):
        # TODO: merge input_texts and context_texts
        with open(prompt_filename, 'r') as f:
            prompt = f.read()
        
        # split video data into chunks
        CHUNK_SIZE = min(self.chunk_size, len(metadata))
        LIMIT = max(min(self.limit, len(metadata)), CHUNK_SIZE)
        if (LIMIT == 0):
            LIMIT = len(metadata)

        ranges = []

        if (len(metadata) == 0):
            for item in input_texts:
                request = ("Context: [" + self.convert_json_list_to_text(context_texts, ", ") + "]"
                           + "\nUser Request: " + item)
                response = self.completion_endpoint(prompt, request)
                #print(response)
                try:
                    ranges += ast.literal_eval(response["content"])
                except:
                    print("RESPONSE: ", response["content"])
                    print("Incorrect format returned by GPT")
            return merge_ranges(ranges)

        for i in range(0, len(metadata[0:LIMIT]), CHUNK_SIZE):
            for item in input_texts:
                request = ("Context: [" + self.convert_json_list_to_text(context_texts, ", ") + "]"
                           + "\nMetadata: " + self.convert_json_list_to_text(metadata[i:i+CHUNK_SIZE], ", ") 
                           + "\nUser Request: " + item)
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
