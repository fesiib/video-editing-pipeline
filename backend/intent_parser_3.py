import os
import json
import sys
import argparse
import openai
import ast

from .transcript_parser import *
from .helpers import Timecode
from .operations import *
from evaluation.sentence_embedder import get_cosine_similarity_score

# OLD_METADATA_FILENAME = "./metadata/metadata.txt"
METADATA_FILENAME = "./metadata/4LdIvyfzoGY_10.txt"
PROMPT_PARSER_FILENAME = "./prompts/prompt_parse_intent_3.txt"
PROMPT_TEMPORAL_POSITION_FILENAME = "./prompts/temporal_position.txt"
PROMPT_TEMPORAL_TRANSCRIPT_FILENAME = "./prompts/temporal_transcript_exp.txt"
PROMPT_TEMPORAL_ACTION_FILENAME = "./prompts/temporal_action.txt"
PROMPT_TEMPORAL_VISUAL_FILENAME = "./prompts/temporal_visual.txt"
PROMPT_TEMPORAL_FILENAME = "./prompts/temporal.txt"
PROMPT_SPATIAL_FILENAME = "./prompts/spatial.txt"
PROMPT_PARAMETERS_FILENAME = "./prompts/parameters.txt"

class IntentParser():
    def __init__(self, chunk_size = 20, limit = 40) -> None:
        self.chunk_size = chunk_size
        self.limit = limit
        openai.organization = "org-z6QgACarPepyUdyAq45DMSiB"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.metadata_filename = METADATA_FILENAME
        self.prompt_parse_filename = PROMPT_PARSER_FILENAME
        self.prompt_temporal_position_filename = PROMPT_TEMPORAL_POSITION_FILENAME
        self.prompt_temporal_transcript_filename = PROMPT_TEMPORAL_TRANSCRIPT_FILENAME
        self.prompt_temporal_action_filename = PROMPT_TEMPORAL_ACTION_FILENAME
        self.prompt_temporal_visual_filename = PROMPT_TEMPORAL_VISUAL_FILENAME
        self.prompt_temporal_filename = PROMPT_TEMPORAL_FILENAME
        self.prompt_spatial_filename = PROMPT_SPATIAL_FILENAME
        self.prompt_parameters_filename = PROMPT_PARAMETERS_FILENAME

    def reset(self,
        metadata_filename = METADATA_FILENAME,
        prompt_parse_filename = PROMPT_PARSER_FILENAME,
        prompt_temporal_position_filename = PROMPT_TEMPORAL_POSITION_FILENAME,
        prompt_temporal_transcript_filename = PROMPT_TEMPORAL_TRANSCRIPT_FILENAME,
        prompt_temporal_action_filename = PROMPT_TEMPORAL_ACTION_FILENAME,
        prompt_temporal_visual_filename = PROMPT_TEMPORAL_VISUAL_FILENAME,
        prompt_temporal_filename = PROMPT_TEMPORAL_FILENAME,
        prompt_spatial_filename = PROMPT_SPATIAL_FILENAME,
        prompt_parameters_filename = PROMPT_PARAMETERS_FILENAME          
    ):
        self.metadata_filename = metadata_filename
        self.prompt_parse_filename = prompt_parse_filename
        self.prompt_temporal_position_filename = prompt_temporal_position_filename
        self.prompt_temporal_transcript_filename = prompt_temporal_transcript_filename
        self.prompt_temporal_action_filename = prompt_temporal_action_filename
        self.prompt_temporal_visual_filename = prompt_temporal_visual_filename
        self.prompt_temporal_filename = prompt_temporal_filename
        self.prompt_spatial_filename = prompt_spatial_filename
        self.prompt_parameters_filename = prompt_parameters_filename
    # model="gpt-4"
    # model="gpt-3.5-turbo-16k-0613"
    def completion_endpoint(self, prompt, msg, model="gpt-4"):
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": msg}
            ],
            temperature=0.1,
        )
        print("Usage Info:", json.dumps(completion.usage, indent=2))
        return completion.choices[0].message
    
    def predict_relevant_text(self, input):
        print("calling: ", self.prompt_parse_filename)
        with open(self.prompt_parse_filename, 'r') as f:
            context = f.read()
        completion = self.completion_endpoint(context, input) 
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
        
        # ranges_position = self.process_temporal_position(all_position, all_context)
        # ranges_transcript = self.process_temporal_transcript(all_transcript, all_context)
        # ranges_transcript = self.process_temporal_transcript_cosine_similarity(all_transcript, all_context)
        ranges_action = self.process_temporal_action(all_action, all_context)
        # ranges_visual = self.process_temporal_visual(all_visual, all_context)
        # ranges = merge_ranges(ranges_position + ranges_transcript + ranges_action + ranges_visual)

        # ranges = self.process_temporal_metadata(temporal_segments)

        ranges = ranges_action
        edits = []
        for interval in ranges:
            edits.append(get_timecoded_edit_instance(interval))
        return edits

    def process_temporal_metadata(self, input_texts):
        with open(self.prompt_temporal_filename, 'r') as f:
            prompt = f.read()
        
        video_data = []
        with open(METADATA_SPLIT_FILENAME) as metadata:
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
    
    def process_temporal_position(self, input_texts, context_texts):
        return self.__process_temporal_specific_segments(
            input_texts, context_texts, self.prompt_temporal_position_filename, [])
    
    def process_temporal_transcript(self, input_texts, context_texts):
        # TODO: merge input_texts and context_texts
        if (len(input_texts) == 0):
            return []
        timecoded_transcript = []
        transcript = []
        with open(self.metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                transcript.append(interval["transcript"])
                timecoded_transcript.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "transcript": interval["transcript"],
                })
        relevant_transcript_indexes = self.__process_temporal_specific_indexes(
            "Transcript", input_texts, context_texts,
            self.prompt_temporal_transcript_filename, transcript
        )
        ranges = []
        
        for index in relevant_transcript_indexes:
            if (index >= len(timecoded_transcript) or index < 0):
                print("ERROR: Transcript segment not found in metadata", index)
            else:
                ranges.append({
                    "start": timecoded_transcript[index]["start"],
                    "end": timecoded_transcript[index]["end"],
                })
        return merge_ranges(ranges)
    
    def process_temporal_transcript_cosine_similarity(
        self, input_texts, context_texts,
        threshold=0.5
    ):
        if (len(input_texts) == 0):
            return []
        # TODO: merge input_texts and context_texts
        timecoded_transcript = []
        with open(self.metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                timecoded_transcript.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "transcript": interval["transcript"],
                })
        # for i, item in enumerate(transcript):
        #     print(json.dumps(i), json.dumps(item))
        
        ranges = []

        for input in input_texts:
            found = False
            for item in timecoded_transcript:
                if (get_cosine_similarity_score(input, item["transcript"]) > threshold):
                    ranges.append({
                        "start": item["start"],
                        "end": item["end"],
                    })
                    found = True
            if (not found):
                print("ERROR: Transcript segment not found in metadata", input)
        return merge_ranges(ranges)
    
    def process_temporal_action(self, input_texts, context_texts):
        if (len(input_texts) == 0):
            return []
        timecoded_metadata = []
        metadata = []
        with open(self.metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                metadata.append({
                    "action": interval["action_pred"],
                    "caption": interval["synth_caption"],
                })
                timecoded_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "action": interval["action_pred"],
                    "caption": interval["synth_caption"],
                })
        relevant_indexes = self.__process_temporal_specific_indexes(
            "Metadata", 
            input_texts, context_texts,
            self.prompt_temporal_action_filename, metadata
        )
        ranges = []
        
        for index in relevant_indexes:
            if (index >= len(timecoded_metadata) or index < 0):
                print("ERROR: Transcript segment not found in metadata", index)
            else:
                ranges.append({
                    "start": timecoded_metadata[index]["start"],
                    "end": timecoded_metadata[index]["end"],
                })
        return merge_ranges(ranges)

    
    def process_temporal_visual(self, input_texts, context_texts):
        if (len(input_texts) == 0):
            return []
        metadata = []
        with open(self.metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    #"visual_description": interval["dense_caption"],
                    "caption": interval["synth_caption"],
                })
        return self.__process_temporal_specific_segments(
            input_texts, context_texts, self.prompt_temporal_visual_filename, metadata)
    
    def __process_temporal_specific_segments(self, input_texts, context_texts, prompt_filename, metadata):
        if (len(input_texts) == 0):
            return []
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

    def __process_temporal_specific_indexes(
        self,
        metadata_name,
        input_texts, context_texts,
        prompt_filename, metadata
    ):
        if (len(input_texts) == 0):
            return []
        with open(prompt_filename, 'r') as f:
            prompt = f.read()
        
        for i, item in enumerate(metadata):
            print(json.dumps(i), json.dumps(item))
        
        text_metadata = self.convert_json_list_to_text(metadata, ", ") 
        
        request = ("Context: [" + self.convert_json_list_to_text(context_texts, ", ") + "]"
                    + "\n" + metadata_name + ": [" + text_metadata + "]"
                    + "\nUser Request: [" + self.convert_json_list_to_text(input_texts, ", ") + "]")
        print("REQ: ", request)
        response = self.completion_endpoint(prompt, request)
        response_content = response["content"].replace('\"', "'").replace('\'', "'")
        print("RESP: ", json.dumps(response_content))
        indexes = []
        try:
            indexes = ast.literal_eval(response_content)
        except:
            print("Incorrect format returned by GPT")
        print(json.dumps(indexes))
        return [int(index) for index in indexes]

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
