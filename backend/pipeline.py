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
PROMPT_TEMPORAL_VIDEO_FILENAME = "./prompts/temporal_video_exp.txt"
PROMPT_TEMPORAL_FILENAME = "./prompts/temporal.txt"
PROMPT_SPATIAL_FILENAME = "./prompts/spatial.txt"
PROMPT_PARAMETERS_FILENAME = "./prompts/parameters_interpret.txt"

class Pipeline():
    def __init__(self, chunk_size = 20, limit = 40) -> None:
        self.top_k = 10
        self.chunk_size = chunk_size
        self.limit = limit
        openai.organization = "org-z6QgACarPepyUdyAq45DMSiB"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.metadata_filename = METADATA_FILENAME
        self.prompt_parse_filename = PROMPT_PARSER_FILENAME
        self.prompt_temporal_position_filename = PROMPT_TEMPORAL_POSITION_FILENAME
        self.prompt_temporal_transcript_filename = PROMPT_TEMPORAL_TRANSCRIPT_FILENAME
        self.prompt_temporal_video_filename = PROMPT_TEMPORAL_VIDEO_FILENAME
        self.prompt_temporal_filename = PROMPT_TEMPORAL_FILENAME
        self.prompt_spatial_filename = PROMPT_SPATIAL_FILENAME
        self.prompt_parameters_filename = PROMPT_PARAMETERS_FILENAME
        self.parameters_of_interest = ["text", "shape", "blur", "crop", "zoom"]

    def reset(self,
        metadata_filename = METADATA_FILENAME,
        prompt_parse_filename = PROMPT_PARSER_FILENAME,
        prompt_temporal_position_filename = PROMPT_TEMPORAL_POSITION_FILENAME,
        prompt_temporal_transcript_filename = PROMPT_TEMPORAL_TRANSCRIPT_FILENAME,
        prompt_temporal_video_filename = PROMPT_TEMPORAL_VIDEO_FILENAME,
        prompt_temporal_filename = PROMPT_TEMPORAL_FILENAME,
        prompt_spatial_filename = PROMPT_SPATIAL_FILENAME,
        prompt_parameters_filename = PROMPT_PARAMETERS_FILENAME          
    ):
        self.metadata_filename = metadata_filename
        self.prompt_parse_filename = prompt_parse_filename
        self.prompt_temporal_position_filename = prompt_temporal_position_filename
        self.prompt_temporal_transcript_filename = prompt_temporal_transcript_filename
        self.prompt_temporal_video_filename = prompt_temporal_video_filename
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
        edit_request = msg["requestParameters"]["text"]
        from_scratch = msg["requestParameters"]["processingMode"] == "from-scratch"
        add_more = msg["requestParameters"]["processingMode"] == "add-more"
        adjust_selected = msg["requestParameters"]["processingMode"] == "adjust-selected"
        skipped_segments = []
        if ("skippedSegments" in msg and isinstance(msg["skippedSegments"], list) == True):
            for skipped in msg["skippedSegments"]:
                skipped_segments.append({
                    "start": skipped["temporalParameters"]["start"],
                    "finish": skipped["temporalParameters"]["finish"],
                })
        if (from_scratch == False):
            for edit in msg["edits"]:
                skipped_segments.append({
                    "start": edit["temporalParameters"]["start"],
                    "finish": edit["temporalParameters"]["finish"],
                })
        
        if from_scratch == False and add_more == False and adjust_selected == False:
            print("ERROR: Invalid processing mode")
            msg["edits"] = []
            msg["requestParameters"]["editOperations"] = [msg["requestParameters"]["editOperation"]]
            msg["requestParameters"]["parameters"] = {}
            return msg
        ### maybe obtain skipped segments from edits????
        relevant_text = self.predict_relevant_text(edit_request)
        print("relevant_text", relevant_text)
        edits = msg["edits"]
        if (from_scratch or add_more):
            edits = self.predict_temporal_segments(relevant_text["temporal"], relevant_text["temporal_labels"], skipped_segments)  
        msg["requestParameters"]["editOperations"] = relevant_text["edit"]
        edits = self.predict_spatial_regions(relevant_text["spatial"], edits)
        edits = self.predict_parameters(relevant_text["parameters"], relevant_text["edit"], edits, adjust_selected)
        msg["requestParameters"]["parameters"] = relevant_text["parameters"]
        msg["edits"] = edits
        return msg
    
    def predict_temporal_segments(self, temporal_segments, temporal_labels, skipped_segments=[]):
        ranges = []
        all_other = ["Duration of the video: 00:00:00 - 00:20:20"]
        all_position = []
        all_transcript = []
        all_video = []

        for (segment, label) in zip(temporal_segments, temporal_labels):
            if label == "other":
                all_other.append(segment)
            if label == "position":
                all_position.append(segment)
            if label == "transcript":
                all_transcript.append(segment)
            if label == "video":
                all_video.append(segment)
        
        ranges_position = self.process_temporal_position(all_position, all_other)
        ranges_transcript = []
        for reference in all_transcript:
            cur_ranges = self.process_temporal_transcript([reference], all_other, skipped_segments)
            ranges_transcript += cur_ranges
        # ranges_transcript = self.process_temporal_transcript_cosine_similarity(all_transcript, all_other, skipped_segments)
        ranges_video = []

        for reference in all_video:
            cur_ranges = self.process_temporal_video([reference], all_other, skipped_segments)
            ranges_video += cur_ranges
        
        # for interval in ranges_position:
        #     print("position:", interval["start"], interval["end"], interval["info"])
        # for interval in ranges_transcript:
        #     print("transcript:", interval["start"], interval["end"], interval["info"])
        # for interval in ranges_video:
        #     print("video:", interval["start"], interval["end"], interval["info"])
        
        ranges = merge_timecodes(ranges_position + ranges_transcript + ranges_video)

        for interval in ranges:
            print("final:", interval["start"], interval["end"], interval["info"], interval["source"])

        edits = []
        for interval in ranges:
            new_edit = get_timecoded_edit_instance(interval)
            new_edit["temporalParameters"]["info"] = interval["info"]
            new_edit["temporalParameters"]["source"] = interval["source"]
            edits.append(new_edit)
        return edits

    def process_temporal_position(self, input_texts, other_texts):
        return self.__process_temporal_specific_segments(
            input_texts, other_texts, self.prompt_temporal_position_filename, [])
    
    ### TODO: Play with the amount of context
    ### 1. top_k
    ### 2. cosine_similarity
    ### 3. give prev & next N snippets
    def process_temporal_transcript(self, input_texts, other_texts, skipped_segments):
        if (len(input_texts) == 0):
            return []
        all_timecoded_transcript = []
        with open(self.metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                all_timecoded_transcript.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "transcript": interval["transcript"],
                })
        timecoded_transcript = []
        # remove transcript segments that are in the skipped segments
        for item in all_timecoded_transcript:
            start = Timecode(item["start"])
            end = Timecode(item["end"])
            skip = False
            for segment in skipped_segments:
                skipped_start = Timecode(segment["start"])
                skipped_end = Timecode(segment["finish"])        
                if ((start > skipped_start or start == skipped_start)
                    and (end < skipped_end or end == skipped_end)):
                    skip = True
                    break
            if (not skip):
                item["score"] = 0
                for input in input_texts:
                    item["score"] = max(get_cosine_similarity_score(input, item["transcript"]), item["score"])
                timecoded_transcript.append(item)
                
        timecoded_transcript.sort(key=lambda x: x["score"], reverse=True)
        # for item in timecoded_transcript:
        #     print(item["score"], item["transcript"])
        
        timecoded_transcript = timecoded_transcript[0:min(len(timecoded_transcript), self.top_k)]
        timecoded_transcript.sort(key=lambda x: x["start"])
        transcript = []
        for item in timecoded_transcript:
            transcript.append(item["transcript"])

        response = self.__process_temporal_specific_indexes(
            "Transcript", input_texts, other_texts,
            self.prompt_temporal_transcript_filename, transcript
        )
        ranges = []
        
        for item in response:
            index = int(item["index"])
            explanation = item["explanation"]
            source = item["source"]
            if (index >= len(timecoded_transcript) or index < 0):
                print("ERROR: Transcript segment not found in metadata", index)
            else:
                ranges.append({
                    "start": timecoded_transcript[index]["start"],
                    "end": timecoded_transcript[index]["end"],
                    "info": [explanation],
                    "source": source,
                })
        return merge_ranges(ranges)
    
    def process_temporal_transcript_cosine_similarity(
        self, input_texts, other_texts, skipped_segments,
        threshold=0.5
    ):
        input_texts += other_texts
        if (len(input_texts) == 0):
            return []
        timecoded_transcript = []
        with open(self.metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                timecoded_transcript.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "transcript": interval["transcript"],
                })
        
        ranges = []

        for input in input_texts:
            found = False
            for item in timecoded_transcript:
                if (get_cosine_similarity_score(input, item["transcript"]) > threshold):
                    ranges.append({
                        "start": item["start"],
                        "end": item["end"],
                        "info": [],
                        "source": [input]
                    })
                    found = True
            if (not found):
                print("ERROR: Transcript segment not found in metadata", input)
        return merge_ranges(ranges)
    
    ### TODO: Play with the amount of context
    ### 1. top_k
    ### 2. cosine_similarity
    ### 3. summarize metadata
    ### 4. reformat metadata: object -> description (action, caption, dense caption)
    ### 5. give prev & next N snippets
    def process_temporal_video(self, input_texts, other_texts, skipped_segments):
        if (len(input_texts) == 0):
            return []
        all_timecoded_metadata = []
        with open(self.metadata_filename) as f:
            for line in f:
                interval = ast.literal_eval(line.rstrip())
                all_timecoded_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "action": interval["action_pred"],
                    "caption": interval["synth_caption"].strip(),
                    "dense_caption": interval["dense_caption"].strip(),
                })
        
        timecoded_metadata = []
        # remove transcript segments that are in the skipped segments
        for item in all_timecoded_metadata:
            start = Timecode(item["start"])
            end = Timecode(item["end"])
            skip = False
            for segment in skipped_segments:
                skipped_start = Timecode(segment["start"])
                skipped_end = Timecode(segment["finish"])        
                if ((start > skipped_start or start == skipped_start)
                    and (end < skipped_end or end == skipped_end)):
                    skip = True
                    break
            if (not skip):
                item["score"] = 0
                cur_metadata = item["action"] + ", " + item["caption"] + ", " + item["dense_caption"]
                for input in input_texts:
                    item["score"] = max(get_cosine_similarity_score(input, cur_metadata), item["score"])
                timecoded_metadata.append(item)
                
        timecoded_metadata.sort(key=lambda x: x["score"], reverse=True)
        # for item in timecoded_metadata:
        #     cur_metadata = item["action"] + ", " + item["caption"]
        #     print(item["score"], cur_metadata)
        timecoded_metadata = timecoded_metadata[0:min(len(timecoded_metadata), self.top_k)]
        timecoded_metadata.sort(key=lambda x: x["start"])
        metadata = []
        for item in timecoded_metadata:
            metadata.append({
                "action": item["action"],
                "caption": item["caption"],
                "dense_caption": item["dense_caption"],
            })
        
        response = self.__process_temporal_specific_indexes(
            "Metadata", 
            input_texts, other_texts,
            self.prompt_temporal_video_filename, metadata
        )
        ranges = []
        
        for item in response:
            index = int(item["index"])
            explanation = item["explanation"]
            source = item["source"]
            if (index >= len(timecoded_metadata) or index < 0):
                print("ERROR: Transcript segment not found in metadata", index)
            else:
                ranges.append({
                    "start": timecoded_metadata[index]["start"],
                    "end": timecoded_metadata[index]["end"],
                    "info": [explanation],
                    "source": source,
                })
        return merge_ranges(ranges)
    
    def __process_temporal_specific_segments(self, input_texts, other_texts, prompt_filename, metadata):
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
                request = ("User Request: " + item)
                response = self.completion_endpoint(prompt, request)
                #print(response)
                try:
                    cur_ranges = ast.literal_eval(response["content"])
                    for interval in cur_ranges:
                        interval["info"] = []
                        interval["source"] = [item] 
                    ranges += cur_ranges
                except:
                    print("RESPONSE: ", response["content"])
                    print("Incorrect format returned by GPT")
            return merge_ranges(ranges)

        for i in range(0, len(metadata[0:LIMIT]), CHUNK_SIZE):
            for item in input_texts:
                request = ("Context: [" + self.convert_json_list_to_text(other_texts, ", ") + "]"
                           + "\nMetadata: " + self.convert_json_list_to_text(metadata[i:i+CHUNK_SIZE], ", ") 
                           + "\nUser Request: " + item)
                response = self.completion_endpoint(prompt, request)
                #print(response)
                try:
                    ranges += ast.literal_eval(response["content"])
                except:
                    print("RESPONSE: ", response["content"])
                    print("Incorrect format returned by GPT")
        for interval in ranges:
            interval["info"] = []
            interval["source"] = []
        return merge_ranges(ranges)

    def __process_temporal_specific_indexes(
        self,
        metadata_name,
        input_texts, other_texts,
        prompt_filename, metadata
    ):
        if (len(input_texts) == 0):
            return []
        with open(prompt_filename, 'r') as f:
            prompt = f.read()
        
        # for i, item in enumerate(metadata):
        #     print(json.dumps(i), json.dumps(item))
        
        text_metadata = self.convert_json_list_to_text(metadata, ", ") 
        
        request = (metadata_name + ": [" + text_metadata + "]"
                    + "\nUser Request: [" + self.convert_json_list_to_text(input_texts, ", ") + "]")

        # completion = self.completion_endpoint(prompt, request, model="gpt-3.5-turbo-16k-0613")
        completion = self.completion_endpoint(prompt, request, model="gpt-4")
        response = completion["content"]
        # response = completion["content"].replace('\"', "'").replace('\'', "'")
        result = []
        try:
            result = ast.literal_eval(response)
        except:
            print("RESP: ", json.dumps(response))
            print("Incorrect format returned by GPT")
        for interval in result:
            interval["source"] = input_texts
        return result
    
    def predict_spatial_regions(self, input_text, edits):
        return edits
    
    ### TODO: post-processing:
    ### 1. "content" -> disambiguate
    ### 2. "source" -> do image search
    ### 3. "intent-references" -> find the id
    ### 4. "edit-references" -> find the id
    def predict_parameters(self, parameters, edit_operations, edits, adjust):
        total_cnt_requests = 0
        for parameter_key in parameters:
            if (parameter_key not in self.parameters_of_interest):
                continue
            total_cnt_requests += len(parameters[parameter_key])
        if (total_cnt_requests == 0 or len(edits) == 0):
            return edits
        with open(self.prompt_parameters_filename, 'r') as f:
            prompt = f.read()
        
        changes_to_apply = []
        results = []

        parameter_keys = []
        for parameter_key in parameters:
            if (parameter_key not in self.parameters_of_interest):
                continue
            parameter_keys.append(parameter_key)

        for i, edit in enumerate(edits):
            which_result = len(results)
            for j in range(0, i):
                if (self.are_equal_edits(edit, edits[j], [key + "Parameters" for key in parameter_keys])):
                    which_result = j
                    break
            changes_to_apply.append(which_result)
            if which_result < len(results):
                continue
            request_parameters = {}
            for parameter_key in parameter_keys:
                request_parameters[parameter_key] = parameters[parameter_key]
            inital_parameters = {}
            for parameter_key in self.parameters_of_interest:
                initial_parameter_key = parameter_key + "Parameters"
                if (initial_parameter_key in edit):
                    inital_parameters[parameter_key] = edit[initial_parameter_key]
            request = ("Initial Edit Parameters: " + json.dumps(inital_parameters)
                    + "\n User Request: " + json.dumps(request_parameters))
            completion = self.completion_endpoint(prompt, request, model="gpt-4")
            response = completion["content"]
            #response = completion["content"].replace('\"', "'").replace('\'', "'")
            try:
                results.append(ast.literal_eval(response))
            except:
                results.append({})
                print("RESP: ", json.dumps(response))
                print("Incorrect format returned by GPT")
        print("changes_to_apply", changes_to_apply, "results", results)
        for i, edit in enumerate(edits):
            result = results[changes_to_apply[i]]
            for parameter_key in result:
                edit[parameter_key + "Parameters"] = result[parameter_key]
        return edits

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
    
    def are_equal_edits(self, edit1, edit2, keys):
        for key in keys:
            if (key not in edit1 and key not in edit2):
                continue
            if (key not in edit1 or key not in edit2):
                return False
            if (isinstance(edit1[key], dict) == True and isinstance(edit2[key], dict) == True):
                if (self.are_equal_edits(edit1[key], edit2[key], edit1[key].keys()) == False):
                    return False
            elif (edit1[key] != edit2[key]):
                return False
        return True

def main():
    pipeline = Pipeline(20, 20)
    pipeline.process_request({
         "requestParameters": {
            "text": "whenever the person mentions the surface go, emphasize the screen response time",
            "editOperation": "",
        },
        "edits": [],
    })

if __name__ == "__main__":
    main()
