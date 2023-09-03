import os
import re
from .helpers import Timecode
import json

TIMECODED_TRANSCRIPT = []

def extract_words(s):
    # Match words that are between >< or outside of any tags
    words = re.findall(r'>([^<]+)<|([^<>\s]+)', s)
    
    # Flatten the list of tuples and remove empty strings
    result = [word for tup in words for word in tup if word]
    return result

def format_transcript(transcript_file, output_file):
    output = ""
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read()
        # Split the transcript into line groups
        line_groups = transcript.strip().split("\n\n")

        # Filter out line groups that don't have a line containing '</c>'
        filtered_line_groups = [group for group in line_groups if any('</c>' in line for line in group.split("\n"))]

        for group in filtered_line_groups:
            first_line = group.split("\n")[0]
            last_line = group.split("\n")[2]

            pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})"
            match = re.search(pattern, first_line)

            if match:
                start_timecode = match.group(1)
                end_timecode = match.group(2)
                filtered_line = '<' + start_timecode + '>' + last_line + '<' + end_timecode + '>'
                cleaned_string = filtered_line.replace('<c>', '').replace('</c>', '')
                words = extract_words(cleaned_string)
                idx = 0
                while idx + 2 < len(words):
                    new_arr = words[idx:idx + 3]
                    timecoded_words = {"start": new_arr[0], "end": new_arr[2], "text": new_arr[1]}
                    TIMECODED_TRANSCRIPT.append({"start": Timecode(new_arr[0]), "end": Timecode(new_arr[2]), "text": new_arr[1]})
                    output += str(timecoded_words) + '\n'
                    idx += 2
            else:
                print("No timecodes found.")
    with open(output_file, 'w') as o:
        o.write(output)
    return TIMECODED_TRANSCRIPT

def extract_range(start_timecode, end_timecode, transcript):
    transcript_snippet = {"start": None, "end": None, "text": ""}
    for i, timecode in enumerate(transcript):
        if timecode["start"] > start_timecode and transcript_snippet["start"] is None and timecode["start"] < end_timecode:
            if i - 1 >=0: transcript_snippet["start"] = transcript[i - 1]["start"]
            else: transcript_snippet["start"] = transcript[i]["start"]
        if transcript_snippet["start"] is not None:
            if timecode["text"][0] != ' ': transcript_snippet["text"] += ' '
            transcript_snippet["text"] += timecode["text"] 
        if timecode["end"] > end_timecode and transcript_snippet["start"] is not None:
            transcript_snippet["end"] = transcript[i]["end"]
            return transcript_snippet
    if transcript_snippet["start"] is not None:
        transcript_snippet["end"] = transcript[-1]["end"]
    return transcript_snippet 

def merge_ranges(timecodes):
    #print(input_timecodes, "input timecodes")
    # run through all timecodes and convert then to Timecode format
    for i in range(len(timecodes)):
        item = timecodes[i]
        timecodeEnd = Timecode(item["end"])
        timecodeStart = Timecode(item["start"])
        timecodes[i]["end"] = timecodeEnd
        timecodes[i]["start"] = timecodeStart
        #item["end"] = Timecode(item["end"])
        #item["start"] = Timecode(item["start"])
    return merge_timecodes(timecodes)

def merge_timecodes(timecodes):
    timecodes.sort(key=lambda x: x["start"].convert_timecode_to_sec())
    
    merged_timecodes = []
    for item in timecodes:
        if len(merged_timecodes) == 0:
            merged_timecodes.append(item)
        else:
            if item["start"] < merged_timecodes[-1]["end"] or item["start"] == merged_timecodes[-1]["end"]:
                merged_timecodes[-1]["end"] = max(item["end"], merged_timecodes[-1]["end"])
                merged_timecodes[-1]["info"] += item["info"]
                merged_timecodes[-1]["source"] += item["source"]
            else:
                merged_timecodes.append(item)

    return merged_timecodes

        
if __name__=="__main__":
    format_transcript("transcript.txt", "filtered_transcript.txt")
    print(extract_range(Timecode('00:00:10'), Timecode('00:00:30'), TIMECODED_TRANSCRIPT))
    timecodes = merge_ranges([{'start': '00:00:40', 'end': '00:00:45'}, {'start': '00:00:45', 'end': '00:00:55'}])
    print(timecodes)