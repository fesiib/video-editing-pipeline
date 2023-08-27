import os
import sys
import argparse
import openai
import ast
from helpers import Timecode

class IntentParser():
    def __init__(self) -> None:
        self.input = "" 
        self.outputs = []
        openai.organization = "org-z6QgACarPepyUdyAq45DMSiB"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.temporal = {"Time Code": "00:00:00", "Transcript": "None", "Action": "None"}
        self.spatial = {"Frame": "None", "Object": "None"}
        self.edit_operation = {}

    def completion_endpoint(self, prompt, msg, model="gpt-3.5-turbo"):
        completion = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": msg}
            ]
        ) 
        return completion.choices[0].message

    def process_message(self, msg):
        file_path = "prompt.txt"
        
        with open(file_path, 'r') as f:
            context = f.read()
        completion = self.completion_endpoint(context, msg) 

        with open("outputs.txt", 'w') as f:
            f.write(str(completion.choices[0].message))
        completion = ast.literal_eval(completion.choices[0].message["content"])
        
        self.temporal = completion["temporal"]
        self.spatial = completion["spatial"]
        self.edit_operation = completion["edit"]
        
        self.process_temporal()
        # self.process_spatial()
        self.process_edit_operation()

    def process_temporal(filename="video_metadata.txt"):
        '''
            Process video metadata retrieved from BLIP2 image captioning + InternVideo Action recognition
        '''
        with open(filename, 'r') as file:
            entries = file.read().split('\n')

        timecoded_metadata = {}
        for entry in entries:
            data = json.loads(entry)
            time = Timecode(data["timecode"])
            content = {
                "caption": data["caption"],
                "tag": data["tag"],
            }
            timecoded_metadata[time] = content

        return
    def process_spatial():
        return
    def process_edit_operation():

        return

    # If any fields results in N/A, ask clarifying question to resolve ambiguity
    def clarify_message():
        return
        

def main():
    intent_parser = IntentParser()
    intent_parser.process_message("When the man gives specific examples in his answer, show the key items/nouns as text next to his head. For example, at 31:48, he mentions caffeine and ‘saw palmetto(?)’ so show these words as text for a brief moment") 

if __name__ == "__main__":
    main()