import os
import sys
import argparse
import openai

class IntentParser():
    def __init__(self) -> None:
        self.input = "" 
        self.outputs = []
        openai.organization = "org-z6QgACarPepyUdyAq45DMSiB"
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def process_message(self, msg):
        file_path = "prompt.txt"
        
        with open(file_path, 'r') as f:
            context = f.read()
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": msg}
            ]
        ) 
        with open("outputs.txt", 'w') as f:
            f.write(str(completion.choices[0].message))
        return str(completion.choices[0].message)

    # If any fields results in N/A, ask clarifying question to resolve ambiguity
    def clarify_message():
        return
        

# def main():
#     intent_parser = IntentParser()
#     intent_parser.process_message("When the man gives specific examples in his answer, show the key items/nouns as text next to his head. For example, at 31:48, he mentions caffeine and ‘saw palmetto(?)’ so show these words as text for a brief moment") 

# if __name__ == "__main__":
#     main()