import ast
import json
with open("../prompts/metadata.txt") as file:
    with open("../prompts/metadata_split.txt", "w") as output:
        for line in file:
            print(line.rstrip())
            interval = ast.literal_eval(line.rstrip())
            interval['dense_caption'] = ""
            output.write(json.dumps(interval) + '\n')

