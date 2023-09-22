import json

from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

PREFIX_TEXT_CONTENT_PROMPT= """
You are a video editor's assistant who is trying to understand natural language request of the editor to find a text to display in the video. You are given a command from the editor, the original context of the command, and relevant content from the video. Relevant content is a list of snippets from the video transcript. You must generate the text to be displayed based on the editor's command, original context, and snippets of transcript.

Note 1: If no relevant text can be generated that satisfies the command, context, and snippets of transcript, output the input command itself with reasonable formatting.
Note 2: Make sure that text is not too long, since it will be displayed on the screen. Keep it under 100 characters.

"""

EXAMPLE_PROMPT = """
Command: {command}
Context: {context}
Transcript snippets: {metadata}
Response: {response}
"""

SUFFIX_TEXT_CONTENT_PROMPT = """
Command: {command}
Context: {context}
Transcript snippets: {metadata}
Response:
"""
#TODO
def get_examples():
    context1 = []
    metadata1 = [
        " oh happy friday everyone so a couple of",
        " serendipitous things sort of happened here uh one i was in the middle of working on a video actually come check this thing out so this is the gpd",
    ]
    command1 = ["a greeting text"]
    response1 = "Happy Friday!"

    context2 = []
    metadata2 = [
        " up before don't finish up now you told me you were done sorry let me just get my wi-fi password in here i'm just gonna throw in some extra motions",
        " so that hopefully you guys reverse engineer my password here because that would be really really really inconvenient for me i would",
        " have to take a whole probably about six minutes or so and log into my interface and then change that all right now we have some important setup to do",
    ]
    command2 = ["an imaginary wifi password"]
    response2 = "Linu*****est"

    context3 = []
    metadata3 = [
        " surface go is using what is it a there it is a 4415 y processor what do you like brandon do you like the unboxing on black",
        " or the unboxing on wood grain which duper you like the wood grain all right we're going to do the wood grain so it's got a 4415 why processor four gigs or eight gigs of ram",
        " it's got 64 gigs or 128 gigs of storage acclaimed nine hours of battery life well this one's",
    ]
    command3 = ["specs of the surface go"]
    response3 = "4415y processor,\n4GB RAM,\n64GB storage,\n9h battery life"

    context4 = []
    metadata4 = [
        " oh happy friday everyone so a couple of",
        " serendipitous things sort of happened here uh one i was in the middle of working on a video actually come check this thing out so this is the gpd",
    ]
    command4 = ["Surface Go price"]
    response4 = "Surface Go price: $399"    

    examples = []
    examples.append({
        "context": json.dumps(context1),
        "metadata": json.dumps(metadata1),
        "command": command1,
        "response": response1,
    })
    examples.append({
        "context": json.dumps(context2),
        "metadata": json.dumps(metadata2),
        "command": command2,
        "response": response2,
    })
    examples.append({
        "context": json.dumps(context3),
        "metadata": json.dumps(metadata3),
        "command": command3,
        "response": response3,
    })
    examples.append({
        "context": json.dumps(context4),
        "metadata": json.dumps(metadata4),
        "command": command4,
        "response": response4,
    })
    return examples

def get_text_content_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["context", "metadata", "command", "response"],
        template=EXAMPLE_PROMPT,
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt_template,
        max_length=300,
    )

    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt_template,
        prefix=PREFIX_TEXT_CONTENT_PROMPT,
        suffix=SUFFIX_TEXT_CONTENT_PROMPT,
        input_variables=["context", "metadata", "command"],
        partial_variables=partial_variables,
    )


def get_text_content_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Command: {command}\nContext: {context}\nTranscript snippets: {metadata}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_TEXT_CONTENT_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Command: {command}\nContext: {context}\nTranscript snippets: {metadata}"),
        ]
    )

    return final_prompt