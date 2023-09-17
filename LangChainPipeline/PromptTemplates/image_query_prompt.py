from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

PREFIX_IMAGE_QUERY_PROMPT= """
You are a video editor's assistant who is trying to understand natural language request of the editor to come up with search query for images to put in the video. You are given a command from the editor and a context which you can use to construct the image query. Context is list of visual descriptions (what action is happening, abstract caption, and descriptions of objects) of 10-second segments of the video. You must generate the search query for the image to be displayed based on the context and editor's command.

Note 1: If no relevant search query can be generated that satisfies both context and the command, generate some reasonable search query based on the command only.
Note 2: Make sure that the search query is not too long, since it should be editable by the editor. Keep it under 100 characters.

"""

EXAMPLE_PROMPT = """
Context: {context}
Command: {command}
Response: {response}
"""

SUFFIX_IMAGE_QUERY_PROMPT = """
Context: {context}
Command: {command}
Response:
"""

def get_examples():
    context1 = [
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop with headphones on.",
            "dense_caption": "small black laptop on wooden desk,a man sitting at a desk,the opened laptop,black laptop on desk,a black and silver watch,a white wall,blue cord on the desk,a closed laptop computer,a laptop screen is turned on,the screen is white,a man with short hair,a mans right hand,white box on desk,black keyboard on laptop,the hand of a man,the man is wearing a black shirt,a laptop on a table,a laptop on a table,a white piece of paper\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man holding a laptop in his hands.",
            "dense_caption": "the opened laptop on the table,this is a laptop,a black box with wires plugged into it,a person is sitting on a wooden table,a black wrist band,a hand holding a laptop,laptop screen is on,the screen is white,laptop computer on desk,the keyboard of a laptop,the hand of a person,the brown and black flip phone\n",
        },
    ]
    command1 = ["of devices that can be seen"]
    response1 = "black laptop, black and sliver watch, cell phone, black keyboard png"

    context2 = [
        {
            "action": "using computer",
            "abstract_caption": "a man sitting at a desk holding a laptop in his hands and a computer on the table.",
            "dense_caption": "the laptop is open,a laptop on a table,screen on the laptop,a black wristwatch with silver band,black keyboard on laptop,a white wall behind the laptop,blue cord on the table,white cord attached to laptop,a hand on a laptop,hand holding a cell phone,blue cord on the desk,a hand holding a laptop,black keyboard portion of laptop\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop computer.",
            "dense_caption": "a white laptop computer,a man on a laptop,a laptop on a desk,the computer monitor is on,the mans hair is dark,a black cell phone,papers on the desk,black cord on the desk,white wall in the background,keyboard on the laptop,a black keyboard on a laptop,hand of computer worker,the screen of the laptop\n",
        }, 
    ]
    command2 = ["icon that represents the scene"]
    response2 = "man working at a desk on a laptop icon png"

    context3 = [
        {
            "action": "using computer",
            "abstract_caption": "a man sitting at a desk holding a laptop in his hands and a computer on the table.",
            "dense_caption": "the laptop is open,a laptop on a table,screen on the laptop,a black wristwatch with silver band,black keyboard on laptop,a white wall behind the laptop,blue cord on the table,white cord attached to laptop,a hand on a laptop,hand holding a cell phone,blue cord on the desk,a hand holding a laptop,black keyboard portion of laptop\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop computer.",
            "dense_caption": "a white laptop computer,a man on a laptop,a laptop on a desk,the computer monitor is on,the mans hair is dark,a black cell phone,papers on the desk,black cord on the desk,white wall in the background,keyboard on the laptop,a black keyboard on a laptop,hand of computer worker,the screen of the laptop\n",
        }, 
        {
            "action": "unboxing",
            "abstract_caption": "a man holding a laptop in his hands.",
            "dense_caption": "a man playing the wii,a black framed projector screen,a silver laptop,a person is holding a wii controller,a white apple computer,black cords on the desk,a man with brown hair\n",
        }
    ]
    command3 = ["appropriate meme"]
    response3 = "meme about young man talking about laptop png"

    examples = []
    examples.append({
        "context": context1,
        "command": command1,
        "response": response1,
    })
    examples.append({
        "context": context2,
        "command": command2,
        "response": response2,
    })
    examples.append({
        "context": context3,
        "command": command3,
        "response": response3,
    })
    return examples

def get_image_query_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["context", "command", "response"],
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
        prefix=PREFIX_IMAGE_QUERY_PROMPT,
        suffix=SUFFIX_IMAGE_QUERY_PROMPT,
        input_variables=["context", "command"],
        partial_variables=partial_variables,
    )


def get_image_query_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Context:{context}\nCommand:{command}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_IMAGE_QUERY_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Context:{context}\nCommand:{command}"),
        ]
    )

    return final_prompt