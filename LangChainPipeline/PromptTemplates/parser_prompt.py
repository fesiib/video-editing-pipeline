from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.References import References

### TODO: Add audio temporal references. 
### TODO: Add spatial reference labels.
### TODO: Deal with it, that, this, those, etc.
PREFIX_INTENT_PARSER= """
You are a video editor's assistant who is trying to understand the natural language command in the context of a given video. You will do it step-by-step.
You have to identify 3 types of references from the command first:

1. Temporal reference: any information in the command that could refer to a segment of the video:
- explicit timecodes or time ranges
- explicit mentions or implicit references to the transcript of the video
- description of the actions that happen in the video
- visual description of objects, moments, and frames in the video

2. Spatial reference: any information in the command that could refer to location or region in the video frame:
- specific locations or positions relative to the frame
- specific objects or areas of interest

3. Edit Operation reference: any information in the command that could refer to one or combination of the following editing operations:
- [text, image, shape, blur, cut, crop, zoom]
- edit operations must be only among the above list

4. Edit Parameter reference: any information in the command that could refer to specific parameters of an edit operation that was identified ([text, image, shape, blur, cut, crop, zoom]).
- text: content, font style, font color, or font size
- image: visual keywords
- shape: type of shape
- blur: degree of blur to apply
- cut: no parameters
- crop: how much to crop
- zoom: how long to perform the zooming animation

As a next step, you will classify each temporal reference into one of the following:
1. "position": reference in the form of a timecode (e.g. "54:43", "0:23"), time segment (e.g. "0:00-12:30", "from 43:30 to 44:20") or more abstract temporal position (e.g. "intro", "ending", "beginning part of the video")
2. "transcript": reference to transcript both implicit or explicit
3. "video": reference to specific action in the video or visual description of the frame, object, or elements
4. "other": reference to other temporal information that does not fall into the above categories

{format_instructions}
"""

EXAMPLE_PROMPT = """
Command: {command}
Response: {response}
"""

SUFFIX_INTENT_PARSER = """
Command: {command}
Response:
"""

def get_examples():
    command1 = 'Zoom into the pan at around 1:31 when he is saying "Make sure to flip chicken after about 6 minutes'
    response1 = References(
        ["1:31", "make sure to flip chicken after about 6 minutes"],
        ["position", "transcript"],
        ["pan"],
        ["zoom"],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    command2 = '27:32 - start video where he is talking about the chicken again - end the cut when he puts chicken into the oven.'
    response2 = References(
        ["27:32", "talking about the chicken again", "puts the chicken into the oven"],
        ["position", "transcript", "video"],
        [],
        ["cut"],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    command3 = 'Whenever he introduces new ingredient and cooking instruction have it listed up in the top left corner in the same arial font but slightly smaller and make sure it doesn’t obstruct his movements for five seconds at which point the text disappears'
    response3 = References(
        ["whenever he introduces new ingredient and cooking instruction", "for five seconds"],
        ["transcript", "other"],
        ["top left corner", "make sure it doesn't obstruct his movements"],
        ["text"],
        ["arial font", "slightly smaller"],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    command4 = '9:22 - Animate graphics of a book and headphones to either side of subject to engage audience and emphasis point.'
    response4 = References(
        ["9:22", ],
        ["position"],
        ["either side of subject"],
        ["image"],
        [],
        ["animate graphics of a book and headphones"],
        [],
        [],
        [],
        [],
        [],
    )

    examples = []
    examples.append({
        "command": command1,
        "response": response1.model_dump_json(),
    })
    examples.append({
        "command": command2,
        "response": response2.model_dump_json(),
    })
    examples.append({
        "command": command3,
        "response": response3.model_dump_json(),
    })

    examples.append({
        "command": command4,
        "response": response4.model_dump_json(),
    })

    return examples

def get_parser_prompt_llm(partial_variables={}):
    example_prompt_template = PromptTemplate(
        input_variables=["command", "response"],
        template=EXAMPLE_PROMPT,
    )

    example_selector = LengthBasedExampleSelector(
        examples=get_examples(),
        example_prompt=example_prompt_template,
        max_length=300,
    )

    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt_template,
        prefix=PREFIX_INTENT_PARSER,
        suffix=SUFFIX_INTENT_PARSER,
        input_variables=["command"],
        partial_variables=partial_variables,
    )


def get_parser_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "{command}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_INTENT_PARSER,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "{command}"),
        ]
    )

    return final_prompt