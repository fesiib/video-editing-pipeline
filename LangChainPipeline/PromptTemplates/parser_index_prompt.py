from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.IndexedReferences import IndexedReferences

### TODO: Add audio temporal references. 
PREFIX_INTENT_PARSER= """
You are a video editor's assistant that is trying to understand the natural language command in the context of a given video. Let's do this step-by-step.

Step 1: You have to identify 4 types of references from the natural language command:
1. Temporal reference: any information in the command that could refer to a segment of the video:
- explicit timecodes or time ranges
- explicit mentions or implicit references to the transcript of the video
- description of the actions that happen in the video
- visual description of objects, moments, and frames in the video

2. Spatial reference: any information in the command that could refer to location or region in the video frame:
- specific locations or positions relative to the frame
- specific objects or areas of interest

3. Edit Operation reference: any information in the command that could refer to specific edit operations, specifically ([text, image, shape, blur, cut, crop, zoom]).
- explicit mentions of edit operations
- implicit references to edit operations (e.g. "summarize", "remove", "mask")
- description of the purpose of the edit operation (e.g. "to emphasize", "to hide", "to remove", "to highlight")

4. Edit Parameter reference: any information in the command that could refer to specific parameters of  edit operations ([text, image, shape, blur, cut, crop, zoom]).
- text: content, font style, font color, or font size
- image: visual keywords
- shape: type of shape
- blur: degree of blur to apply
- cut: no parameters
- crop: how much to crop
- zoom: how long to perform the zooming animation

Step 2: Based on the references to edit operations you have recognized, you will identify the list of edit operations that the command is referring to:
- choose only among "text", "image", "shape", "blur", "cut", "crop", "zoom"
- make sure that the edit operation is only one of the above
- if none of the above edit operations is directly relevant, give the one that is most relevant to the command

Step 3-1: Classify each temporal reference you have recognized into one of the following:
1. "position": reference in the form of a timecode (e.g. "54:43", "0:23"), time segment (e.g. "0:00-12:30", "from 43:30 to 44:20") or more abstract temporal position (e.g. "intro", "ending", "beginning part of the video")
2. "transcript": reference to transcript both implicit or explicit
3. "video": reference to specific action in the video or visual description of the frame, object, or elements
4. "other": reference to other temporal information that does not fall into the above categories

Step 3-2: Classify each spatial reference you have recognized into one of the following:
1. "visual-dependent": reference to specific objects, elements, or regions in the video frame that depend on the visual content of the video
2. "independent": reference to specific locations or positions relative to the frame independent of the visual content of the video
3. "other": any other spatial information that does not fall into the above categories

Step 4: Format the output based on the result of each step. Make sure to include the offset of the reference in the given natural language command. For example, if the command is "Zoom into the pan at around 1:31 when he is saying "Make sure to flip chicken after about 6 minutes", the temporal reference "1:31" has an offset of 28. The final output should be:

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

    command1 = 'Do 2 seconds long zoom out at the beginning of the video and add shorter text with flashy shape.'
    response1 = IndexedReferences.get_instance(
        [
            [3, "2 seconds long"], [34, "beginning of the video"]
        ],
        ["position", "position"],
        [],
        [],
        [
            [18, "zoom out"], [73, "text"], [90, "shape"]
        ],
        ["zoom", "text", "shape"],
        [
            [65, "shorter text"]
        ],
        [],
        [
            [83, "flashy shape"]
        ],
        [],
        [],
        [],
        [
            [3, "2 seconds long zoom out"]
        ],
    )

    command2 = '27:32 - start video where he is talking about the chicken again - end the cut when he puts chicken into the oven.'
    response2 = IndexedReferences.get_instance(
        [
            [0, "27:32"], [32, "talking about the chicken again"], [86, "puts the chicken into the oven"]
        ],
        ["position", "transcript", "video"],
        [],
        [],
        [
            [8, "start video"], [66, "end the cut"]
        ],
        ["cut"],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    command3 = 'For each moment where the speaker is talking about learning points, add a text with bigger font and red background.'
    response3 = IndexedReferences.get_instance(
        [
            [4, "each moment where the speaker is talking about learning points"]
        ],
        ["transcript"],
        [],
        [],
        [
            [74, "text"],
        ],
        ["text"],
        [
            [45, 'about learning points'],
            [74, "text with bigger font"], [100, "red background"],
        ],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    command4 = '4:38 - Animate graphics of a book and headphones to either side of subject to engage audience and emphasis point.'
    response4 = IndexedReferences.get_instance(
        [
            [0, "4:38"]
        ],
        ["position"],
        [
            [52, "either side of subject"]
        ],
        ["visual-dependent"],
        [
            [7, "animate graphics"],
            [29, "book"],
            [38, "headphones"],
            [78, "engage audience"],
            [98, "emphasis point"],
        ],
        ["image"],
        [],
        [
            [7, "animate graphics"], [29, "book"], [38, "headphones"]
        ],
        [],
        [],
        [],
        [],
        [],
    )

    command5 = 'Whenever there is laptop seen, highlight it with a transparent star around it'
    response5 = IndexedReferences.get_instance(
        [
            [0, "whenever there is laptop seen"]
        ],
        ["video"],
        [
            [68, "around it"]
        ],
        ["visual-dependent"],
        [
            [31, "highlight it"], [51, "transparent star"]
        ],
        ["shape"],
        [],
        [],
        [
            [51, "transparent star"]
        ],
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
    # examples.append({
    #     "command": command2,
    #     "response": response2.model_dump_json(),
    # })
    examples.append({
        "command": command3,
        "response": response3.model_dump_json(),
    })

    examples.append({
        "command": command4,
        "response": response4.model_dump_json(),
    })
    examples.append({
        "command": command5,
        "response": response5.model_dump_json(),
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