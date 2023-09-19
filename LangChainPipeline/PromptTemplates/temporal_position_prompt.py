from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.TemporalSegments import TemporalSegments


PREFIX_TEMPORAL_POSITION_PROMPT= """
You are a video editor's assistant who is trying to understand natural language temporal reference in the video. You will do it step-by-step.

First step: Identify the type of temporal reference based on the user's command.
1. Timecode: a specific time in the video
2. Time range: a range of time in the video
3. More high level temporal reference: a reference to a generic event in the video (introduction, ending, etc.)

Second step: Identify the timecode or time range with additional context.
Note 1: If the temporal reference is just a timecode, output any 10 second interval containing the timecode.
Note 2: If there are more than one segment of video that matches the temporal reference, output all of them in a list.

{format_instructions}
"""

EXAMPLE_PROMPT = """
Context: {context}
Command: {command}
Response: {response}
"""

SUFFIX_TEMPORAL_POSITION_PROMPT = """
Context: {context}
Command: {command}
Response:
"""

def get_examples():
    context1 = ["A video is 00:20:13 long."]
    command1 = ["0:07"]
    response1 = TemporalSegments.get_instance(start="00:00:04", finish="00:00:09")

    
    context2 = ["A video is 00:20:13 long."]
    command2 = ["intro"]
    response2 = TemporalSegments.get_instance(start="00:00:00", finish="00:00:30")

    context3 = ["A video is 00:20:13 long."]
    command3 = ["between 5:10 and 5:20"]
    response3 = TemporalSegments.get_instance(start="00:05:10", finish="00:05:20")

    examples = []
    examples.append({
        "context": context1,
        "command": command1,
        "response": response1.model_dump_json(),
    })
    examples.append({
        "context": context2,
        "command": command2,
        "response": response2.model_dump_json(),
    })
    examples.append({
        "context": context3,
        "command": command3,
        "response": response3.model_dump_json(),
    })
    return examples

def get_temporal_position_prompt_llm(partial_variables={}, examples = []):
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
        prefix=PREFIX_TEMPORAL_POSITION_PROMPT,
        suffix=SUFFIX_TEMPORAL_POSITION_PROMPT,
        input_variables=["context", "command"],
        partial_variables=partial_variables,
    )


def get_temporal_position_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Context: {context}\nCommand: {command}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_TEMPORAL_POSITION_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Context: {context}\nCommand: {command}"),
        ]
    )

    return final_prompt