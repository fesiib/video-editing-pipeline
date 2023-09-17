#context, command, candidate
from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.Rectangle import Rectangle


PREFIX_SPATIAL_POSITION_PROMPT= """
You are a video editor's assistant who is trying to understand editor's natural language description of the spatial location within the frame. The description is based on the rectangle that is already present in the frame. You will have to refine its location and resize (if necessary) based on the command. You will do it step-by-step.

Instructions:
1. You will be given an initial location of the rectangle in the frame and the boundaries of the frame (do not exceed outside the boundaries) (i.e width = 480, height = 854).
2. You will be given a command that describes the desired spatial location of the rectangle in the frame.
3. You will have to refine the location of the rectangle based on the command.
4. You will have to resize the rectangle based on the command.

Perform each step one-by-one and output the final location of the rectangle in the frame.

{format_instructions}
"""

EXAMPLE_PROMPT = """
Frame Size: {context}
Rectangle: {rectangle}
Command: {command}
Response: {response}
"""

SUFFIX_SPATIAL_POSITION_PROMPT = """
Frame Size: {context}
Rectangle: {rectangle}
Command: {command}
Response:
"""

def get_examples():
    #example1
    context1 = "height=100, width=100"
    rectangle1 = Rectangle.get_instance(
        x=0,
        y=0,
        width=100,
        height=100,
        rotation=0,
    )
    command1 = ["top left corner"]
    response1 = Rectangle.get_instance(
        x=0,
        y=0,
        width=40,
        height=40,
        rotation=0,
    )
    #example2
    context2 = "height=480, width=854"
    rectangle2 = Rectangle.get_instance(
        x=150,
        y=300,
        width=300,
        height=100,
        rotation=0,
    )
    command2 = ["right side of the frame"]
    response2 = Rectangle.get_instance(
        x=554,
        y=300,
        width=300,
        height=100,
        rotation=0,
    )
    #example3
    context3 = "height=200, width=200"
    rectangle3 = Rectangle.get_instance(
        x=50,
        y=0,
        width=100,
        height=50,
        rotation=0,
    )
    command3 = ["bottom center"]
    response3 = Rectangle.get_instance(
        x=50,
        y=150,
        width=100,
        height=50,
        rotation=0,
    )
    #example4
    context4 = "height=500, width=1000"
    rectangle4 = Rectangle.get_instance(
        x=300,
        y=190,
        width=100,
        height=100,
        rotation=0,
    )
    command4 = ["title-like"]
    response4 = Rectangle.get_instance(
        x=250,
        y=0,
        width=500,
        height=100,
        rotation=0,
    )
    examples = []
    examples.append({
        "context": context1,
        "rectangle": rectangle1.model_dump_json(),
        "command": command1,
        "response": response1.model_dump_json(),
    })
    examples.append({
        "context": context2,
        "rectangle": rectangle2.model_dump_json(),
        "command": command2,
        "response": response2.model_dump_json(),
    })
    examples.append({
        "context": context3,
        "rectangle": rectangle3.model_dump_json(),
        "command": command3,
        "response": response3.model_dump_json(),
    })
    examples.append({
        "context": context4,
        "rectangle": rectangle4.model_dump_json(),
        "command": command4,
        "response": response4.model_dump_json(),
    })
    return examples

def get_spatial_position_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["context", "rectangle", "command", "response"],
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
        prefix=PREFIX_SPATIAL_POSITION_PROMPT,
        suffix=SUFFIX_SPATIAL_POSITION_PROMPT,
        input_variables=["context", "rectangle", "command"],
        partial_variables=partial_variables,
    )


def get_spatial_position_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Frame Size:{context}\nRectangle:{rectangle}\nCommand:{command}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_SPATIAL_POSITION_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Frame Size:{context}\nRectangle:{rectangle}\nCommand:{command}"),
        ]
    )

    return final_prompt