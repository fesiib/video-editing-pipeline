from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.EditParameters import EditParameters


PREFIX_ALL_PARAMETERS_PROMPT= """
You are a video editor's assistant who is trying to understand video edit parameter change requests in natural language. You will do it step-by-step.

Step 1: Identify the type of each edit parameter change based on the user's command. There are three types of video edit parameter change requests:
1. Explicit: explicit values for a parameter (e.g. 12px, 10%, "Introduction", etc.)
2. Relative: a relative change to a parameter (e.g. 5 seconds longer, 10% less, fewer words, etc.)
3. Abstract: an abstract change to a parameter (e.g. shorter, longer, more, less, etc.)

Step 2: Transform each type of parameter change request into parameter values based on the "Initial parameters" provided and output the adjusted set of video edit parameters.

{format_instructions}
"""

EXAMPLE_PROMPT = """
Initial Parameters: {initial_parameters}
Command: {command}
Response: {response}
"""

SUFFIX_ALL_PARAMETERS_PROMPT = """
Initial Parameters: {initial_parameters}
Command: {command}
Response:
"""

def get_examples():
    initial_parameters1 = EditParameters.get_instance(
        textParameters={"content":"Text","style":{"fill":"#000000","fontSize":12,"fontFamily":"Arial","align":"center","verticalAlign":"middle"},"background":{"fill":"#ffffff","alpha":1}},
        # imageParameters={"source": "/placeholder.jpg", "searchQuery": ""},
        shapeParameters={"type":"rectangle","background":{"fill":"#ffffff","alpha":1},"stroke":{"width":2,"fill":"#000000","alpha":1},"star":{"numPoints":6,"innerRadius":100}},
        blurParameters={"blur":6},
        # cutParameters={},
        cropParameters={"x":0,"y":0,"width":0,"height":0,"cropX":0,"cropY":0,"cropWidth":0,"cropHeight":0},
        zoomParameters={"zoomDurationStart":0,"zoomDurationEnd":0},
    )
    command1 = {
        "textParameters": ["bigger font","red background"],
        # "imageParameters": [],
        "shapeParameters": [],
        "blurParameters": [],
        # "cutParameters": [],
        "cropParameters": [],
        "zoomParameters": [],
    }
    response1 = EditParameters.get_instance(
        textParameters={"content":"Text","style":{"fill":"#000000","fontSize":15,"fontFamily":"Arial","align":"center","verticalAlign":"middle"},"background":{"fill":"#ff0000","alpha":1}},
        # imageParameters={"source": "/placeholder.jpg", "searchQuery": ""},
        shapeParameters={"type":"rectangle","background":{"fill":"#ffffff","alpha":1},"stroke":{"width":2,"fill":"#000000","alpha":1},"star":{"numPoints":6,"innerRadius":100}},
        blurParameters={"blur":6},
        # cutParameters={},
        cropParameters={"x":0,"y":0,"width":0,"height":0,"cropX":0,"cropY":0,"cropWidth":0,"cropHeight":0},
        zoomParameters={"zoomDurationStart":0,"zoomDurationEnd":0},
    )

    
    initial_parameters2 = EditParameters.get_instance(
        textParameters={"content":"The text is about how to make it work?","style":{"fill":"#000000","fontSize":12,"fontFamily":"Courier New","align":"left","verticalAlign":"bottom"},"background":{"fill":"#ffffff","alpha":1}},
        # imageParameters={"source": "/placeholder.jpg", "searchQuery": ""},
        shapeParameters={"type":"rectangle","background":{"fill":"#ffffff","alpha":1},"stroke":{"width":2,"fill":"#000000","alpha":1},"star":{"numPoints":6,"innerRadius":100}},
        blurParameters={"blur":6},
        # cutParameters={},
        cropParameters={"x":0,"y":0,"width":0,"height":0,"cropX":0,"cropY":0,"cropWidth":0,"cropHeight":0},
        zoomParameters={"zoomDurationStart":0,"zoomDurationEnd":0},
    )
    command2 = {
        "textParameters": ["shorter text"],
        # "imageParameters": [],
        "shapeParameters": ["flashy shape"],
        "blurParameters": [],
        # "cutParameters": [],
        "cropParameters": [],
        "zoomParameters": ["2 seconds long zoom out"],
    }
    response2 = EditParameters.get_instance(
        textParameters={"content":"How to make it work?","style":{"fill":"#000000","fontSize":12,"fontFamily":"Courier New","align":"left","verticalAlign":"bottom"},"background":{"fill":"#ffffff","alpha":1}},
        # imageParameters={"source": "/placeholder.jpg", "searchQuery": ""},
        shapeParameters={"type":"star","background":{"fill":"#ffff00","alpha":1},"stroke":{"width":2,"fill":"#000000","alpha":1},"star":{"numPoints":10,"innerRadius":200}},
        blurParameters={"blur":6},
        # cutParameters={},
        cropParameters={"x":0,"y":0,"width":0,"height":0,"cropX":0,"cropY":0,"cropWidth":0,"cropHeight":0},
        zoomParameters={"zoomDurationStart":0,"zoomDurationEnd":2},
    )


    examples = []
    examples.append({
        "initial_parameters": initial_parameters1.model_dump_json(),
        "command": command1,
        "response": response1.model_dump_json(),
    })
    examples.append({
        "initial_parameters": initial_parameters2.model_dump_json(),
        "command": command2,
        "response": response2.model_dump_json(),
    })
    return examples

def get_all_parameters_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["initial_parameters", "command", "response"],
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
        prefix=PREFIX_ALL_PARAMETERS_PROMPT,
        suffix=SUFFIX_ALL_PARAMETERS_PROMPT,
        input_variables=["initial_parameters", "command"],
        partial_variables=partial_variables,
    )


def get_all_parameters_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Initial Parameters:{initial_parameters}\nCommand:{command}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_ALL_PARAMETERS_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Initial Parameters:{initial_parameters}\nCommand:{command}"),
        ]
    )

    return final_prompt