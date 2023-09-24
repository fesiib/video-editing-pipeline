from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

PREFIX_SUMMARIZE_REQUEST_PROMPT= """
You are an assistant to a video editor. You are given a video edit request and you need to summarize the purpose of the request in 4 words or less.
"""

EXAMPLE_PROMPT = """
Request: {request}
Response: {response}
"""

SUFFIX_SUMMARIZE_REQUEST_PROMPT = """
Request: {request}
Response:
"""

def get_examples():
    examples = []
    request1 = "I want to remove the part where I say 'um' a lot."
    response1 = "Remove 'um' parts."
    request2 = "Whenever laptop is mentioned, put a white text with a transparent background saying 'laptop'."
    response2 = "Add 'laptop' text."
    
    examples.append({
        "request": request1,
        "response": response1,
    })
    examples.append({
        "request": request2,
        "response": response2,
    })
    return examples

def get_summarize_request_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["request", "response"],
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
        prefix=PREFIX_SUMMARIZE_REQUEST_PROMPT,
        suffix=SUFFIX_SUMMARIZE_REQUEST_PROMPT,
        input_variables=["request"],
        partial_variables=partial_variables,
    )


def get_summarize_request_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Request: {request}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_SUMMARIZE_REQUEST_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Request: {request}"),
        ]
    )

    return final_prompt