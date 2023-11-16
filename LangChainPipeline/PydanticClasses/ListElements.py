from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class SingleElement(BaseModel):
    index: int = Field(
        ...,
        title="Index of the element",
        description="Index of the element",
    )
    explanation: str = Field(
        ...,
        title="Explanation why the element is relevant",
        description="Explanation why the element is relevant",
    )

    def __init__(self, index="0", explanation=""):
        super().__init__(
            index=index,
            explanation=explanation,
        )

    @validator("explanation")
    def explanation_must_be_valid_string(cls, v):
        return v
    
    @validator("index")
    def index_must_be_nonnegative(cls, v):
        if v < 0 or type(v) != int:
            print("ERROR: index must be nonnegative")
            return "0"
        return v


class ListElements(BaseModel):
    list_elements: List[SingleElement] = Field(
        ...,
        title="List of selected elements with their explanations or empty list if no relevant elements are found",
        description="List of selected elements with their explanations or empty list if no relevant elements are found",
    )

    def __init__(self, list_elements=[]):
        super().__init__(
            list_elements=list_elements,
        )

    @validator("list_elements")
    def list_elements_must_be_valid_list(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: list_elements must be a valid list")
        return []


    @classmethod
    def get_instance(cls, indexes, explanations):
        return cls(
            list_elements=[
                SingleElement(
                    index=index,
                    explanation=explanation,
                ) for index, explanation in zip(indexes, explanations)
            ]
        )
    
    @classmethod
    def get_dummy_instance(cls):
        return cls(
            list_elements=[
                SingleElement(
                    index=0,
                    explanation="dummy explanation",
                ),
                SingleElement(
                    index=1,
                    explanation="dummy explanation",
                ),
            ]
        )