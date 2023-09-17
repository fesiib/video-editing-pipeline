from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class SingleElement(BaseModel):
    index: str = Field(
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

    @validator("index")
    def index_must_be_valid_integer(cls, v):
        assert v.isdigit(), "index must be a valid integer"
        return v

    @validator("explanation")
    def explanation_must_be_valid_string(cls, v):
        return v
    
    @validator("index")
    def index_must_be_nonnegative(cls, v):
        assert int(v) >= 0, "index must be a valid index"
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
        assert isinstance(v, list), "list_elements must be a valid list"
        return v

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