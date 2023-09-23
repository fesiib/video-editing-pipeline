from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class Rectangle(BaseModel):
    x: int = Field(
        ...,
        title="X coordinate of the top left corner of the rectangle",
        description="X coordinate of the top left corner of the rectangle",
    )
    y: int = Field(
        ...,
        title="Y coordinate of the top left corner of the rectangle",
        description="Y coordinate of the top left corner of the rectangle",
    )
    width: int = Field(
        ...,
        title="Width of the rectangle",
        description="Width of the rectangle",
    )
    height: int = Field(
        ...,
        title="Height of the rectangle",
        description="Height of the rectangle",
    )
    rotation: float = Field(
        0,
        title="Rotation of the rectangle. Default is 0. Range is 180 to -180",
        description="Rotation of the rectangle. Default is 0. Range is 180 to -180",
    )

    def __init__(
            self,
            x=0,
            y=0,
            width=0,
            height=0,
            rotation=0,
    ):
        super().__init__(
            x=x,
            y=y,
            width=width,
            height=height,
            rotation=rotation,
        )
    
    @validator("x")
    def x_must_be_valid(cls, v):
        if type(v) != int:
            print("ERROR: X coordinate must be an integer")
            return 0
        return v
    
    @validator("y")
    def y_must_be_valid(cls, v):
        if type(v) != int:
            print("Y coordinate must be an integer")
            return 0
        return v
    
    @validator("width")
    def width_must_be_valid_and_positive(cls, v):
        if type(v) != int or v < 0:
            print("ERROR: Width must be an integer")
            return 100
        return v
    
    @validator("height")
    def height_must_be_valid_and_positive(cls, v):
        if type(v) != int or v < 0:
            print("ERROR: Height must be an integer")
            return 100
        return v

    @validator("rotation")
    def rotation_must_be_valid(cls, v):
        if type(v) != float or v > 180 or v < -180:
            print("ERROR: Rotation must be a float between -180 and 180")
            return 0
        return v
    
    @classmethod
    def get_instance(cls, **kwargs):
        return cls(**kwargs)