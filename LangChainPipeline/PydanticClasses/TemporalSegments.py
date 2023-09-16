from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class SingleSegment(BaseModel):
    start: str = Field(
        ...,
        title="Start timecode of the segment",
        description="Start timecode of the segment",
    )
    end: str = Field(
        ...,
        title="End timecode of the segment",
        description="End timecode of the segment",
    )

    def __init__(self, **data):
        super().__init__(**data)

    @validator("start")
    def start_must_be_valid_timecode(cls, v):
        return v
    
    @validator("end")
    def end_must_be_valid_timecode(cls, v):
        return v


class TemporalSegments(BaseModel):
    segments: List[SingleSegment] = Field(
        ...,
        title="List of temporal segments",
        description="List of temporal segments",
    )

    def __init__(self, **data):
        super().__init__(**data)

    @validator("segments")
    def segments_must_be_valid_timecode(cls, v):
        return v

    @classmethod
    def get_instance(cls, start, end):
        return cls(segments=[SingleSegment(start=start, end=end)])