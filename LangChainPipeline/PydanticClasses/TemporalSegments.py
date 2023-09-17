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

    def __init__(self, start="00:00:00", end="00:00:10"):
        super().__init__(
            start=start,
            end=end,
        )

    @validator("start")
    def start_must_be_valid_timecode(cls, v):
        return v
    
    @validator("end")
    def end_must_be_valid_timecode(cls, v):
        return v


class TemporalSegments(BaseModel):
    segments: List[SingleSegment] = Field(
        ...,
        title="List of temporal segments, empty list if no segments are found",
        description="List of temporal segments, empty list if no segments are found",
    )

    def __init__(self, segments = []):
        super().__init__(
            segments=segments,
        )

    @validator("segments")
    def segments_must_be_valid_list(cls, v):
        if isinstance(v, list) == False:
            raise ValueError("List of segments must be a list")
        return v

    @classmethod
    def get_instance(cls, start, end):
        return cls(
            segments=[SingleSegment(start=start, end=end)]
        )