from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class SingleSegment(BaseModel):
    start: str = Field(
        ...,
        title="Start timecode of the segment",
        description="Start timecode of the segment",
    )
    finish: str = Field(
        ...,
        title="Finish timecode of the segment",
        description="Finish timecode of the segment",
    )

    def __init__(self, start="00:00:00", finish="00:00:10"):
        super().__init__(
            start=start,
            finish=finish,
        )

    @validator("start")
    def start_must_be_valid_timecode(cls, v):
        return v
    
    @validator("finish")
    def finish_must_be_valid_timecode(cls, v):
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
            print("ERROR: segments must be a valid list")
            return []
        return v

    @classmethod
    def get_instance(cls, start, finish):
        return cls(
            segments=[SingleSegment(start=start, finish=finish)]
        )
    
    @classmethod
    def get_dummy_instance(cls):
        return cls(
            segments=[
                SingleSegment(start="00:00:00", finish="00:00:10"),
                SingleSegment(start="00:00:30", finish="00:00:35"),
                SingleSegment(start="00:01:20", finish="00:01:40"),
            ]
        )