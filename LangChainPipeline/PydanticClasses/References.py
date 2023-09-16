from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class References(BaseModel):
    """References to temporal, spatial, and edit operations in a user's video editing command"""

    temporal: List[str] = Field(..., description="Temporal references")
    temporal_labels: List[str] = Field(..., description="Temporal reference labels")
    spatial: List[str] = Field(..., description="Spatial references")
    edit: List[str] = Field(..., description="Edit operation references")
    textParameters: List[str] = Field(..., description="Text edit parameter references")
    imageParameters: List[str] = Field(..., description="Image edit parameter references")
    shapeParameters: List[str] = Field(..., description="Shape edit parameter references")
    blurParameters: List[str] = Field(..., description="Blur edit parameter references")
    cutParameters: List[str] = Field(..., description="Cut edit parameter references")
    cropParameters: List[str] = Field(..., description="Crop edit parameter references")
    zoomParameters: List[str] = Field(..., description="Zoom edit parameter references")

    # temporal: Sequence[str] = Field(..., description="Temporal references and if there are no temporal references, please leave it empty")
    # temporal_labels: Sequence[str] = Field(..., description="Temporal reference labels and if there are no temporal references, please leave it empty")
    # spatial: Sequence[str] = Field(..., description="Spatial references and if there are no spatial references, please leave it empty")
    # edit: Sequence[str] = Field(..., description="Edit operation references and if there are no edit operation references, please leave it empty")
    # textParameters: Sequence[str] = Field(..., description="Text edit parameter references and if there are no text edit parameter references, please leave it empty")
    # imageParameters: Sequence[str] = Field(..., description="Image edit parameter references and if there are no image edit parameter references, please leave it empty")
    # shapeParameters: Sequence[str] = Field(..., description="Shape edit parameter references and if there are no shape edit parameter references, please leave it empty")
    # blurParameters: Sequence[str] = Field(..., description="Blur edit parameter references and if there are no blur edit parameter references, please leave it empty")
    # cutParameters: Sequence[str] = Field(..., description="Cut edit parameter references and if there are no cut edit parameter references, please leave it empty")
    # cropParameters: Sequence[str] = Field(..., description="Crop edit parameter references and if there are no crop edit parameter references, please leave it empty")
    # zoomParameters: Sequence[str] = Field(..., description="Zoom edit parameter references and if there are no zoom edit parameter references, please leave it empty")
    
    def __init__(
        self,
        temporal,
        temporal_labels,
        spatial,
        edit,
        textParameters,
        imageParameters,
        shapeParameters,
        blurParameters,
        cutParameters,
        cropParameters,
        zoomParameters,
    ):
        super().__init__(
            temporal=temporal,
            temporal_labels=temporal_labels,
            spatial=spatial,
            edit=edit,
            textParameters=textParameters,
            imageParameters=imageParameters,
            shapeParameters=shapeParameters,
            blurParameters=blurParameters,
            cutParameters=cutParameters,
            cropParameters=cropParameters,
            zoomParameters=zoomParameters,
        )

    @validator("temporal")
    def check_temporal(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Temporal reference {v[i]} is empty"
        return v

    @validator("temporal_labels")
    def check_temporal_labels(cls, v):
        for i in range(len(v)):
            assert v[i] in ["position", "transcript", "video", "other"], f"Temporal label {v[i]} is not valid"
        return v
    
    @validator("spatial")
    def check_spatial(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Spatial reference {v[i]} is empty"
        return v
    
    @validator("edit")
    def check_edit(cls, v):
        for i in range(len(v)):
            assert v[i] in ["text", "image", "shape", "blur", "cut", "crop", "zoom"], f"Edit operation {v[i]} is not valid"
        return v
        

    @validator("textParameters")
    def check_textParameters(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Text parameter reference {v[i]} is empty"
        return v
    
    @validator("imageParameters")
    def check_imageParameters(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Image parameter reference {v[i]} is empty"
        return v
    
    @validator("shapeParameters")
    def check_shapeParameters(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Shape parameter reference {v[i]} is empty"
        return v
    
    @validator("blurParameters")
    def check_blurParameters(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Blur parameter reference {v[i]} is empty"
        return v
    
    @validator("cutParameters")
    def check_cutParameters(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Cut parameter reference {v[i]} is empty"
        return v
    
    @validator("cropParameters")
    def check_cropParameters(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Crop parameter reference {v[i]} is empty"
        return v
    
    @validator("zoomParameters")
    def check_zoomParameters(cls, v):
        for i in range(len(v)):
            assert v[i] != "", f"Zoom parameter reference {v[i]} is empty"
        return v

    def get_parameters(self):
        return {
            "text": self.textParameters,
            "image": self.imageParameters,
            "shape": self.shapeParameters,
            "blur": self.blurParameters,
            "cut": self.cutParameters,
            "crop": self.cropParameters,
            "zoom": self.zoomParameters,
        }