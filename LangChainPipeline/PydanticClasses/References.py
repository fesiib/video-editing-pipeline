from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class References(BaseModel):
    """References to temporal, spatial, and edit operations in a user's video editing command"""

    temporal: List[str] = Field(..., description="Temporal references")
    temporal_labels: List[str] = Field(..., description="Temporal reference labels")
    spatial: List[str] = Field(..., description="Spatial references")
    spatial_labels: List[str] = Field(..., description="Spatial reference labels")
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
    # spatial_labels: Sequence[str] = Field(..., description="Spatial reference labels and if there are no spatial references, please leave it empty")
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
        temporal=[],
        temporal_labels=[],
        spatial=[],
        spatial_labels=[],
        edit=[],
        textParameters=[],
        imageParameters=[],
        shapeParameters=[],
        blurParameters=[],
        cutParameters=[],
        cropParameters=[],
        zoomParameters=[],
    ):
        super().__init__(
            temporal=temporal,
            temporal_labels=temporal_labels,
            spatial=spatial,
            spatial_labels=spatial_labels,
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
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Temporal reference is empty")
        return result

    @validator("temporal_labels")
    def check_temporal_labels(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] in ["position", "transcript", "video", "other"]:
                result.append(v[i])
            else:
                print(f"WARNING: Temporal label {v[i]} is not valid")
        return result
    
    @validator("spatial")
    def check_spatial(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Spatial reference is empty")
        return result
    
    @validator("spatial_labels")
    def check_spatial_labels(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] in ["visual-dependent", "independent", "other"]:
                result.append(v[i])
            else:
                print(f"WARNING: Spatial label {v[i]} is not valid")
        return result
    
    @validator("edit")
    def check_edit(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] in ["text", "image", "shape", "blur", "cut", "crop", "zoom"]:
                result.append(v[i])
            else:
                print(f"WARNING: Edit operation {v[i]} is not valid")
        return result
        

    @validator("textParameters")
    def check_textParameters(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Text parameter reference is empty")
        return result
    
    @validator("imageParameters")
    def check_imageParameters(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Image parameter reference is empty")
        return result
    
    @validator("shapeParameters")
    def check_shapeParameters(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Shape parameter reference is empty")
        return result
    
    @validator("blurParameters")
    def check_blurParameters(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Blur parameter reference is empty")
        return result
    
    @validator("cutParameters")
    def check_cutParameters(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Cut parameter reference is empty")
        return result
    
    @validator("cropParameters")
    def check_cropParameters(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Crop parameter reference is empty")
        return result
    
    @validator("zoomParameters")
    def check_zoomParameters(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] != "":
                result.append(v[i])
            else:
                print("WARNING: Zoom parameter reference is empty")
        return result

    def get_parameters(self):
        return {
            "textParameters": self.textParameters,
            "imageParameters": self.imageParameters,
            "shapeParameters": self.shapeParameters,
            "blurParameters": self.blurParameters,
            "cutParameters": self.cutParameters,
            "cropParameters": self.cropParameters,
            "zoomParameters": self.zoomParameters,
        }

    def get_parameters_short(self):
        return {
            "text": self.textParameters,
            "image": self.imageParameters,
            "shape": self.shapeParameters,
            "blur": self.blurParameters,
            "cut": self.cutParameters,
            "crop": self.cropParameters,
            "zoom": self.zoomParameters,
        }