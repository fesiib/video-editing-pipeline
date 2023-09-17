from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

class TextStyleParameters(BaseModel):
    """Text style parameters"""
    fill: str = Field(..., description="Text fill color")
    fontSize: int = Field(..., description="Text font size")
    fontFamily: str = Field(..., description="Text font family")
    align: str = Field(..., description="Text alignment")
    verticalAlign: str = Field(..., description="Text vertical alignment")

    def __init__(self, fill, fontSize, fontFamily, align, verticalAlign):
        super().__init__(
            fill=fill,
            fontSize=fontSize,
            fontFamily=fontFamily,
            align=align,
            verticalAlign=verticalAlign,
        )
    
    @validator("fill")
    def fill_must_be_hex(cls, v):
        if not v.startswith("#"):
            raise ValueError("Fill must be a hex value")
        return v
    
    @validator("fontSize")
    def font_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Font size must be positive")
        return v
    
    @validator("fontFamily")
    def font_family_must_be_valid(cls, v):
        if v not in ["Arial", "Times New Roman", "Courier New"]:
            raise ValueError("Font family must be valid")
        return v
    
    @validator("align")
    def align_must_be_valid(cls, v):
        if v not in ["left", "center", "right"]:
            raise ValueError("Align must be valid")
        return v
    
    @validator("verticalAlign")
    def vertical_align_must_be_valid(cls, v):
        if v not in ["top", "middle", "bottom"]:
            raise ValueError("Vertical align must be valid")
        return v
    
class BackgroundParameters(BaseModel):
    """Text background parameters"""
    fill: str = Field(..., description="Text background fill color")
    alpha: float = Field(..., description="Text background alpha")

    def __init__(self, fill, alpha):
        super().__init__(
            fill=fill,
            alpha=alpha,
        )
    
    @validator("fill")
    def fill_must_be_hex(cls, v):
        if not v.startswith("#"):
            raise ValueError("Fill must be a hex value")
        return v
    
    @validator("alpha")
    def alpha_must_be_between_zero_and_one(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Alpha must be between 0 and 1")
        return v

class StrokeParameters(BaseModel):
    """Text stroke parameters"""
    width: int = Field(..., description="Text stroke width")
    fill: str = Field(..., description="Text stroke fill color")
    alpha: float = Field(..., description="Text stroke alpha")

    def __init__(self, width, fill, alpha):
        super().__init__(
            width=width,
            fill=fill,
            alpha=alpha,
        )
    
    @validator("width")
    def width_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Width must be positive")
        return v
    
    @validator("fill")
    def fill_must_be_hex(cls, v):
        if not v.startswith("#"):
            raise ValueError("Fill must be a hex value")
        return v
    
    @validator("alpha")
    def alpha_must_be_between_zero_and_one(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Alpha must be between 0 and 1")
        return v
    
class StarParameters(BaseModel):
    """Shape star parameters"""
    numPoints: int = Field(..., description="Number of points in star")
    innerRadius: int = Field(..., description="Inner radius of star")

    def __init__(self, numPoints, innerRadius):
        super().__init__(
            numPoints=numPoints,
            innerRadius=innerRadius,
        )
    
    @validator("numPoints")
    def num_points_must_be_at_least_2(cls, v):
        if v < 2:
            raise ValueError("Number of points must be at least 2")
        return v
    
    @validator("innerRadius")
    def inner_radius_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Inner radius must be positive")
        return v

class TextParameters(BaseModel):
    """Text edit parameters"""
    content: str = Field(..., description="Text content")
    style: TextStyleParameters = Field(..., description="Text style")
    background: BackgroundParameters = Field(..., description="Text background")

    def __init__(self, content, style, background):
        style = TextStyleParameters(**style)
        background = BackgroundParameters(**background)
        super().__init__(
            content=content,
            style=style,
            background=background,
        )
    
    @validator("content")
    def content_must_be_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("Content must be a string")
        return v
    
class ImageParameters(BaseModel):
    """Image edit parameters"""
    source: str = Field(..., description="Image source")
    searchQuery: str = Field(..., description="Image search query")

    def __init__(self, source, searchQuery):
        super().__init__(
            source=source,
            searchQuery=searchQuery,
        )
    
    @validator("source")
    def source_must_be_link_or_directory(cls, v):
        if not v.startswith("http") and not v.startswith("/"):
            raise ValueError("Source must be a link or directory")
    
    @validator("searchQuery")
    def search_query_must_be_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("Search query must be a string")
        return v

class ShapeParameters(BaseModel):
    """Shape background parameters"""
    type: str = Field(..., description="Shape type")
    background: BackgroundParameters = Field(..., description="Shape background")
    stroke: StrokeParameters = Field(..., description="Shape stroke")
    star: StarParameters = Field(..., description="Shape star parameters")

    def __init__(self, type, background, stroke, star):
        background = BackgroundParameters(**background)
        stroke = StrokeParameters(**stroke)
        star = StarParameters(**star)
        super().__init__(
            type=type,
            background=background,
            stroke=stroke,
            star=star,
        )

    @validator("type")
    def type_must_be_valid(cls, v):
        if v not in ["rectangle", "circle", "star"]:
            raise ValueError("Type must be valid")
        return v
    
class ZoomParameters(BaseModel):
    """Zoom edit parameters"""
    zoomDurationStart: int = Field(..., description="Zoom duration start")
    zoomDurationEnd: int = Field(..., description="Zoom duration end")

    def __init__(self, zoomDurationStart, zoomDurationEnd):
        super().__init__(
            zoomDurationStart=zoomDurationStart,
            zoomDurationEnd=zoomDurationEnd,
        )
    
    @validator("zoomDurationStart")
    def zoom_duration_start_must_be_nonnegative(cls, v):
        if v < 0:
            raise ValueError("Zoom duration start must be positive")
        return v
    
    @validator("zoomDurationEnd")
    def zoom_duration_end_must_be_nonnegative(cls, v):
        if v < 0:
            raise ValueError("Zoom duration end must be positive")
        return v
    
class CropParameters(BaseModel):
    """Crop edit parameters"""
    x: int = Field(..., description="Crop x")
    y: int = Field(..., description="Crop y")
    width: int = Field(..., description="Crop width")
    height: int = Field(..., description="Crop height")
    cropX: int = Field(..., description="Crop x")
    cropY: int = Field(..., description="Crop y")
    cropWidth: int = Field(..., description="Crop width")
    cropHeight: int = Field(..., description="Crop height")

    def __init__(self, x, y, width, height, cropX, cropY, cropWidth, cropHeight):
        super().__init__(
            x=x,
            y=y,
            width=width,
            height=height,
            cropX=cropX,
            cropY=cropY,
            cropWidth=cropWidth,
            cropHeight=cropHeight,
        )
    
    @validator("width")
    def width_must_be_nonnegative(cls, v):
        if v < 0:
            raise ValueError("Width must be positive")
        return v
    
    @validator("height")
    def height_must_be_nonnegative(cls, v):
        if v < 0:
            raise ValueError("Height must be positive")
        return v
    
    @validator("cropWidth")
    def crop_width_must_be_nonnegative(cls, v):
        if v < 0:
            raise ValueError("Crop width must be positive")
        return v
    
    @validator("cropHeight")
    def crop_height_must_be_nonnegative(cls, v):
        if v < 0:
            raise ValueError("Crop height must be positive")
        return v
        
class BlurParameters(BaseModel):
    """Blur edit parameters"""
    blur: int = Field(..., description="Blur")

    def __init__(self, blur):
        super().__init__(
            blur=blur,
        )
    
    @validator("blur")
    def blur_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Blur must be positive")
        return v
    
class CutParameters(BaseModel):
    """Cut edit parameters"""
    pass

class EditParameters(BaseModel):
    """All Edit Parameters together"""

    textParameters: TextParameters = Field(..., description="Text edit parameters")
    # imageParameters: ImageParameters = Field(..., description="Image edit parameters")
    shapeParameters: ShapeParameters = Field(..., description="Shape edit parameters")
    blurParameters: BlurParameters = Field(..., description="Blur edit parameters")
    # cutParameters: CutParameters = Field(..., description="Cut edit parameters")
    cropParameters: CropParameters = Field(..., description="Crop edit parameters")
    zoomParameters: ZoomParameters = Field(..., description="Zoom edit parameters")
    
    def __init__(
        self,
        textParameters,
        # imageParameters,
        shapeParameters,
        blurParameters,
        # cutParameters,
        cropParameters,
        zoomParameters,
    ):
        super().__init__(
            textParameters=textParameters,
            # imageParameters=imageParameters,
            shapeParameters=shapeParameters,
            blurParameters=blurParameters,
            # cutParameters=cutParameters,
            cropParameters=cropParameters,
            zoomParameters=zoomParameters,
        )

    def get_parameters(self):
        return {
            "textParameters": self.textParameters,
            # "imageParameters": self.imageParameters,
            "shapeParameters": self.shapeParameters,
            "blurParameters": self.blurParameters,
            # "cutParameters": self.cutParameters,
            "cropParameters": self.cropParameters,
            "zoomParameters": self.zoomParameters,
        }

    def get_parameters_short(self):
        return {
            "text": self.textParameters,
            # "image": self.imageParameters,
            "shape": self.shapeParameters,
            "blur": self.blurParameters,
            # "cut": self.cutParameters,
            "crop": self.cropParameters,
            "zoom": self.zoomParameters,
        }
    
    @classmethod
    def get_instance(cls, textParameters={}, imageParameters={}, shapeParameters={}, blurParameters={}, cutParameters={}, cropParameters={}, zoomParameters={}):
        textParameters = TextParameters(**textParameters)
        # imageParameters = ImageParameters(**imageParameters)
        shapeParameters = ShapeParameters(**shapeParameters)
        blurParameters = BlurParameters(**blurParameters)
        # cutParameters = CutParameters(**cutParameters)
        cropParameters = CropParameters(**cropParameters)
        zoomParameters = ZoomParameters(**zoomParameters)

        return cls(
            textParameters=textParameters,
            # imageParameters=imageParameters,
            shapeParameters=shapeParameters,
            blurParameters=blurParameters,
            # cutParameters=cutParameters,
            cropParameters=cropParameters,
            zoomParameters=zoomParameters,
        )