# Video Editing ML Pipeline

This is the repository for the pipeline for ExpressEdit: video editing tool based on text and sketch.

## Repository Structure

- [`backend`](): Helper scripts.
- [`LangChainPipeline`](): Main Pipeline implementation based on LangChain framework.
- [`metadata`](): Metadata files for available videos.
- [`segmentation_data`](): Frame Segmentation data for available videos.
- [`run.py`](): Script that runs single processing of the pipeline with given parameters.
- [`requirements.txt`](): Installation packages.
- [`README.md`](): Instructions file.

## Development environment

-   Ubuntu 18.04, CUDA 11.5

## Installation

Create a new [conda](https://docs.conda.io/en/latest/) environment (Python 3.10)

```bash
conda create --name <env> --file requirements.txt
conda activate <env>
```

## Run

Run a single instance of the pipeline for a specific text & script description and a video:
```bash
# -h, --help            
#                       show this help message and exit
# -t TEXT, --text TEXT  
#                       The text description of the edit. Ex: add a text whenever the person is not in the frame
# -s SKETCH, --sketch SKETCH
#                       The sketch rectangle description of the edit. Ex: 0,0,854,480
# -sf SKETCHTIMESTAMP, --sketchTimestamp SKETCHTIMESTAMP
#                       The timestamp of the sketch rectangle (in seconds). Ex: 10
# -v VIDEOID, --videoId VIDEOID
#                       The video-id of the video. Ex: kdN41iYTg3U
# -vd VIDEODURATION, --videoDuration VIDEODURATION
#                       The duration of the video (in seconds). Ex: 3236
python run.py [-h] [-t TEXT] [-s SKETCH] [-sf SKETCHFRAME] [-v VIDEOID] [-vd VIDEODURATION]
```

For example:
```bash
python run.py -v kdN41iYTg3U -vd 3236 -t "whenever there is no person in the video add white text with the name of the person"
```

Notes:
- Make sure to export environment variable $OPENAI_API_KEY in your bash with appropriate OPENAI_API_KEY.
- Sketch format: `<left-x>,<top-y>,<width>,<height>`
- Timestamp/Duration must be an integer: `<int>`
- Video Id must be from available below.

## Available Videos

| Video Id | Video Duration (seconds) | Video Link |
| :------- | :------------- | :--------- |
| kdN41iYTg3U | 3236 | [Youtube Link](https://www.youtube.com/watch?v=kdN41iYTg3U) |
| 3_nLdcHBJY4 | 3774 | [Youtube Link](https://www.youtube.com/watch?v=3_nLdcHBJY4) |
| OKQpOzEY_A4 | 3501 | [Youtube Link](https://www.youtube.com/watch?v=OKQpOzEY_A4) |
| sz8Lo3NY1m0 | 2125 | [Youtube Link](https://www.youtube.com/watch?v=sz8Lo3NY1m0) |
| 4LdIvyfzoGY | 1218 | [Youtube Link](https://www.youtube.com/live/4LdIvyfzoGY?feature=share) |

## Output Format

```python
{
    "parsing_results": {
        "temporal": [str],
        "spatial": [str],
        "edit": [str],
        "parameters": {
            "text": [str],
            "image": [str],
            "shape": [str],
            "blur": [str],
            "cut": [str],
            "crop": [str],
            "zoom": [str]
        },
    },
    "edit_operations": [str],
    "edits": [{
        "start": float,
        "finish": float,
        "temporal_reasoning": [str],
        "temporal_source": [str],
        "spatial": {
            "x": float,
            "y": float,
            "width": float,
            "height": float,
            "rotation": float,
        },
        "spatial_reasoning": [str],
        "spatial_source": [str],
    }, ...],
}
```

For any questions please contact: [Bekzat Tilekbay](mailto:tlekbay.b@gmail.com)
