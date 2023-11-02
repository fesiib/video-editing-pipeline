# Video Editing ML Pipeline

This is the repository for the pipeline for ExpressEdit: video editing tool based on text and sketch.

## Repository Structure

- [`backend`](/backend/): Helper scripts.
- [`LangChainPipeline`](/LangChainPipeline/): Main Pipeline implementation based on LangChain framework.
- [`metadata`](/metadata/): Metadata files for available videos.
- [`segmentation_data`](/segmentation-data/): Frame Segmentation data for available videos.
- [`run.py`](/run.py): Script that runs single processing of the pipeline with given parameters.
- [`requirements.txt`](/requirements.txt): Installation packages.
- [`README.md`](/README.md): Instructions file.

## Development environment

-   Ubuntu 18.04, CUDA 11.5

## Installation

1. Create a new [conda](https://docs.conda.io/en/latest/) environment (Python 3.10)

```bash
conda env create -f environment.yml
conda activate test-env
```

2. Install [CLIP](https://github.com/openai/CLIP) package.
```bash
pip install git+https://github.com/openai/CLIP.git
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
python run.py -v kdN41iYTg3U -vd 3236 -t "whenever there is no person in the video add picture of a person on the right" -s 0,0,200,200 -sf 100
```

Notes:
- Export environment variable `$OPENAI_API_KEY` with appropriate OPENAI_API_KEY from [OpenAI](https://openai.com/).
- Export environment variables `$GOOGLE_API_KEY` and `$GOOGLE_CSE_ID` with appropriate API keys for Image Search [Tutorial](https://developers.google.com/custom-search/v1/overview).
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
        "edit_parameters": {
                "text": dict(),
                "image": dict(),
                "shape": dict(),
                "blur": dict(),
                "cut": dict(),
                "crop": dict(),
                "zoom": dict(),
            },
    }, ...],
}
```

For any questions please contact: [Bekzat Tilekbay](mailto:tlekbay.b@gmail.com)
