USER_DATA_PATH = "user-data"

MSG_TYPES = [
    # manual editing -> work on canvas
    "canvasBatchSelect",
    "canvasObjectDragged",
    "canvasObjectTransformed",
    
    # anything -> just select
    "canvasSingleSelect",
    "editBatchSelect",
    "editBatchSelectSuggested",
    "editSingleSelect",
    "editSingleSelectSuggested",
    "transcriptBatchSelect",
    "transcriptSingleSelect",

    # anything -> zoom change
    "canvasZoomChange",
    "timelineZoomChange",
    
    # describing -> end
    "commandspaceProcess",
    # describing -> process of describing
    "commandspaceTextChange",
    
    # examination -> make sure nothing is missed or suggest more
    "timelineSearchMore",
    # examination -> start of examination
    "processingComplete",
    
    # ideation -> history use
    "intentHistoryDelete",
    "intentHistorySelect",
    
    # ideation -> edit list use
    "intentSelect",
    "intentAdd",
    "intentDelete",
    "intentDuplicate",
    
    # anything -> navigation
    "navigationPlay",
    "transcriptSnippetClicked"
    "timelineIndicatorDragged",
    "timelineIndicatorLabelClicked",
    
    # manual editing / ideation -> if selected
    # examiniation -> if deselected?
    "operationSelect",
    
    # manual editing -> parameter change
    "parameterAlign",
    "parameterCollapse",
    "parameterColor",
    "parameterDropDown",
    "parameterNumberChange",
    "parameterNumberChangeStep",
    "parameterRange",
    "parameterSearch",
    "parameterSearchClick",
    "parameterTextChange",
    "parameterTimeChange",
    "parameterTimeChangeStep",
    "parameterUrlChange",
    
    # describing -> sketching
    "sketchClear",
    "sketchClick",
    "sketchDraw",
    
    # admin -> start of the work
    "taskDone",
    # admin -> end of the work
    "taskSelect",
    
    # manual editing -> make decision
    "timelineDecisionAccept",
    "timelineDecisionReject",
    
    # manual editing -> edit
    "timelineAddSegment",
    "timelineDelete",
    "timelineDuplicate",
    "timelineSegmentDragged",
    "timelineSplit", # -> remove
    "editSplit", # -> "timelineSpilt"
    
    
    # manual editing -> boundaries
    "timelineEditTrimmed",
    "transcriptEditTrimmed",

    # examination / manual editing -> depending on what we are jumping with
    "timelineJumpNext",
    "timelineJumpPrev",

    # manual editing?
    "timelineKey", # -> remove
    
    "transcriptEditTrimming", # -> remove
]

HIGH_LEVEL_PROCESS = {
    "admin": "admin",
    "ideate": "ideate",
    "describe": "ideate",
    "examine": "execute",
    "manual": "execute",
    "anything": "evaluate",
}

COLOR_MAPPING_EDITING_PROCESS = {
    "admin": "white",
    "ideate": "blue",
    "describe": "red",
    "examine": "green",
    "manual": "grey",
    "anything": "black",
    "execute": "purple",
    "evaluate": "pink",
    "anything": "black",
}

NAVIGATION_LOGS = {
    "canvasSingleSelect",
    "editBatchSelect",
    "editBatchSelectSuggested",
    "editSingleSelect",
    "editSingleSelectSuggested",
    "transcriptBatchSelect",
    "transcriptSingleSelect",
    "transcriptSnippetClicked"
    "timelineIndicatorDragged",
    "timelineIndicatorLabelClicked",
    "timelineJumpNext",
    "timelineJumpPrev",
    "navigationPlay",
}

EDITING_PROCESS = {
    "admin": [
        "taskDone",
        "taskSelect",
    ],
    "ideate": [
        "intentSelect",
        "intentAdd",
        "intentDelete",
        "intentDuplicate",
        "intentHistoryDelete",
        "intentHistorySelect",
    ],
    "describe": [
        "commandspaceProcess",
        "commandspaceTextChange",
        "sketchClear",
        "sketchClick",
        "sketchDraw",  
    ],
    "examine": [
        "timelineSearchMore",
        "processingComplete",
        "timelineJumpNext",
        "timelineJumpPrev",
    ],
    "manual": [
        "canvasBatchSelect",
        "canvasObjectDragged",
        "canvasObjectTransformed",
        "operationSelect",
        "parameterAlign",
        "parameterCollapse",
        "parameterColor",
        "parameterDropDown",
        "parameterNumberChange",
        "parameterNumberChangeStep",
        "parameterRange",
        "parameterSearch",
        "parameterSearchClick",
        "parameterTextChange",
        "parameterTimeChange",
        "parameterTimeChangeStep",
        "parameterUrlChange",
        "timelineDecisionAccept",
        "timelineDecisionReject",
        "timelineAddSegment",
        "timelineDelete",
        "timelineDuplicate",
        "timelineSegmentDragged",
        "timelineSplit",
        "timelineEditTrimmed",
        "transcriptEditTrimmed",
        "timelineJumpNext",
        "timelineJumpPrev",
    ],
    "anything": [
        "canvasSingleSelect",
        "editBatchSelect",
        "editBatchSelectSuggested",
        "editSingleSelect",
        "editSingleSelectSuggested",
        "transcriptBatchSelect",
        "transcriptSingleSelect",
        "canvasZoomChange",
        "timelineZoomChange",
        "navigationPlay",
        "transcriptSnippetClicked",
        "timelineIndicatorDragged",
        "timelineIndicatorLabelClicked",
    ],
}

PARTICIPANT_DATABASE = {
    #comparative study
    "a6a25f9c-3abe-441d-a058-fef54e07b7f6": {
        "id": 5,
        "order": 0,
        "study": 1,
        "videos": ["video-1", "video-2"],
        "baseline": "video-1",
    },
    "cad83010-5e5d-41ea-b7a6-923173add11a": {
        "id": 8,
        "order": 1,
        "study": 1,
        "videos": ["video-2", "video-1"],
        "baseline": "video-2",
    },

    #observational study
    "d41e5c77-6237-4672-93df-74f699b16a0d": {
        "id": 7,
        "order": 0,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "5cc37f82-eff0-44ea-9381-3ec46503d9dd": {
        "id": 11,
        "order": 1,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "e81608ac-855a-42cf-a4d6-f97bc6a99446": {
        "id": 13,
        "order": 2,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "edaefe4f-7910-4c08-88f9-5af9219ea8ea": {
        "id": 14,
        "order": 3,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "f1fd7ff6-464b-48e4-a5a7-ad43e2bf93e3": {
        "id": 6,
        "order": 4,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "7141af95-bf36-4bc0-b15d-268412181991": {
        "id": 9,
        "order": 5,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "687f4a25-018a-4652-8991-3b29adbc644c": {
        "id": 22,
        "order": 6,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "38eb9eac-675b-4b00-85cb-453f078228b8": {
        "id": 20,
        "order": 7,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "d3db24d9-4ac6-46b9-8964-fee5f7c3f476": {
        "id": 21,
        "order": 8,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "30f8a4cc-d826-47f5-8745-b519f823f7ba": {
        "id": 24,
        "order": 9,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "c8a80f0b-fa28-4565-a053-ad235f92dc18": { #1
        "id": 25,
        "order": 1,
        "study": 3,
        "videos": ["video-2", "video-1"],
        "baseline": "video-2",
    },
    "595a4105-1a6a-4d95-9d64-e24b32d1f926": { #3
        "id": 26,
        "order": 2,
        "study": 3,
        "videos": ["video-1", "video-2"],
        "baseline": "video-2",
    },
    "778c57a1-083f-46f6-8ef2-e7eec2c113c6": { #3
        "id": 27,
        "order": 3,
        "study": 3,
        "videos": ["video-1", "video-2"],
        "baseline": "video-2",
    },
    "c83c3ad8-62b2-4391-a89b-ab3a38114cdf": { #0
        "id": 28,
        "order": 4,
        "study": 3,
        "videos": ["video-1", "video-2"],
        "baseline": "video-1",
    },
    "e45f5300-bcbb-49c0-9e1b-8e08ed49f80b": { #1
        "id": 29,
        "order": 5,
        "study": 3,
        "videos": ["video-2", "video-1"],
        "baseline": "video-2",
    },
}

EDIT_TYPES = [
    "text", "image", "shape", "cut", "zoom", "crop", "blur", "",
]

REPLACE_LIST = {
    "timelineSplit": None,
    "timelineKey": None,
    "transcriptEditTrimming": None,
    "play": "navigationPlay",
    "editSplit": "timelineSplit",
}