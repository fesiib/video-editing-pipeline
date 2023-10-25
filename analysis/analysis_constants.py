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
    "asansyzbai@gmail.com": {
        "id": 5,
        "order": 0,
        "study": 1,
        "videos": ["video-1", "video-2"],
        "baseline": "video-1",
    },
    "azhybaev.baktynur@gmail.com": {
        "id": 8,
        "order": 1,
        "study": 1,
        "videos": ["video-2", "video-1"],
        "baseline": "video-2",
    },

    #observational study
    "sultanidinov0567@gmail.com": {
        "id": 7,
        "order": 0,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "yernaz.akhmetov@gmail.com": {
        "id": 11,
        "order": 1,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "enlik.tleukulova@gmail.com": {
        "id": 13,
        "order": 2,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "kamila240373@gmail.com": {
        "id": 14,
        "order": 3,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "ysamargarita2002@gmail.com": {
        "id": 6,
        "order": 4,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "tomirisnurullo1000@gmail.com": {
        "id": 9,
        "order": 5,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "kh.mukashev@gmail.com": {
        "id": 22,
        "order": 6,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
    },
    "saryalovich@gmail.com": {
        "id": 20,
        "order": 7,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "samgenius42@gmail.com": {
        "id": 21,
        "order": 8,
        "study": 2,
        "videos": ["video-1"],
        "baseline": "",
    },
    "trangnguyen20062003@gmail.com": {
        "id": 24,
        "order": 9,
        "study": 2,
        "videos": ["video-2"],
        "baseline": "",
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