from LangChainPipeline.ParserChains.IntentParserChain import IntentParserChain
from LangChainPipeline.ParserChains.TemporalChain import TemporalChain
from LangChainPipeline.ParserChains.EditChain import EditChain

from LangChainPipeline.utils import merge_segments, timecode_to_seconds

from backend.operations import get_edit_segment

class LangChainPipeline():
    def __init__(self, verbose=False):
        self.input_parser = IntentParserChain(verbose=verbose)
        self.temporal_interpreter = TemporalChain(
            verbose=verbose, video_id="4LdIvyfzoGY", interval=10
        )
        self.parameters_interpreter = EditChain(
            verbose=verbose, video_id="4LdIvyfzoGY", interval=10
        )
        self.spatial_interpreter = None
        self.set_parameters_interpreter = None

    def set_video(self, video_id, interval):
        self.temporal_interpreter.set_video(video_id, interval)

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.temporal_interpreter.set_parameters(top_k, neighbors_left, neighbors_right)
        self.parameters_interpreter.set_parameters(top_k, neighbors_left, neighbors_right)

    def predict_spatial_positions(self, 
        spatial, spatial_labels,
        edits, sketches, video_shape,
    ):
        return edits

    def predict_edit_parameters(self,
        parameters,
        edits, sketches, video_shape,
    ):
        for edit in edits:
            initial_edit_parameters = {
                "textParameters": edit["textParameters"],
                "imageParameters": edit["imageParameters"],
                "shapeParameters": edit["shapeParameters"],
                "blurParameters": edit["blurParameters"],
                "cutParameters": edit["cutParameters"],
                "cropParameters": edit["cropParameters"],
                "zoomParameters": edit["zoomParameters"],
            }
            start = timecode_to_seconds(edit["temporalParameters"]["start"])
            finish = timecode_to_seconds(edit["temporalParameters"]["finish"])
            new_edit_parameters = self.parameters_interpreter.run(
                parameters, initial_edit_parameters,
                start, finish,
                video_shape
            )
            edit["textParameters"] = new_edit_parameters["textParameters"]
            edit["imageParameters"] = new_edit_parameters["imageParameters"]
            edit["shapeParameters"] = new_edit_parameters["shapeParameters"]
            edit["blurParameters"] = new_edit_parameters["blurParameters"]
            edit["cutParameters"] = new_edit_parameters["cutParameters"]
            edit["cropParameters"] = new_edit_parameters["cropParameters"]
            edit["zoomParameters"] = new_edit_parameters["zoomParameters"]

        return edits

    def predict_temporal_segments(self,
        temporal,
        temporal_labels,
        current_player_position,
        video_shape,
        skipped_segments,
    ):
        segments = []

        for reference, label in zip(temporal, temporal_labels):
            partial_segments = self.temporal_interpreter.run([reference], label, skipped_segments)
            segments.extend(partial_segments)
        
        temporal_segments = merge_segments(segments)
        edit_segments = []

        for segment in temporal_segments:
            start = timecode_to_seconds(segment["start"])
            end = timecode_to_seconds(segment["end"])
            explanation = segment["explanation"]
            source = segment["source"]
            edit = get_edit_segment(start, end, explanation, source, video_shape)
            edit_segments.append(edit)

        if len(edit_segments) == 0 and len(temporal) == 0:
            #### if no temporal reference is found, use default 10 seconds from current player position
            start = current_player_position
            finish = current_player_position + 10
            while len(skipped_segments) > 0:
                no_intersection = True
                for skipped_segment in skipped_segments:
                    skipped_start = float(skipped_segment["start"])
                    skipped_end = float(skipped_segment["finish"])
                    if skipped_start <= start < skipped_end:
                        start = skipped_end
                        finish = start + 10
                        no_intersection = False
                    elif start < skipped_start < finish:
                        finish = skipped_start
                if no_intersection:
                    break
            print(start, finish)
            edit_segments = [
                get_edit_segment(start, finish, "", "", video_shape)
            ]
        return edit_segments 

    def build_skipped_segments(self, request):
         ### build skipped segments
        skipped_segments = []
        if ("skippedSegments" in request and isinstance(request["skippedSegments"], list) == True):
            for skipped in request["skippedSegments"]:
                skipped_segments.append({
                    "start": skipped["temporalParameters"]["start"],
                    "finish": skipped["temporalParameters"]["finish"],
                })
        if ("segmentOfInterest" in request):
            segmentStart = request["segmentOfInterest"]["start"]
            segmentFinish = request["segmentOfInterest"]["finish"]
            if segmentStart > 0:
                skipped_segments.append({
                    "start": 0,
                    "finish": segmentStart,
                })
            if segmentFinish < request["projectMetadata"]["duration"]:
                skipped_segments.append({
                    "start": segmentFinish,
                    "finish": request["projectMetadata"]["duration"],
                })
        
        if request["requestParameters"]["processingMode"] != "from-scratch":
            for edit in request["edits"]:
                skipped_segments.append({
                    "start": edit["temporalParameters"]["start"],
                    "finish": edit["temporalParameters"]["finish"],
                })
        
        return skipped_segments


    def process_request(self, request):
        command = request["requestParameters"]["text"]
        skipped_segments = self.build_skipped_segments(request)
        prev_edits = request["edits"]
        current_player_position = request["curPlayPosition"]
        # sketchRectangles: [...this.sketchCommand], sketchFrameTimestamp: this.sketchPlayPosition,
        sketches = request["requestParameters"]["sketchRectangles"]
        for sketch in sketches:
            sketch["timestamp"] = request["requestParameters"]["sketchFrameTimestamp"]
        print(sketches)

        ### TODO: build sketches

        video_shape = [request["projectMetadata"]["height"], request["projectMetadata"]["width"]]

        from_scratch = request["requestParameters"]["processingMode"] == "from-scratch"
        add_more = request["requestParameters"]["processingMode"] == "add-more"
        adjust_selected = request["requestParameters"]["processingMode"] == "adjust-selected"

        if from_scratch == False and add_more == False and adjust_selected == False:
            print("ERROR: Invalid processing mode")
            request["edits"] = []
            request["requestParameters"]["editOperations"] = []
            request["requestParameters"]["parameters"] = {}
            return request
        
        ### parse command
        references = self.input_parser.run(command)
        print(references)

        ### set edit operations
        request["requestParameters"]["editOperations"] = references.edit
        request["requestParameters"]["parameters"] = references.get_parameters_short()
        
        ### predict temporal segments
        if from_scratch == True or add_more == True:
            edits = self.predict_temporal_segments(
                references.temporal, references.temporal_labels, 
                current_player_position, video_shape,
                skipped_segments, 
            )

        else:
            edits = prev_edits

        ### predict spatial positions
        edits = self.predict_spatial_positions(
            references.spatial, [],
            edits, sketches, video_shape,
        )

        ### predict edit parameters
        edits = self.predict_edit_parameters(
            references.get_parameters(),
            edits, sketches, video_shape,
        )   

        request["edits"] = edits

        return request









