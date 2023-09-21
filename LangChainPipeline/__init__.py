import copy

from LangChainPipeline.ParserChains.IntentParserChain import IntentParserChain
from LangChainPipeline.ParserChains.TemporalChain import TemporalChain
from LangChainPipeline.ParserChains.EditChain import EditChain
from LangChainPipeline.ParserChains.SpatialChain import SpatialChain

from LangChainPipeline.utils import merge_segments, timecode_to_seconds, are_same_objects

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
        self.spatial_interpreter = SpatialChain(
            verbose=verbose, video_id="4LdIvyfzoGY", interval=10
        )
        self.set_parameters_interpreter = None

    def set_video(self, video_id, interval):
        self.temporal_interpreter.set_video(video_id, interval)

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.temporal_interpreter.set_parameters(top_k, neighbors_left, neighbors_right)
        self.parameters_interpreter.set_parameters(top_k, neighbors_left, neighbors_right)

    def predict_spatial_locations(self, 
        spatial, spatial_labels,
        edits, sketches, video_shape,
    ):
        for edit in edits:
            start = timecode_to_seconds(edit["temporalParameters"]["start"])
            finish = timecode_to_seconds(edit["temporalParameters"]["finish"])
            spatial_position_set = False
            
            ### check if sketch is available
            for sketch in sketches:
                if sketch["timestamp"] >= start and sketch["timestamp"] < finish:
                    edit["spatialParameters"] = {
                        "x": round(sketch["x"]),
                        "y": round(sketch["y"]),
                        "width": round(sketch["width"]),
                        "height": round(sketch["height"]),
                        "rotation": 0,
                        "info": ["sketch"],
                        "source": ["sketch"],
                    }
                    spatial_position_set = True
                    break
            if spatial_position_set == True:
                continue

            candidates = [{
                "x": 0,
                "y": 0,
                "width": video_shape[1],
                "height": video_shape[0],
                "rotation": 0,
                "info": ["full"],
                "source": ["default"],
            }]

            for sketch in sketches:
                candidates.append({
                    "x": round(sketch["x"]),
                    "y": round(sketch["y"]),
                    "width": round(sketch["width"]),
                    "height": round(sketch["height"]),
                    "rotation": 0,
                    "info": ["sketch"],
                    "source": ["sketch"],
                })
            for reference, label in zip(spatial, spatial_labels):
                print("Spatial: ", reference, label)
                candidates = self.spatial_interpreter.run(
                    [reference], candidates, label,
                    start, finish,
                    video_shape,
                )
            
            if len(candidates) == 0:
                continue

            ### choose the one with the medium area (can also, 0 or last one)
            candidates.sort(key=lambda x: x["width"] * x["height"])
            print(candidates)
            edit["spatialParameters"] = candidates[(len(candidates) + 1) // 2 - 1]
        return edits

    def predict_edit_parameters(self,
        parameters,
        edits, sketches, video_shape,
    ):
        for i in range(len(edits)):
            edit = edits[i]
            edit_parameters = {
                "textParameters": edit["textParameters"],
                "imageParameters": edit["imageParameters"],
                "shapeParameters": edit["shapeParameters"],
                "blurParameters": edit["blurParameters"],
                "cutParameters": edit["cutParameters"],
                "cropParameters": edit["cropParameters"],
                "zoomParameters": edit["zoomParameters"],
            }

            run_all_parameters = True

            for j in range(i):
                prev_edit = edits[j]
                prev_edit_parameters = {
                    "textParameters": prev_edit["textParameters"],
                    "imageParameters": prev_edit["imageParameters"],
                    "shapeParameters": prev_edit["shapeParameters"],
                    "blurParameters": prev_edit["blurParameters"],
                    "cutParameters": prev_edit["cutParameters"],
                    "cropParameters": prev_edit["cropParameters"],
                    "zoomParameters": prev_edit["zoomParameters"],
                }
                if are_same_objects(edit_parameters, prev_edit_parameters):
                    edit_parameters = prev_edit_parameters
                    run_all_parameters = False
                    break
            
            if run_all_parameters == True:
                edit_parameters = self.parameters_interpreter.run_all_parameters(
                    parameters, edit_parameters,
                    video_shape
                )

            start = timecode_to_seconds(edit["temporalParameters"]["start"])
            finish = timecode_to_seconds(edit["temporalParameters"]["finish"])
            edit_parameters = self.parameters_interpreter.run_text_content(
                parameters,
                edit_parameters,
                start, finish,
            )
            edit_parameters = self.parameters_interpreter.run_image_query(
                parameters,
                edit_parameters,
                start, finish,
            )

            edit["textParameters"] = edit_parameters["textParameters"]
            edit["imageParameters"] = edit_parameters["imageParameters"]
            edit["shapeParameters"] = edit_parameters["shapeParameters"]
            edit["blurParameters"] = edit_parameters["blurParameters"]
            edit["cutParameters"] = edit_parameters["cutParameters"]
            edit["cropParameters"] = edit_parameters["cropParameters"]
            edit["zoomParameters"] = edit_parameters["zoomParameters"]

        return edits

    def __get_non_intersecting_segment(self, start, finish, segments):
        while len(segments) > 0:
            no_intersection = True
            for segment in segments:
                s_start = timecode_to_seconds(segment["start"])
                s_finish = timecode_to_seconds(segment["finish"])
                if s_start <= start < s_finish:
                    start = s_finish
                    finish = start + 10
                    no_intersection = False
                elif start < s_start < finish:
                    finish = s_start
            if no_intersection:
                break
        return start, finish

    def predict_temporal_segments(self,
        temporal,
        temporal_labels,
        current_player_position,
        sketch_timestamp,
        video_shape,
        skipped_segments,
    ):
        segments = []
        for reference, label in zip(temporal, temporal_labels):
            partial_segments = self.temporal_interpreter.run(
                [reference], label, 
                current_player_position, skipped_segments
            )
            segments.extend(partial_segments)
        temporal_segments = merge_segments(segments)
        if len(temporal_segments) == 0 and len(temporal) == 0:
            #### if no temporal reference is found, use default 10 seconds from current player position
            start, finish = self.__get_non_intersecting_segment(
                current_player_position, current_player_position + 10, skipped_segments
            )
            temporal_segments.append({
                "start": start,
                "finish": finish,
                "explanation": ["current_play_position"],
                "source": ["default"],
            })
        
        ### check if sketch_timestamp is in any of the edit segments
        if sketch_timestamp >= 0:
            sketch_covered = False
            for segment in temporal_segments:
                start = timecode_to_seconds(segment["start"])
                finish = timecode_to_seconds(segment["finish"])
                if start <= sketch_timestamp < finish:
                    sketch_covered = True
                    break
            if sketch_covered == False:
                start, finish = self.__get_non_intersecting_segment(
                    sketch_timestamp, sketch_timestamp + 10, skipped_segments + temporal_segments
                )
                temporal_segments.append({
                    "start": start,
                    "finish": finish,
                    "explanation": ["sketch_timestamp"],
                    "source": ["sketch"],
                })
        
        edit_segments = []
        for segment in temporal_segments:
            start = timecode_to_seconds(segment["start"])
            finish = timecode_to_seconds(segment["finish"])
            explanation = segment["explanation"]
            source = segment["source"]
            edit = get_edit_segment(start, finish, explanation, source, video_shape)
            edit_segments.append(edit)        

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
        response = copy.deepcopy(request)

        command = response["requestParameters"]["text"]
        skipped_segments = self.build_skipped_segments(response)
        prev_edits = response["edits"]
        current_player_position = response["curPlayPosition"]
        sketches = response["requestParameters"]["sketchRectangles"]
        sketch_timestamp = response["requestParameters"]["sketchFrameTimestamp"]
        for sketch in sketches:
            sketch["timestamp"] = sketch_timestamp
        print(sketches)

        video_shape = [response["projectMetadata"]["height"], response["projectMetadata"]["width"]]

        from_scratch = response["requestParameters"]["processingMode"] == "from-scratch"
        add_more = response["requestParameters"]["processingMode"] == "add-more"
        adjust_selected = response["requestParameters"]["processingMode"] == "adjust-selected"

        if from_scratch == False and add_more == False and adjust_selected == False:
            print("ERROR: Invalid processing mode")
            response["edits"] = []
            response["requestParameters"]["editOperations"] = []
            response["requestParameters"]["parameters"] = {}
            return response
        
        ### parse command
        references = self.input_parser.run(command)
        print(references)

        ### set edit operations
        response["requestParameters"]["editOperations"] = references.edit
        response["requestParameters"]["parameters"] = references.get_parameters_short()
        
        ### predict temporal segments
        if from_scratch == True or add_more == True:
            edits = self.predict_temporal_segments(
                references.temporal, references.temporal_labels, 
                current_player_position, sketch_timestamp,
                video_shape, skipped_segments, 
            )
        else:
            edits = prev_edits

        ### predict spatial positions
        edits = self.predict_spatial_locations(
            references.spatial, references.spatial_labels,
            edits, sketches, video_shape,
        )

        ### predict edit parameters
        edits = self.predict_edit_parameters(
            references.get_parameters(),
            edits, sketches, video_shape,
        )   

        response["edits"] = edits
        print(edits)
        return response









