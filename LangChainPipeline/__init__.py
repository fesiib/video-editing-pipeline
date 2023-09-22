import copy

from LangChainPipeline.ParserChains.IntentParserChain import IntentParserChain
from LangChainPipeline.ParserChains.IndexedIntentParserChain import IndexedIntentParserChain
from LangChainPipeline.ParserChains.TemporalChain import TemporalChain
from LangChainPipeline.ParserChains.EditChain import EditChain
from LangChainPipeline.ParserChains.SpatialChain import SpatialChain

from LangChainPipeline.utils import merge_segments, timecode_to_seconds, are_same_objects

from backend.operations import get_edit_segment

class LangChainPipeline():
    def __init__(self, verbose=False):
        self.input_parser = IntentParserChain(verbose=verbose)
        self.indexed_input_parser = IndexedIntentParserChain(verbose=verbose)

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
        command,
        spatial, spatial_labels,
        spatial_offsets,
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
                        "offsets": [-1],
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
                "offsets": [-1],
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
                    "offsets": [-1],
                })
            for reference, label, offsets in zip(spatial, spatial_labels, spatial_offsets):
                print("Spatial: ", reference, label)
                candidates = self.spatial_interpreter.run(
                    command,
                    [reference], candidates, label, [offsets],
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
        command,
        parameters,
        parameter_offsets,
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
                    command,
                    parameters, edit_parameters,
                    video_shape
                )

            start = timecode_to_seconds(edit["temporalParameters"]["start"])
            finish = timecode_to_seconds(edit["temporalParameters"]["finish"])
            edit_parameters = self.parameters_interpreter.run_text_content(
                command,
                parameters,
                edit_parameters,
                start, finish,
            )
            edit_parameters = self.parameters_interpreter.run_image_query(
                command,
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
        command,
        temporal,
        temporal_labels,
        temporal_offsets,
        current_player_position,
        sketch_timestamp,
        video_shape,
        skipped_segments,
    ):
        segments = []
        for reference, label, offsets in zip(temporal, temporal_labels, temporal_offsets):
            partial_segments = self.temporal_interpreter.run(
                command,
                [reference], label, [offsets],
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
                "offsets": [-1],
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
                    "offsets": [-1],
                })
        
        edit_segments = []
        for segment in temporal_segments:
            start = timecode_to_seconds(segment["start"])
            finish = timecode_to_seconds(segment["finish"])
            explanation = segment["explanation"]
            source = segment["source"]
            offsets = segment["offsets"]
            edit = get_edit_segment(start, finish, explanation, source, offsets, video_shape)
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
            response["requestParameters"]["indexedReferences"] = {}
            return response
        
        ### parse command
        references = self.input_parser.run(command)
        print(references)

        ### set edit operations
        response["requestParameters"]["editOperations"] = references.edit
        response["requestParameters"]["parameters"] = references.get_parameters_short()
        response["requestParameters"]["indexedReferences"] = {}
        
        ### predict temporal segments
        if from_scratch == True or add_more == True:
            edits = self.predict_temporal_segments(
                command,
                references.temporal, references.temporal_labels, [-1 for _ in references.temporal],
                current_player_position, sketch_timestamp,
                video_shape, skipped_segments, 
            )
        else:
            edits = prev_edits

        ### predict spatial positions
        edits = self.predict_spatial_locations(
            command,
            references.spatial, references.spatial_labels,
            edits, sketches, video_shape,
        )

        ### predict edit parameters
        edits = self.predict_edit_parameters(
            command,
            references.get_parameters(),
            {},
            edits, sketches, video_shape,
        )   

        response["edits"] = edits
        print(edits)
        return response



    def process_request_indexed(self, request):
        response = copy.deepcopy(request)

        command = response["requestParameters"]["text"]
        skipped_segments = self.build_skipped_segments(response)
        prev_edits = response["edits"]
        current_player_position = response["curPlayPosition"]
        sketches = response["requestParameters"]["sketchRectangles"]
        sketch_timestamp = response["requestParameters"]["sketchFrameTimestamp"]
        for sketch in sketches:
            sketch["timestamp"] = sketch_timestamp

        video_shape = [response["projectMetadata"]["height"], response["projectMetadata"]["width"]]

        from_scratch = response["requestParameters"]["processingMode"] == "from-scratch"
        add_more = response["requestParameters"]["processingMode"] == "add-more"
        adjust_selected = response["requestParameters"]["processingMode"] == "adjust-selected"

        if from_scratch == False and add_more == False and adjust_selected == False:
            print("ERROR: Invalid processing mode")
            response["edits"] = []
            response["requestParameters"]["editOperations"] = []
            response["requestParameters"]["parameters"] = {}
            response["requestParameters"]["indexedReferences"] = {}
            return response
        
        ### parse command
        references = self.indexed_input_parser.run(command)
        simple_references = references.get_references()
        print(references)

        ### set edit operations
        response["requestParameters"]["editOperations"] = simple_references.edit
        response["requestParameters"]["parameters"] = simple_references.get_parameters_short()
        response["requestParameters"]["indexedReferences"] = references.model_dump_json()

        
        ### predict temporal segments
        if from_scratch == True or add_more == True:
            edits = self.predict_temporal_segments(
                command,
                simple_references.temporal, simple_references.temporal_labels,
                [item.offset for item in references.temporal_references],
                current_player_position, sketch_timestamp,
                video_shape, skipped_segments, 
            )
        else:
            edits = prev_edits

        ### predict spatial positions
        edits = self.predict_spatial_locations(
            command,
            simple_references.spatial, simple_references.spatial_labels,
            [item.offset for item in references.spatial_references],
            edits, sketches, video_shape,
        )

        ### predict edit parameters TODO
        edits = self.predict_edit_parameters(
            command,
            simple_references.get_parameters(),
            {},
            edits, sketches, video_shape,
        )   

        response["edits"] = edits
        print(edits)
        return response





