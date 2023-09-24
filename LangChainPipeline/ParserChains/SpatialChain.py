import ast
import json
import cv2
import random

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.chains import LLMChain

from LangChainPipeline.PromptTemplates.spatial_position_prompt import get_spatial_position_prompt_chat as get_spatial_position_prompt

from LangChainPipeline.PydanticClasses.Rectangle import Rectangle

from backend.image_processor import ImageProcessor

class SpatialChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
        video_id="4LdIvyfzoGY",
        interval=10
    ):
        self.visual_metadata = None
        self.transcript_metadata = None
        self.interval = interval
        self.video_id = video_id
        self.set_video(video_id, interval)
        self.position = SpatialPositionChain(verbose)
        self.image_processor = ImageProcessor()
        
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized SpatialChain")

    def set_video(self, video_id, interval):
        self.video_id = video_id
        metadata_filepath = f"metadata/{video_id}_{str(interval)}.txt"
        self.visual_metadata = []
        self.transcript_metadata = []
        with open(metadata_filepath) as f:
            raw_lines = f.readlines()
            for line in raw_lines:
                interval = ast.literal_eval(line.rstrip())
                visual_data = {
                    "action": interval["action_pred"],
                    "abstract_caption": interval["synth_caption"],
                    "dense_caption": interval["dense_caption"],
                }
                visual_data_str = (interval["synth_caption"].strip() + ", " 
                    + interval["dense_caption"].strip() + ", " 
                    + interval["action_pred"].strip())
                transcript = interval["transcript"].strip()
                self.visual_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "data": visual_data_str,
                    "structured_data": visual_data,
                })
                self.transcript_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "data": transcript,
                })
        print("Set video SpatialChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        # self.transcript.set_parameters(top_k, neighbors_left, neighbors_right)
        # self.visual.set_parameters(top_k, neighbors_left, neighbors_right)
        print("Set parameters SpatialChain")

    def process_visual_command(self, command, frame_sec, video_shape, offsets):
        input_images, input_bboxes, frame_id, img = self.image_processor.get_candidates_from_frame(frame_sec, self.video_id)
        bbox = self.image_processor.extract_related_crop(
            command[0],
            input_bboxes,
            input_images,
            frame_id,
            img
        )
        image_shape = img.shape
        candidate = {
            "x": bbox[0] / image_shape[1] * video_shape[1], 
            "y": bbox[1] / image_shape[0] * video_shape[0], 
            "width": bbox[2] / image_shape[1] * video_shape[1], 
            "height": bbox[3] / image_shape[0] * video_shape[0], 
            "rotation": 0,
            "source": command.copy(),
            "offsets": offsets.copy(),
        }
        return candidate

    def get_intersection(self, candidate1, candidate2):
        x1 = max(candidate1["x"], candidate2["x"])
        y1 = max(candidate1["y"], candidate2["y"])
        x2 = min(candidate1["x"] + candidate1["width"], candidate2["x"] + candidate2["width"])
        y2 = min(candidate1["y"] + candidate1["height"], candidate2["y"] + candidate2["height"])
        if x2 <= x1 or y2 <= y1:
            return {
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "info": ["intersection"],
            }
        else:
            return {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "info": ["intersection"],
            }
    
    def get_union(self, candidate1, candidate2):
        x1 = min(candidate1["x"], candidate2["x"])
        y1 = min(candidate1["y"], candidate2["y"])
        x2 = max(candidate1["x"] + candidate1["width"], candidate2["x"] + candidate2["width"])
        y2 = max(candidate1["y"] + candidate1["height"], candidate2["y"] + candidate2["height"])
        return {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "info": ["union"],
        }

    def get_candidate_bboxes(self,
        start, finish,
        commands, labels, offsets,
        ref_timestamp, ref_bboxes,
        video_shape,
    ):
        ### if no visual-dependent & no sketches -> return full frame as candidate
        ### if no visual-dependent -> return sketches as candidates
        ### if visual-dependent -> return visual-dependent+sketches x segmentations (sum or argmax) as candidate
        ### if visual-dependent but no sketches -> return visual-dependent x segmentations (sum or argmax) as candidate
        vs_texts = []
        vs_offsets = []
        for i in range(len(commands)):
            if labels[i] == "visual-dependent":
                vs_texts.append(commands[i])
                vs_offsets.append(offsets[i])
        if len(vs_texts) == 0:
            if len(ref_bboxes) > 0:
                ### return sketches as candidates
                return [
                    {
                        "x": bbox["x"], 
                        "y": bbox["y"], 
                        "width": bbox["width"], 
                        "height": bbox["height"], 
                        "rotation": 0,
                        "info": ["sketch"],
                        "source": ["sketch"],
                        "offsets": [-1],
                    } for bbox in ref_bboxes
                ]
            else:
                ### return full frame as candidate
                return [
                    {
                        "x": 0, 
                        "y": 0,
                        "width": video_shape[1],
                        "height": video_shape[0],
                        "rotation": 0,
                        "info": ["full_frame"],
                        "source": ["full_frame"],
                        "offsets": [-1],
                    }
                ]
        if len(ref_bboxes) > 0:
            filename_prefix = "{}_{}".format(self.video_id, str(random.randint(0, 100000)))
            _, _, _, image = self.image_processor.get_candidates_from_frame(int(ref_timestamp), self.video_id)
            scaled_ref_bboxes = []
            for bbox in ref_bboxes:
                scaled_bbox = [
                    int(bbox['x'] / video_shape[1] * image.shape[1]),
                    int(bbox['y'] / video_shape[0] * image.shape[0]),
                    int(bbox['width'] / video_shape[1] * image.shape[1]),
                    int(bbox['height'] / video_shape[0] * image.shape[0]),
                ]
                scaled_ref_bboxes.append(scaled_bbox)
            output_ref_image = image.copy()
            for bbox in scaled_ref_bboxes:
                cv2.rectangle(output_ref_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
            cv2.imwrite("./images/{}_!ref_image.jpg".format(filename_prefix), output_ref_image)
            ref_image = image.copy()
            
            frame_sec = int(start + finish) // 2
            (
                _, segmentation_bboxes, _, image
            ) = self.image_processor.get_candidates_from_frame(frame_sec, self.video_id)

            (
                candidate_bbox_argmax,
                candidate_bbox_sum,
            ) = self.image_processor.get_candidate_ref_text_single(
                image.copy(),
                ref_image.copy(), segmentation_bboxes, vs_texts, scaled_ref_bboxes
            )
            final_candidate = candidate_bbox_sum
            return [{
                "x": final_candidate[0] / image.shape[1] * video_shape[1],
                "y": final_candidate[1] / image.shape[0] * video_shape[0],
                "width": final_candidate[2] / image.shape[1] * video_shape[1],
                "height": final_candidate[3] / image.shape[0] * video_shape[0],
                "rotation": 0,
                "info": ["visual-dependent"],
                "source": vs_texts,
                "offsets": vs_offsets,
            }]
        else:
            frame_sec = int(start + finish) // 2
            (
                _, segmentation_bboxes, _, image
            ) = self.image_processor.get_candidates_from_frame(frame_sec, self.video_id)
            (
                candidate_bbox_argmax,
                candidate_bbox_sum,
            ) = self.image_processor.get_candidate_text(
                image.copy(),
                segmentation_bboxes, vs_texts
            )
            final_candidate = candidate_bbox_sum
            return [{
                "x": final_candidate[0] / image.shape[1] * video_shape[1],
                "y": final_candidate[1] / image.shape[0] * video_shape[0],
                "width": final_candidate[2] / image.shape[1] * video_shape[1],
                "height": final_candidate[3] / image.shape[0] * video_shape[0],
                "rotation": 0,
                "info": ["visual-dependent"],
                "source": vs_texts,
                "offsets": vs_offsets,
            }]

    def run(self,
        original_command, 
        command, candidates, label,
        offsets,
        start, finish,
        video_shape,
    ):
        new_candidates = []
        if label == "independent":
            # left top
            context = [f"Frame Size: height: {str(video_shape[0])}, width: {str(video_shape[1])}",
                f'The original command was: {original_command}',
            ]
            for candidate in candidates:
                new_candidates.append(self.position.run(
                    context,
                    command,
                    candidate,
                    offsets,
                ))
        elif label == "visual-dependent":
            refining_candidates = [self.process_visual_command(command, int((start + finish) // 2), video_shape, offsets)]
            for candidate in candidates:
                for refinement in refining_candidates:
                    ### if there is intersection, then take the intersection
                    ### otherwise, take the union
                    intersection = self.get_intersection(candidate, refinement)
                    if intersection["width"] * intersection["height"] < 0.5 * refinement["width"] * refinement["height"]:
                        union = self.get_union(candidate, refinement)
                        if union["width"] * union["height"] > 2 * candidate["width"] * candidate["height"]:
                            continue
                        else:
                            candidate["x"] = round(union["x"])
                            candidate["y"] = round(union["y"])
                            candidate["width"] = round(union["width"])
                            candidate["height"] = round(union["height"])
                            candidate["info"].extend(union["info"])
                            candidate["source"].extend(refinement["source"])
                            candidate["offsets"].extend(refinement["offsets"])
                    else:
                        candidate["x"] = round(intersection["x"])
                        candidate["y"] = round(intersection["y"])
                        candidate["width"] = round(intersection["width"])
                        candidate["height"] = round(intersection["height"])
                        candidate["info"].extend(intersection["info"])
                        candidate["source"].extend(refinement["source"])
                        candidate["offsets"].extend(refinement["offsets"])
                new_candidates.append(candidate)
        elif label == "other":
            print("\"other\" label detected", command)
        else:
            print("ERROR: label not recognized", label)
            new_candidates = candidates
        #elif label == "condition":
            ### foreground/background rotoscoping
        return new_candidates
        

class SpatialPositionChain():
    def __init__(
            self,
            verbose=False,
    ):
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")
        self.parser = PydanticOutputParser(pydantic_object=Rectangle)

        self.prompt_template = get_spatial_position_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
        )
        print("Initialized SpatialPositionChain")

    def run(self, context, command, candidate, offsets):
        result = self.chain.predict(
            context=json.dumps(context),
            command=json.dumps(command),
            rectangle=Rectangle.get_instance(
                x=candidate["x"],
                y=candidate["y"],
                width=candidate["width"],
                height=candidate["height"],
                rotation=0,
            ).model_dump_json()
        )

        candidate["x"] = result.x
        candidate["y"] = result.y
        candidate["width"] = result.width
        candidate["height"] = result.height
        candidate["rotation"] = result.rotation
        candidate["info"].append("gpt")
        candidate["source"] = candidate["source"].copy() + command.copy()
        candidate["offsets"] = candidate["offsets"].copy() + offsets.copy()
        return candidate