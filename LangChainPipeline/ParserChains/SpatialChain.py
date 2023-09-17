import ast

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
        self.set_video(video_id, interval)
        self.position = SpatialPositionChain(verbose)
        self.image_processor = ImageProcessor()
        
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized SpatialChain")

    def set_video(self, video_id, interval):
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

    def process_visual_command(self, command, frame_sec, video_shape):
        input_images, input_bboxes, frame_id, img = self.image_processor.get_candidates_from_frame(frame_sec)
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
            "source": command,
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
            }
        else:
            return {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
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
        }

    def run(self, 
        command, candidates, label,
        start, finish,
        video_shape,
    ):
        new_candidates = []
        if label == "position":
            context = "height = " + str(video_shape[0]) + ", width = " + str(video_shape[1])
            for candidate in candidates:
                new_candidates.append(self.position.run(
                    context,
                    command,
                    candidate,
                ))
        elif label == "visual":
            refining_candidates = [self.process_visual_command(command, int((start + finish) // 2), video_shape)]
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
                    else:
                        candidate["x"] = round(intersection["x"])
                        candidate["y"] = round(intersection["y"])
                        candidate["width"] = round(intersection["width"])
                        candidate["height"] = round(intersection["height"])
                new_candidates.append(candidate)
        else:
            print("ERROR: label not recognized")
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

    def run(self, context, command, candidate):
        result = self.chain.predict(
            context=context,
            command=command,
            rectangle={
                "x": candidate["x"],
                "y": candidate["y"],
                "width": candidate["width"],
                "height": candidate["height"],
                "rotation": 0,
            },
        )

        candidate["x"] = result.x
        candidate["y"] = result.y
        candidate["width"] = result.width
        candidate["height"] = result.height
        candidate["rotation"] = result.rotation
        return candidate