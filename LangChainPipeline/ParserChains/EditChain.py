import ast

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.chains import LLMChain

from LangChainPipeline.PromptTemplates.all_parameters_prompt import get_all_parameters_prompt_chat as get_all_parameters_prompt
from LangChainPipeline.PromptTemplates.text_content_prompt import get_text_content_prompt_chat as get_text_content_prompt
from LangChainPipeline.PromptTemplates.image_query_prompt import get_image_query_prompt_chat as get_image_query_prompt

from LangChainPipeline.PydanticClasses.EditParameters import EditParameters

from LangChainPipeline.DataFilters.semantic_filters import filter_metadata_by_semantic_similarity
from LangChainPipeline.utils import timecode_to_seconds

class EditChain():
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
        
        self.all_parameters = AllParametersChain(
            verbose=verbose,
        )

        self.text_content = TextContentChain(
            verbose=verbose,
            top_k=top_k,
            neighbors_left=neighbors_left,
            neighbors_right=neighbors_right,
        )

        self.image_query = ImageQueryChain(
            verbose=verbose,
            top_k=top_k,
            neighbors_left=neighbors_left,
            neighbors_right=neighbors_right,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized EditChain")

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
        print("Set video")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        self.text_content.set_parameters(top_k, neighbors_left, neighbors_right)
        self.image_query.set_parameters(top_k, neighbors_left, neighbors_right)
        print("Set parameters")

    def run_all_parameters(self, 
        parameters, initial_edit_parameters,
        video_shape
    ):
        common_context = "Video Properties: height = " + str(video_shape[0]) + ", width = " + str(video_shape[1])

        total_references = 0
        for parameter in parameters:
            total_references += len(parameters[parameter])
        
        if total_references == 0:
            return initial_edit_parameters
        
        new_edit_parameters = self.all_parameters.run(common_context, parameters, initial_edit_parameters)
        return new_edit_parameters

    def run_text_content(
        self,
        initial_edit_parameters,
        start, finish,
    ):
        text_content_context = list(filter(lambda x: start <= timecode_to_seconds(x["start"]) < finish, self.transcript_metadata))
        if len(initial_edit_parameters["textParameters"]["content"]) > 0 and len(text_content_context) > 0:
            initial_edit_parameters["textParameters"]["content"] = self.text_content.run(
                text_content_context,
                [initial_edit_parameters["textParameters"]["content"]],
            )
        return initial_edit_parameters

    def run_image_query(self,
        initial_edit_parameters,
        start, finish,
    ):
        image_query_context = list(filter(lambda x: start <= timecode_to_seconds(x["start"]) < finish, self.visual_metadata))
        if len(initial_edit_parameters["imageParameters"]["searchQuery"]) > 0 and len(image_query_context) > 0:
            initial_edit_parameters["imageParameters"]["searchQuery"] = self.image_query.run(
                image_query_context,
                [initial_edit_parameters["imageParameters"]["searchQuery"]],
            )
        return initial_edit_parameters


class AllParametersChain():
    def __init__(
            self,
            verbose=False,
    ):
        self.skip_parameters = ["imageParameters", "cutParameters"]
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")
        self.parser = PydanticOutputParser(pydantic_object=EditParameters)

        self.prompt_template = get_all_parameters_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
        )
        print("Initialized AllParametersChain")

    def filter_parameters(self, parameters):
        filtered_parameters = {}
        for parameter in parameters:
            if parameter not in self.skip_parameters:
                filtered_parameters[parameter] = parameters[parameter]
        return filtered_parameters

    def run(self, context, parameters, initial_edit_parameters):
        result = self.chain.predict(
            context=context,
            command=self.filter_parameters(parameters),
            initial_parameters=self.filter_parameters(initial_edit_parameters),
        )
        dict_result = result.dict()
        for parameter in self.skip_parameters:
            dict_result[parameter] = initial_edit_parameters[parameter]
        return dict_result
    
class TextContentChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
    ):
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")

        self.prompt_template = get_text_content_prompt()

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized TextContentChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Set parameters")

    ### TODO: need to consider only metadata from the relevant segment
    def run(self, context, command):

        filtered_context = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=context,
            k=self.top_k,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )

        result = self.chain.predict(
            context=[data["data"] for data in filtered_context],
            command=command,
        )
        return result
    
class ImageQueryChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
    ):
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")

        self.prompt_template = get_image_query_prompt()

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized ImageQueryChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Set parameters")

    ### TODO: need to consider only metadata from the relevant segment
    def run(self, context, command):
        filtered_context = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=context,
            k=self.top_k,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )

        result = self.chain.predict(
            context=[data["structured_data"] for data in filtered_context],
            command=command,
        )
        return result