from LangChainPipeline.ParserChains.IntentParserChain import IntentParserChain
from LangChainPipeline.ParserChains.TemporalChain import TemporalChain

from LangChainPipeline.utils import merge_segments

class LangChainPipeline():
    def __init__(self, verbose=False):
        self.input_parser = IntentParserChain(verbose=verbose)
        self.temporal_interpreter = TemporalChain(
            verbose=verbose, video_id="4LdIvyfzoGY", interval=10
        )
        self.spatial_interpreter = None
        self.set_parameters_interpreter = None

    def set_video(self, video_id, interval):
        self.temporal_interpreter.set_video(video_id, interval)

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.temporal_interpreter.set_parameters(top_k, neighbors_left, neighbors_right)

    def run(self, command, skipped_segments=[]):
        references = self.input_parser.run(command)
        
        print(references)
        
        temporal = references.temporal
        temporal_labels = references.temporal_labels

        segments = []

        for reference, label in zip(temporal, temporal_labels):
            partial_segments = self.temporal_interpreter.run([reference], label, skipped_segments)
            segments.extend(partial_segments)

        return merge_segments(segments)