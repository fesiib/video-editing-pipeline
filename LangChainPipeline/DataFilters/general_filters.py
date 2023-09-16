from LangChainPipeline.utils import timecode_to_seconds

def filter_metadata_skipped(metadata, skipped_segments):
    result = []
    for item in metadata:
        start = timecode_to_seconds(item["start"])
        end = timecode_to_seconds(item["end"])
        skip = False
        for segment in skipped_segments: 
            skipped_start = timecode_to_seconds(segment["start"])
            skipped_end = timecode_to_seconds(segment["finish"])        
            intersection_start = max(start, skipped_start)
            intersection_end = min(end, skipped_end)
            if intersection_end > intersection_start:
                skip = True
                break
        if (not skip):
            result.append(item)
    return result