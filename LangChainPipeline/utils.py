def timecode_to_seconds(timecode):
    timecode = timecode.split(":")
    hours = int(timecode[0])
    minutes = int(timecode[1])
    seconds = int(timecode[2])
    return seconds + minutes * 60 + hours * 60 * 60

def merge_segments(segments):
    result = []
    segments.sort(key=lambda x: x["start"])
    for interval in segments:
        if len(result) == 0:
            result.append(interval)
        else:
            last_interval = result[-1]
            last_end = timecode_to_seconds(last_interval["end"])
            interval_start = timecode_to_seconds(interval["start"])
            if last_end >= interval_start:
                last_interval["end"] = interval["end"]
                last_interval["explanation"].extend(interval["explanation"])
                last_interval["source"].extend(interval["source"])
            else:
                result.append(interval)
    return result