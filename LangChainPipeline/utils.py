def timecode_to_seconds(timecode):
    if isinstance(timecode, float) or isinstance(timecode, int):
        return timecode

    timecode = timecode.split(":")
    while len(timecode) < 3:
        timecode = ["0"] + timecode
    hours = float(timecode[0])
    minutes = float(timecode[1])
    seconds = float(timecode[2])
    return seconds + minutes * 60 + hours * 60 * 60

def merge_segments(segments):
    result = []
    segments.sort(key=lambda x: x["start"])
    for interval in segments:
        if len(result) == 0:
            result.append(interval)
        else:
            last_interval = result[-1]
            last_finish = timecode_to_seconds(last_interval["finish"])
            interval_start = timecode_to_seconds(interval["start"])
            if last_finish >= interval_start:
                last_interval["finish"] = interval["finish"]
                last_interval["explanation"].extend(interval["explanation"])
                last_interval["source"].extend(interval["source"])
                last_interval["offsets"].extend(interval["offsets"])
            else:
                result.append(interval)
    return result

def are_same_objects(obj1, obj2):
    if isinstance(obj1, list) and isinstance(obj2, list):
        if len(obj1) != len(obj2):
            return False
        for i in range(len(obj1)):
            if not are_same_objects(obj1[i], obj2[i]):
                return False
        return True
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        if len(obj1) != len(obj2):
            return False
        for key in obj1:
            if key not in obj2:
                return False
            if not are_same_objects(obj1[key], obj2[key]):
                return False
        return True
    else:
        return obj1 == obj2