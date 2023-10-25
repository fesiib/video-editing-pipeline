

def combine_dicts(dict1, dict2):
    combined_dict = {}
    for key in dict1:
        combined_dict[key] = dict1[key]
    for key in dict2:
        if key not in combined_dict:
            if isinstance(dict2[key], dict):
                combined_dict[key] = combine_dicts({}, dict2[key])
            else:
                combined_dict[key] = dict2[key]
                combined_dict[key + "_str"] = str(dict2[key])
        else:
            if isinstance(dict2[key], dict):
                combined_dict[key] = combine_dicts(combined_dict[key], dict2[key])
            else:
                if key + "_str" not in combined_dict:
                    combined_dict[key + "_str"] = str(combined_dict[key]) + " + " + str(dict2[key])
                else:
                    combined_dict[key + "_str"] += " + " + str(dict2[key])
                combined_dict[key] += dict2[key]
    return combined_dict