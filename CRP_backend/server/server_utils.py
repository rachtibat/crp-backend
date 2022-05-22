
def extract_json_keys(json, session, keys):
    extracted_values = []
    optional = {"descending": True, "abs_norm": True, "x" : 0,
                "y" : 0, "width" : 1, "height" : 1, "view_prev": 0, "activations": 0, 
                "range": "0:8",  "weight_activation": 0, "n_classes": 5,
                "threshold": 0.2, "sigma": 1, "top_N": 3, "stats_mode": "activation_stats",
                "zero_list_filter": [], "zero_layer": "", "init_rel" : "activation",
                "parent_c_id": None, "parent_layer": None}

    for key in keys:

        if key in json:
            value = json[key]
        elif key in session:
            value = session[key]
        elif key in optional:
            value = optional[key]
        else:
            print(f"{key} not in JSON, Session or Optional! Error!")
            return 0

        if key == "mask_id" or key == "size" or key == "index" or key == "synthetic" \
                or key == "cnn_activation" or key == "view_prev" or key == "activations"  \
                    or key == "weight_activation" or key == "n_classes" or key == "filter_index" \
                    or key == "top_N" or key == "target" or key == "concept_id":
            value = int(value)

        elif key == "x" or key == "y" or key == "width" or key == "height" or key == "threshold" or key == "sigma":
            value = float(value)

        elif key == "range":
            first, last = value.split(":")
            first, last = map(int, (first, last))
            value = (first, last)

        extracted_values.append(value)

    return extracted_values
