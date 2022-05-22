from flask import Flask, redirect, render_template, request, session, url_for, send_file, jsonify, Response
from datetime import timedelta

from flask_socketio import SocketIO
from flask_socketio import join_room, leave_room
import datetime
import time
import json

import matplotlib.pyplot as plt
import numpy as np

from CRP_backend.server.server_utils import *
from CRP_backend.core.Experiment import edit_config

METHODS = ["epsilon_plus_flat", "epsilon_plus",
           "all_epsilon", "alpha_beta_plus_flat"]
EXPERIMENTS = {}
user_room = {}


def home():
    # return render_template("test_website.html")
    return render_template("test_socket.html")


def connected(auth):
    print(datetime.datetime.now(), " Client connected.")

   # user_id = request.sid
   # clients.append(request.sid)
   # room = session.get('room')
   # join_room(room)


def disconnected():
    print(datetime.datetime.now(), " Client disconnected")


def global_analysis(json_response, mode, json_mode):

    keys = ["image_index", "experiment", "method", "layer", "sorting",
            "target_class", "filter_indices", "zero_list_filter", "zero_layer"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        image_index, experiment, method, layer, sorting, \
            target_class, filter_indices, zero_list_filter, zero_layer = key_values

    first_filter, last_filter = filter_indices
    EXP = EXPERIMENTS[experiment]

    if layer not in EXP.layer_analyzed:
        print(f"layer {layer} for {experiment} not supported")
        return f"layer {layer} for {experiment} not supported", 404

    start_time = time.time()

    target = EXP.DMI.decode_class_name(target_class)
    image_tensor, _ = EXP.DMI.get_data_sample(image_index, preprocessing=True)

    EXP.XAI.set_zero_hook(zero_layer, zero_list_filter)

    sorted_ch_indices, sorted_rel_relevance = EXP.XAI.find_relevant_in_sample(image_tensor, image_index, layer, method, target,
                                                                              selected=(first_filter, last_filter), sorting=sorting)

    EXP.XAI.remove_zero_hook()

    sorted_ch_indices = sorted_ch_indices.tolist()
    #filter_names = get_names_from_cookie(session, layer, sorted_ch_indices)

    meta_data = {
        "filter_indices": sorted_ch_indices,
        "relevance": {f: sorted_rel_relevance[i] for i, f in enumerate(sorted_ch_indices)},
        "filter_names": {},
        "mode": mode
    }
    socketio.emit("receive_global_analysis", meta_data, to=request.sid)

    print("time elapsed for global analysis", time.time() - start_time)

    if mode == "max_activation" or mode == "max_relevance" or mode == "max_relevance_target":
        vis_realistic({
            "experiment": experiment,
            "list_filter": sorted_ch_indices,
            "layer": layer,
            "size": json_mode["size"],
            "sample_indices": json_mode["sample_indices"],
            "mode": mode,
            "target_class": target_class,
        })
    elif mode == "synthetic":
        vis_synthetic({
            "experiment": experiment,
            "list_filter": sorted_ch_indices,
            "layer": layer,
            "size": json_mode["size"]
        })


def vis_synthetic(json_response, attr_graph=False):

    keys = ["experiment", "size", "layer", "list_filter"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, list_filter = key_values

    EXP = EXPERIMENTS[experiment]
    if not EXP.GA.available:
        return f"Not defined for {experiment}", 404

    for i, (binary_dict, e) in enumerate(gen_load_and_encode_synthetic(EXP, list_filter, layer, size)):

        meta_data = {
            "step_percent": e,
            "layer": layer,
        }
        if not attr_graph:
            socketio.emit("receive_synthetic",
                          (binary_dict, meta_data), to=request.sid)
        else:
            socketio.emit("receive_synthetic_graph",
                          (binary_dict, meta_data), to=request.sid)


def vis_realistic(json_response, attr_graph=False):

    keys = ["experiment", "size", "layer", "list_filter",
            "sample_indices", "mode", "target_class"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, list_filter, sample_indices, mode, target_class = key_values

    first_sample, last_sample = sample_indices
    EXP = EXPERIMENTS[experiment]

    for i, binary_dict in enumerate(gen_load_and_encode_real(
            EXP, list_filter, layer, size, first_sample, last_sample, mode=mode, target_class=target_class, codec="png")):

        meta_data = {
            "filter_index": list_filter[i],
            "layer": layer,
            "mode": mode,
        }
        if not attr_graph:
            socketio.emit("receive_realistic",
                          (binary_dict, meta_data), to=request.sid)
        else:
            socketio.emit("receive_realistic_graph",
                          (binary_dict, meta_data), to=request.sid)


def vis_realistic_heatmaps(json_response):

    keys = ["experiment", "size", "layer", "filter_index",
            "sample_indices", "mode", "target_class", "method"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, filter_index, sample_indices, mode, target_class, method = key_values

    first_sample, last_sample = sample_indices
    EXP = EXPERIMENTS[experiment]

    binary_dict = load_and_encode_realistic_heatmaps(
        EXP, filter_index, layer, size, first_sample, last_sample, method, mode=mode, target_class=target_class)

    meta_data = {
        "filter_index": filter_index,
        "mode": mode
    }
    socketio.emit("receive_example_heatmaps",
                  (binary_dict, meta_data), to=request.sid)


def vis_statistics(json_response):

    keys = ["experiment", "size", "layer",
            "filter_index", "sample_indices", "n_classes", "stats_mode"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, filter_index, sample_indices, n_classes, stats_mode = key_values
        first_sample, last_sample = sample_indices

    EXP = EXPERIMENTS[experiment]

    sorted_names, sorted_rel_class, d_indices, n_indices, r_value = \
        EXP.XAI.get_statistics(layer, filter_index, n_classes, stats_mode)

    sorted_names = sorted_names.tolist()

    for i, binary_dict in enumerate(gen_load_and_encode_statistics(EXP, filter_index, d_indices, n_indices, layer, size, first_sample, last_sample)):

        meta_data = {
            "class_name": sorted_names[i],
            "class_rel": sorted_rel_class[i],
            "data_rel": r_value[i],
            "filter_index": filter_index
        }
        socketio.emit("receive_statistics",
                      (binary_dict, meta_data), to=request.sid)


def vis_statistics_heatmaps(json_response):

    keys = ["experiment", "size", "layer", "method",
            "filter_index", "n_classes", "stats_mode", "sample_indices"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, method, filter_index, n_classes, stats_mode, sample_indices = key_values
        first_sample, last_sample = sample_indices

    EXP = EXPERIMENTS[experiment]

    sorted_names, sorted_rel_class, most_d_indices, most_n_indices, r_value = \
        EXP.XAI.get_statistics(layer, filter_index, n_classes, stats_mode)

    sorted_names = sorted_names.tolist()

    if stats_mode == "activation_stats":
        weight_activation = True
    if stats_mode == "relevance_stats":
        weight_activation = False

    # BUG:add multi target support by übergeben von zugehörigen sorted_names

    for i in range(len(sorted_names)):

        d_indices = most_d_indices[i][first_sample:last_sample, filter_index]
        n_indices = most_n_indices[i][first_sample:last_sample, filter_index]
        binary_dict = load_and_encode_example_heatmaps(EXP, filter_index, layer, size, d_indices,
                                                       n_indices, method, weight_activation)

        meta_data = {
            "class_name": sorted_names[i],
            "class_rel": sorted_rel_class[i],
            "data_rel": r_value[i],
            "filter_index": filter_index
        }
        socketio.emit("receive_statistics_heatmaps",
                      (binary_dict, meta_data), to=request.sid)


def get_XAI_available():
    """
    Send:
    ---
    Nothing

    Return:
    ------
    JSON {"experiments" : array of strings, "methods" : array of strings, "layers" : 2d dictionary of strings,
     "cnn_layers" : 2d dictionary of 0/1, synthetic, max_index}

    methods = {method1, method2 ...}
    layers = { experiment_1 : [layer1, layer2, ...], experiment_2 : [layer1, layer2, ...] ...}
    max_index = {experiment_1 : int, experiment_2: int ...}
    class_to_indices = {experiment_1: {class_name1: [int, int ,int..], class_name2: [int, int int...] ..}, experiment2: ...}
    layer_modes = {experiment_1: {layer1: {
                "max_activation" : 1,
                "max_relevance" : 0,
                "max_relevance_target": 0,
                "relevance_stats" : 0,
                "activation_stats" : 0,
                "cnn_activation": 0,
                "synthetic": 0
            }, layer2 : { ....} }, experiment_2: {...}}
    targets_names = [name1, name2, ..]
    """

    layers = {}
    layer_modes = {}
    max_index = {}
    class_to_indices, target_names = {}, {}
    for key in EXPERIMENTS:

        layers[key] = EXPERIMENTS[key].layer_analyzed
        layer_modes[key] = EXPERIMENTS[key].layer_modes

        class_to_indices[key] = EXPERIMENTS[key].class_to_indices
        target_names[key] = EXPERIMENTS[key].DMI.get_all_classes()

        max_index[key] = len(EXPERIMENTS[key].DMI.dataset) - 1

    dict_response = {
        "experiments": list(EXPERIMENTS.keys()), "methods": METHODS, "layers": layers, "layer_modes": layer_modes,
        "max_index": max_index, "class_to_indices": class_to_indices, "target_names": target_names
    }

    socketio.emit("receive_XAI_available", dict_response, to=request.sid)


def select_datapoint(json_response):
    """
    Return:
        ground_truth: list of strings
    """

    keys = ["image_index", "experiment", "size"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        image_index, experiment, size = key_values

    session["image_index"] = image_index  # save as cookie #TODO: delete
    session["experiment"] = experiment

    EXP = EXPERIMENTS[experiment]

    image_tensor, target = EXP.DMI.get_data_sample(
        image_index, preprocessing=False)
    image_np = image_tensor.detach().cpu().numpy()[0]
    loaded_image = EXP.DMI.visualize_data_sample(image_np, size, image_index)
    encoded_image = encode_image_to_binary(loaded_image)

    class_names = []
    try:
        single_targets = EXP.DMI.multitarget_to_single(target)
        for s_t in single_targets:
            name = EXP.DMI.decode_target(s_t)
            class_names.append(name)

    except NotImplementedError:
        name = EXP.DMI.decode_target(target)
        class_names.append(name)

    # default target if no selected saved in cookie
    default_t = EXP.DMI.select_standard_target(target)
    session["target_class"] = EXP.DMI.decode_target(default_t)

    meta_data = {"image": encoded_image, "image_index": image_index, "ground_truth": class_names,
                 "default_target_name": session["target_class"]}
    socketio.emit("receive_data", meta_data, to=request.sid)


def get_heatmap(json_response):
    """
    send { experiment : string, image_index : integer , method : string, size : integer, N_pred: int, target_class: str}
    :return: jsonify({ "image" : b64 string, "image_index" : integer, "prediction": string, "accuracy" : integer})

    """
    keys = ["image_index", "experiment", "size",
            "method", "N_pred", "target_class", "zero_list_filter", "zero_layer"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        image_index, experiment, size, method, N_pred, target_class, \
            zero_list_filter, zero_layer = key_values

    session["method"] = method  # save as cookie

    EXP = EXPERIMENTS[experiment]

    image_tensor, _ = EXP.DMI.get_data_sample(image_index, preprocessing=True)

    target = EXP.DMI.decode_class_name(target_class)

    EXP.XAI.set_zero_hook(zero_layer, zero_list_filter)
    inp_attrib, relevances, pred, _ = EXP.XAI.calc_basic_attribution(
        image_tensor, method, image_index, target)
    EXP.XAI.remove_zero_hook()

    # TODO merge both lines in visualize_heatmap
    heatmap = EXP.DMI.visualize_heatmap(inp_attrib, size, image_index)
    encoded_image = encode_image_to_binary(heatmap)
    # EXP.IC.save_heatmap(encoded_image, image_index,
    #                   size, method, target_class, pred)
    l_rel = {}
    for l_key in relevances:
        l_rel[l_key] = str(np.sum(relevances[l_key]))

    dec_pred = EXP.DMI.decode_pred(pred, N_pred)

    # TODO: delete tolist()
    socketio.emit("receive_heatmap", {"image": encoded_image, "image_index": image_index, "wrt": target_class,
                                      "pred_classes": dec_pred[0], "pred_confidences": dec_pred[1].tolist(), "rel_layer": l_rel}, to=request.sid)

    torch.cuda.empty_cache()  # TODO: remove?


def local_analysis(json_response, mode, json_mode):
    """
    Send
    ----
    JSON.stringify({x : int, y : int, width : int, height : int, size : int, mask_id : int, target_class: str})
    size optional
    if mask_id exists, parameters x,y,width and height are redundant

    Return
    ------
    filters like <global_analysis>
    """

    keys = ["image_index", "experiment", "method", "layer", "sorting", "target_class",
            "filter_indices", "x", "y", "width", "height"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        image_index, experiment, method, layer, sorting, target_class, \
            filter_indices, x, y, width, height = key_values

    first_filter, last_filter = filter_indices

    EXP = EXPERIMENTS[experiment]

    if layer not in EXP.layer_analyzed:
        print(f"layer {layer} for {experiment} not supported")
        return f"layer {layer} for {experiment} not supported", 404
    # if not EXP.layer_modes[layer][mode]:
     #   print(f"{mode} mode keyword for {experiment} not supported")
      #  return f"{mode} mode keyword for {experiment} not supported", 404

    start_time = time.time()

    target = EXP.DMI.decode_class_name(target_class)
    image_tensor, _ = EXP.DMI.get_data_sample(image_index, preprocessing=True)

    mask = EXP.DMI.generate_input_attr_mask(image_tensor, x, y, width, height)

    sorted_ch_indices, sorted_rel_relevance = EXP.XAI.find_relevant_in_region(image_tensor, image_index, mask, layer, method, target,
                                                                              (first_filter, last_filter), sorting)
    sorted_ch_indices = sorted_ch_indices.tolist()
    #filter_names = get_names_from_cookie(session, layer, sorted_ch_indices)

    meta_data = {
        "filter_indices": sorted_ch_indices,
        "relevance": {f: sorted_rel_relevance[i] for i, f in enumerate(sorted_ch_indices)},
        "filter_names": {}
    }
    socketio.emit("receive_local_analysis", meta_data, to=request.sid)

    print("time elapsed for local analysis", time.time() - start_time)

    if mode == "max_activation" or mode == "max_relevance_target":
        vis_realistic({
            "experiment": experiment,
            "list_filter": sorted_ch_indices,
            "layer": layer,
            "size": json_mode["size"],
            "sample_indices": json_mode["sample_indices"],
            "mode": mode,
        })
    elif mode == "synthetic":
        vis_synthetic({
            "experiment": experiment,
            "list_filter": sorted_ch_indices,
            "layer": layer,
            "size": json_mode["size"]
        })

#TODO: delete


def get_local_segments():
    """
    Send
    ----
    { threshold: float between 0 and 1, sigma: float}

    Return
    ------
    masks = { id : image as string base64}

    """

    json_response = request.get_json()
    keys = ["image_index", "experiment",
            "size", "method", "threshold", "sigma", "target_class"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        image_index, experiment, size, method, threshold, sigma, target_class = key_values

    EXP = EXPERIMENTS[experiment]
    target = EXP.DMI.decode_class_name(target_class)
    image_tensor, _ = EXP.DMI.get_data_sample(image_index, preprocessing=True)

    all_masks = EXP.XAI.calc_watershed(
        image_tensor, image_index, method, target, threshold=threshold, sigma=sigma)

    image_tensor, _ = EXP.DMI.get_data_sample(image_index, preprocessing=False)
    masked_imgs = image_tensor.detach().cpu().numpy() * all_masks[:, None, ...]

    send_images = {}
    for i, m_img in enumerate(masked_imgs):
        img = EXP.DMI.visualize_data_sample(m_img, size)
        send_images[i] = encode_image_to_base64(img)

    return jsonify(masks=send_images)


def get_attribution_graph(json_response, mode, json_mode):

    keys = ["image_index", "experiment", "method",
            "layer", "filter_index", "size", "view_prev", "target_class",
            "weight_activation"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        image_index, experiment, method, layer, filter_index, size,\
            view_prev, target_class, weight_activation = key_values

    EXP = EXPERIMENTS[experiment]
    if layer not in EXP.layer_analyzed:
        print(f"layer {layer} for {experiment} not supported")
        return f"layer {layer} for {experiment} not supported", 404

    time1 = time.time()

    target = EXP.DMI.decode_class_name(target_class)
    image_tensor, _ = EXP.DMI.get_data_sample(image_index, preprocessing=True)

    _, relevances, _, activations = EXP.XAI.calc_basic_attribution(
        image_tensor, method, image_index, target)

    if weight_activation:
        weight_array = activations
    else:
        weight_array = relevances

    EXP.AG.analyze(image_index, image_tensor, weight_array,
                   method, target, layer, [filter_index], [4, 2])
    if view_prev:
        attr_graph = EXP.AG.filter_selected_only(layer, filter_index, 3)
    else:
        attr_graph = EXP.AG.attr_graph

    if len(attr_graph) == 0 or len(attr_graph["nodes"]) == 1:

        socketio.emit("receive_attribution_graph", ("empty"), to=request.sid)
        return

    filter_layer = {}
    for key in attr_graph["nodes"]:
        id = key["id"]
        layer_name, ch_index = attr_graph["properties"][id]["layer"], attr_graph["properties"][id]["filter_index"]

        if layer_name not in filter_layer:
            filter_layer[layer_name] = []

        filter_layer[layer_name].append(int(ch_index))

       # filter_name = get_names_from_cookie(session, layer, [ch_index])
        # if ch_index in filter_name:
        #   attr_graph["properties"][str(id)]["name"] = filter_name[ch_index]

    print("AG elapsed", time.time() - time1)

    socketio.emit("receive_attribution_graph", (attr_graph), to=request.sid)

    for layer_name in filter_layer:

        list_filter = filter_layer[layer_name]

        if mode == "max_activation" or mode == "max_relevance" or mode == "max_relevance_target":

            vis_realistic({
                "experiment": experiment,
                "list_filter": list_filter,
                "layer": layer_name,
                "size": json_mode["size"],
                "sample_indices": json_mode["sample_indices"],
                "mode": mode,
                "target_class": target_class
            }, attr_graph=True)
        elif mode == "synthetic":
            vis_synthetic({
                "experiment": experiment,
                "list_filter": list_filter,
                "layer": layer_name,
                "size": json_mode["size"]
            }, attr_graph=True)


def vis_partial_heatmap(json_response):
    """
       send { experiment : string, image_index : integer , method : string, filter_index : int, size : int,
            weight_activation: 0/1}
       size optional
       :return: jsonify({ "image" : b64 string, "filter_index" : int})

    """

    keys = ["image_index", "experiment", "size", "method",
            "layer", "list_filter", "weight_activation", "target_class"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        image_index, experiment, size, method, layer,\
            list_filter, weight_activation, target_class = key_values

    EXP = EXPERIMENTS[experiment]

    target = EXP.DMI.decode_class_name(target_class)
    data_tensor, _ = EXP.DMI.get_data_sample(image_index, preprocessing=True)

    attr_ch = EXP.XAI.calc_attr_channel(
        list_filter, data_tensor, image_index, layer, method, target, weight_activation)

    pos_dict = {}
    binary_dict = {}
    for i, attr_i in enumerate(attr_ch):
        img = EXP.DMI.visualize_heatmap(attr_i, size, image_index)
        binary_dict[list_filter[i]] = encode_image_to_binary(img)
        pos_t = np.unravel_index(
            np.argmax(abs(attr_i)), shape=attr_ch.shape[1:])
        pos_t = [p for p in pos_t]  # elements of tuples can not be changed
        for k, shape in enumerate(attr_ch.shape[1:]):
            pos_t[k] = pos_t[k]/shape
        pos_dict[list_filter[i]] = str(pos_t)

    meta_data = {
        "pos_filter": pos_dict,
        "weight_activation": weight_activation
    }
    socketio.emit("receive_partial_heatmap",
                  (binary_dict, meta_data), to=request.sid)


def get_filter_statistics():
    """
       send { experiment : string, image_index : integer , filter_index : int, size : int,
            sample_indices: "int:int", n_classes: int, stats_mode: str}
       size optional

       :return: jsonify({ "image" : list of images as b64 string, "class_rel" : list of relevances in percentage,
       "class_name": list of string names})

       n_classes: select amount of most relevant classes to visualize
       sample_indices: which samples per class for instance 0:9 shows 8 most relevant samples per class

    """

    json_response = request.get_json()
    keys = ["experiment", "size", "layer",
            "filter_index", "sample_indices", "n_classes", "stats_mode"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, filter_index, sample_indices, n_classes, stats_mode = key_values
        first_sample, last_sample = sample_indices

    EXP = EXPERIMENTS[experiment]

    sorted_names, sorted_rel_class, sorted_d_indices, sorted_n_indices, sorted_r_value = \
        EXP.XAI.get_statistics(layer, filter_index, n_classes, stats_mode)

    send_images = load_and_encode_rel_images(
        EXP, filter_index, sorted_d_indices, sorted_n_indices, layer, size, first_sample, last_sample)

    # TODO: rel of each image
    return jsonify({"class_name": sorted_names.tolist(), "class_rel": sorted_rel_class, "image": send_images})


# TEST ### VVVVVV

def suggest_filter_name():
    """
       send { experiment : string, layer: string, filter_index : int, concept_name: string}

    """
    json_response = request.get_json()
    keys = ["experiment", "layer", "filter_index", "concept_name"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, layer, filter_index, concept_name = key_values

    EXP = EXPERIMENTS[experiment]

    EXP.CN.add_suggestion(layer, filter_index, concept_name)

    add_concept_name_cookie(session, layer, filter_index, concept_name)

    return "Success", 202

# TODO: useful?


def overwrite_concept_name():
    """
       send { experiment : string, layer: string, filter_index : int, concept_name: string}

    """

    json_response = request.get_json()
    keys = ["experiment", "layer", "filter_index", "concept_name"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, layer, filter_index, concept_name = key_values

    EXP = EXPERIMENTS[experiment]

    EXP.CN.change_concept_name(layer, filter_index, concept_name)

    return "Success", 202

# TODO: add route


def change_config_tag():
    """
       send>=3.7 { experiment : string, tag: string, value: string}

        mode: "replace" replace name of filter
    """

    json_response = request.get_json()
    keys = ["experiment", "tag", "value"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, tag, value = key_values

    EXP = EXPERIMENTS[experiment]

    EXP.config_message[tag] = value
    edit_config(experiment, tag, value)

    return "Success", 202


def get_all_examples_filter():

    json_response = request.get_json()
    keys = ["experiment", "layer", "filter_index", "size"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "", 404
    else:
        experiment, layer, filter_index, size = key_values

    EXP = EXPERIMENTS[experiment]

    send_images = load_and_encode_filter_examples(
        EXP, [filter_index], layer, size, 0, 100)

    return jsonify(images=send_images)

# /\


def create_app(experiments: list, devices: list):
    # every experiment can be loaded on a different device
    if len(devices) > 1 and len(devices) != len(experiments):
        raise ValueError(
            f"<devices> argument {devices} must be at least one element or equal to the length of {experiments}.")

    # for zip(...)
    if len(devices) == 1:
        devices = devices * len(experiments)

    for exp, dev in zip(experiments, devices):
        EXPERIMENTS[exp] = load_experiment(exp, dev)

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'silent'

    app.add_url_rule('/', 'home', view_func=home, methods=["GET"])

    socketio = SocketIO(app, cors_allowed_origins="*")
    socketio.on('connect')(connected)
    socketio.on('disconnect')(disconnected)
    socketio.on("get_data")(select_datapoint)
    socketio.on("get_XAI_available")(get_XAI_available)
    socketio.on("get_global_analysis")(global_analysis)
    socketio.on("vis_realistic")(vis_realistic)
    socketio.on("vis_synthetic")(vis_synthetic)
    socketio.on("get_heatmap")(get_heatmap)
    socketio.on("get_local_analysis")(local_analysis)
    socketio.on("vis_partial_heatmap")(vis_partial_heatmap)
    socketio.on("vis_realistic_heatmaps")(vis_realistic_heatmaps)
    # TODO: use memap for this also, really slow
    socketio.on("vis_statistics")(vis_statistics)
    socketio.on("vis_statistics_heatmaps")(vis_statistics_heatmaps)
    socketio.on("get_attribution_graph")(get_attribution_graph)

    return app, socketio


def get_app():
    app, socketio = create_app(["LeNet"], ["cuda:0"])

    return app

#app, socketio = create_app(["VGG16_ImageNet_a_sum_r_sum_rel"], ["cuda:0"]*2)


if __name__ == "__main__":
    #################### INITIALIZE SERVER #################

    # 172.17.13.225 = titan
    # home http://192.168.178.30:5050

    # gunicorn --bind 0.0.0.0:5050 main_server:app --workers=1 --timeout=120
    # VGG16_ImageNet_a_sum_r_sum_rel #VGG16_Adience_small_sum
    #app, socketio = create_app(["LeNet", "VGG16_ImageNet_a_sum_r_sum_rel", "Resnet34_birds"], ["cuda:0"]*3)
    exps = ["LeNet"]
    #exps = ["LeNet_max_all_unnormed_pred", "LeNet_max_unnormed_pred"]
    devices = ["cuda:0"]*len(exps)
    app, socketio = create_app(exps, devices)
    # app, socketio = create_app(["VGG16_ImageNet_a_sum_r_sum_rel", "Resnet34_birds"],
    #                          ["cuda:0"]*2)
    # app, socketio = create_app(["VGG16_ImageNet_a_max_r_max_rel", "VGG16_ImageNet_a_sum_r_sum_rel"],
    #     ["cuda:0"]*2)
    #app, socketio = create_app(["LeNet", "Resnet34_birds"], ["cuda:0"]*2)
    # app, socketio = create_app(["Resnet34_a_max_r_max_rel_normed", "Resnet34_birds_a_max_r_max_rel",
    #                           ], ["cuda:0"]*2)
    # app, socketio = create_app(["VGG16_Adience_ImageNet", "VGG16_bn_Adience_ImageNet",
    # "VGG16_bn_Adience", "VGG16_Adience_no_bn"], ["cuda:0"]*4)
   # app, socketio = create_app(["VGG16_ImageNet_a_sum_r_sum_rel", "VGG16_ImageNet_a_sum_r_sum_act",
    #            "VGG16_ImageNet_a_max_r_max_act",  "VGG16_ImageNet_a_max_r_max_rel"], ["cuda:0"]*4)
    #app, socketio = create_app(["Resnet34_birds", "Resnet34_birds_DBN"], ["cuda:0", "cuda:0"])
    # app = create_app(["LeNet", "VGG16_ImageNet", "VGG16_ImageNet_activations"],
    #               ["cuda:0", "cuda:0", "cuda:0"])
    socketio.run(app, host="0.0.0.0", port=5052)

    # gunicorn --bind 0.0.0.0: 5050 backend: create_app(["LeNet", "ECG", "VGG16_ImageNet", "VGG16_ImageNet_activations"],
    #                ["cuda:0", "cuda:0", "cuda:0", "cuda:0"]) --workers=1 --timeout=120

    # gunicorn --bind 0.0.0.0:5050 --worker-class eventlet -w 1 --threads 10 socket_backend:app
    
    # gunicorn --bind 0.0.0.0:5050 -w 1 --threads 100 socket_backend:app
    # gunicorn -w 1 --threads 100 module:app

    pass
