import datetime
from flask import Flask, redirect, render_template, request, session, url_for, send_file, jsonify, Response
from flask_socketio import SocketIO
from kornia import compute_intensity_transformation
from numpy import require

from CRP_backend.server.server_utils import extract_json_keys
from CRP_backend.server.celery_tasks import *

import eventlet
eventlet.monkey_patch()

EXPERIMENTS = {}
SOCKETIO = None


def home():
    # return render_template("test_website.html")
    return render_template("test_socket.html")


def connected(auth):
    print(datetime.datetime.now(), " Client connected.")


def disconnected():
    print(datetime.datetime.now(), " Client disconnected")


def get_available(json_response):
    """
    Answers:
        {"experiment name": {
            "layer_names": list of str,
            "target_map": dict[int/str, str]
        }}
    """

    exp_dict = {}
    for e_name, device in EXPERIMENTS.items():

        task = get_available_exp.apply_async((e_name, device), queue=e_name)
        exp_dict[e_name] = task.get()

    socketio.emit("receive_available", exp_dict, to=request.sid)


def vis_sample(json_response):
    """
    Answers:
        {"image": binary, "index": int, "target": list}
    """

    keys = ["index", "experiment", "size"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        index, experiment, size = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    session["index"] = index  # save as cookie #TODO: delete
    session["experiment"] = experiment

    get_sample.apply_async((experiment, request.sid, index, size))


def vis_heatmap(json_response):
    """
    Receives:
        {"index": int, "experiment:" str, "size": int, "method": str, "top_N": int, "target": int}

    Answers:
        {"image": binary, "index": int, "dec_wrt": str, "dec_truth": str, "pred_names": list, 
        "pred_confidences": list, "rel_layer": dict}
    """

    keys = ["index", "experiment", "size",
            "method", "top_N", "target", "zero_list_filter", "zero_layer"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        index, experiment, size, method, top_N, target, \
            zero_list_filter, zero_layer = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    # session["method"] = method  # save as cookie

    device = EXPERIMENTS[experiment]
    calc_heatmap.apply_async((experiment, device, request.sid, index, method, target, size), queue=experiment)


def get_global_analysis(json_response):
    """
    Receives:
    --------
        descending: boolean
            If True, sort concepts descending

    """

    keys = ["index", "experiment", "method", "layer", "descending", "abs_norm",
            "target", "zero_list_filter", "zero_layer"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        index, experiment, method, layer, descending, abs_norm, \
            target, zero_list_filter, zero_layer = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    device = EXPERIMENTS[experiment]
    attribute_concepts.apply_async((experiment, device, request.sid, index, method,
                                   target, layer, abs_norm, descending), queue=experiment)


def vis_realistic(json_response):
    """
    Answers:
        tuple(binary_list, meta_data)

        meta_data = {
            "concept_id": int,
            "layer": str,
            "mode": str,
        }
    """

    keys = ["experiment", "size", "layer", "concept_id",
            "range", "mode"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        experiment, size, layer, concept_id, s_range, mode = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    get_max_reference.apply_async((experiment, request.sid, concept_id, layer, mode, s_range, size))


def vis_realistic_heatmaps(json_response):
    """
    Answers:
        tuple(binary_list, meta_data)

        meta_data = {
            "concept_id": int,
            "layer": str,
            "mode": str,
        }
    """

    keys = ["experiment", "size", "layer", "concept_id",
            "range", "mode", "method"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        experiment, size, layer, concept_id, s_range, mode, method = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    device = EXPERIMENTS[experiment]
    get_max_reference_heatmap.apply_async(
        (experiment, device, request.sid, concept_id, layer, mode, method, s_range, size),
        queue=experiment)


def vis_conditional_heatmaps(json_response):

    keys = ["experiment", "size", "layer", "list_concept_ids",
            "method", "init_rel", "index", "target"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        experiment, size, layer, concept_ids, method, init_rel, index, \
            target = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    device = EXPERIMENTS[experiment]

    concept_condional_heatmaps.apply_async(
        (experiment, device, request.sid, index, concept_ids, layer, target, method, init_rel, size),
        queue=experiment)


def get_statistics(json_response):

    keys = ["experiment", "layer", "concept_id", "top_N", "mode"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, layer, concept_id, top_N, mode = key_values

    concept_statistics.apply_async((experiment, request.sid, concept_id, layer, mode, top_N))


def vis_stats_realistic(json_response):

    keys = ["experiment", "size", "layer", "concept_id", "range", "target", "mode"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, concept_id, s_range, target, mode = key_values

    concept_statistics_realistic.apply_async((experiment, request.sid, concept_id, layer, target, mode, s_range, size))


def vis_stats_heatmaps(json_response):

    keys = ["experiment", "size", "layer", "concept_id", "range", "target", "mode", "method"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, size, layer, concept_id, s_range, target, mode, method = key_values

    device = EXPERIMENTS[experiment]

    concept_statistics_heatmaps.apply_async(
        (experiment, device, request.sid, concept_id, layer, target, mode, s_range, method, size),
        queue=experiment)


def get_attribution_graph(json_response):

    keys = ["experiment", "layer", "concept_id", "target", "method", "index", "parent_c_id", "parent_layer", "abs_norm"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values == -1:
        return "wrong keyword values or keyword missing", 404
    else:
        experiment, layer, concept_id, target, method, index, parent_c_id, parent_layer, abs_norm = key_values

    device = EXPERIMENTS[experiment]

    compute_attribution_graph.apply_async(
        (experiment, device, request.sid, index, method, concept_id, layer, target, parent_c_id, parent_layer,
         abs_norm),
        queue=experiment)


def get_local_analysis(json_response):

    keys = ["index", "experiment", "method", "layer", "abs_norm",
            "target",  "x", "y", "width", "height", "descending"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        index, experiment, method, layer, abs_norm, \
            target, x, y, width, height, descending = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    device = EXPERIMENTS[experiment]

    compute_local_analysis.apply_async(
        (experiment, device, request.sid, index, target, method, layer, abs_norm, x, y, width, height, descending),
        queue=experiment)


def run_socket_flask(experiments, host="0.0.0.0", port=5052, debug=False):

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'silent'
    app.add_url_rule('/', 'home', view_func=home, methods=["GET"])

    global socketio
    socketio = SocketIO(app, message_queue='amqp://', cors_allowed_origins="*")
    socketio.on('connect')(connected)
    socketio.on('disconnect')(disconnected)

    socketio.on("vis_sample")(vis_sample)
    socketio.on("get_available")(get_available)
    socketio.on("vis_heatmap")(vis_heatmap)

    socketio.on("vis_realistic")(vis_realistic)
    socketio.on("vis_realistic_heatmaps")(vis_realistic_heatmaps)

    socketio.on("vis_conditional_heatmaps")(vis_conditional_heatmaps)

    socketio.on("get_statistics")(get_statistics)
    socketio.on("vis_stats_realistic")(vis_stats_realistic)
    socketio.on("vis_stats_heatmaps")(vis_stats_heatmaps)

    socketio.on("get_local_analysis")(get_local_analysis)
    socketio.on("get_global_analysis")(get_global_analysis)
    socketio.on("get_attribution_graph")(get_attribution_graph)

    EXPERIMENTS.update(experiments)

    socketio.run(app, host=host, debug=debug, port=port)


if __name__ == "__main__":

    run_socket_flask()
