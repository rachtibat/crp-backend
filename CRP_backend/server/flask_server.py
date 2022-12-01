import datetime
from flask import Flask, render_template, request, session 
from flask_socketio import SocketIO
from celery import group, chain

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
    Receives:
    {}

    Answers:
        {"experiment name": {
            "layer_names": list of str,
            "target_map": dict[int/str, str]
        }}
    """

    exp_dict = {}
    for e_name, device in EXPERIMENTS.items():

        task = get_available_exp.apply_async((e_name, device), queue=e_name+"_cuda") #TODO: make asynchronous
        exp_dict[e_name] = task.get()

    socketio.emit("receive_available", exp_dict, to=request.sid)


def vis_sample(json_response):
    """
    Answers:
        {"image": binary, "index": int, "target": list}
    """

    keys = ["job_id", "index", "experiment", "size"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, index, experiment, size = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    session["index"] = index  # save as cookie
    session["experiment"] = experiment

    get_sample.apply_async((job, experiment, request.sid, index, size), queue=experiment+"_cpu")


def vis_heatmap(json_response):
    """
    Receives:
        {"index": int, "experiment:" str, "size": int, "method": str, "top_N": int, "target": int}

    Answers:
        {"image": binary, "index": int, "dec_wrt": str, "dec_truth": str, "pred_names": list, 
        "pred_confidences": list, "rel_layer": dict}
    """

    keys = ["job_id", "index", "experiment", "size",
            "method", "top_N", "target", "zero_list_filter", "zero_layer"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, index, experiment, size, method, top_N, target, \
            zero_list_filter, zero_layer = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    # session["method"] = method  # save as cookie

    device = EXPERIMENTS[experiment]
    calc_heatmap.apply_async((job, experiment, device, request.sid, index, method, target, size), queue=experiment+"_cuda")


def get_global_analysis(json_response):
    """
    Receives:
    --------
        descending: boolean
            If True, sort concepts descending

    """

    keys = ["job_id", "index", "experiment", "method", "layer", "descending", "abs_norm",
            "target", "zero_list_filter", "zero_layer"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, index, experiment, method, layer, descending, abs_norm, \
            target, zero_list_filter, zero_layer = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    device = EXPERIMENTS[experiment]
    attribute_concepts.apply_async((job, experiment, device, request.sid, index, method,
                                   target, layer, abs_norm, descending), queue=experiment+"_cuda")


def vis_max_reference(json_response):
    """
    Answers:
        tuple(binary_list, binary_list, meta_data)

        meta_data = {
            ...
        }
    """

    keys = ["job_id", "experiment", "size", "layer", "concept_id",
            "range", "mode", "plot_mode", "rf", "method"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, experiment, size, l_name, concept_id, r_range, mode, plot_mode, rf, method = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    fn_name = "get_max_reference"
    device = EXPERIMENTS[experiment]

    load_fn = load_cache.s(experiment, concept_id, l_name, mode, r_range, method, rf, fn_name, plot_mode).set(queue=experiment + "_cpu")
    compute_fn = get_max_reference.si(experiment, device, concept_id, l_name, mode, r_range, method, rf, plot_mode).set(queue=experiment + "_cuda")
    
    save_fn = save_cache.s(experiment, l_name, mode, r_range, method, rf, fn_name, plot_mode).set(queue=experiment + "_cpu")
    send_fn = send_reference.s(job, experiment, request.sid, concept_id, l_name, mode, fn_name, plot_mode, size, None).set(queue=experiment + "_cpu")

    tasks = chain(
        load_fn,
        compute_fn,
        group(
            send_fn,
            save_fn
        )
    )

    tasks.apply_async()


def vis_stats_reference(json_response):
    """
    Answers:
        same as 'vis_max_reference' with extra 'target' keyword in 'meta_data'
    """

    keys = ["job_id", "experiment", "size", "layer", "concept_id", "range", "target", "mode", "plot_mode", "rf", "method"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, experiment, size, l_name, concept_id, r_range, target, mode, plot_mode, rf, method  = key_values
    else:
        return f"Keyword error. Available are {keys}", 404
        
    fn_name = "get_stats_reference"
    device = EXPERIMENTS[experiment]

    load_fn = load_cache.s(experiment, concept_id, l_name, mode, r_range, method, rf, fn_name, plot_mode).set(queue=experiment + "_cpu")
    compute_fn = get_stats_reference.si(experiment, device, concept_id, l_name, target, mode, r_range, method, rf, plot_mode).set(queue=experiment + "_cuda")
 
    save_fn = save_cache.s(experiment, l_name, mode, r_range, method, rf, fn_name, plot_mode).set(queue=experiment + "_cpu")
    send_fn = send_reference.s(job, experiment, request.sid, concept_id, l_name, mode, fn_name, plot_mode, size, target).set(queue=experiment + "_cpu")

    tasks = chain(
        load_fn,
        compute_fn,
        group(
            send_fn,
            save_fn
        )
    )

    tasks.apply_async()


def vis_conditional_heatmaps(json_response):

    keys = ["job_id", "experiment", "size", "layer", "list_concept_ids",
            "method", "init_rel", "index", "target"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, experiment, size, layer, concept_ids, method, init_rel, index, \
            target = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    device = EXPERIMENTS[experiment]

    concept_condional_heatmaps.apply_async(
        (job, experiment, device, request.sid, index, concept_ids, layer, target, method, init_rel, size),
        queue=experiment+"_cuda")


def get_statistics(json_response):

    keys = ["job_id", "experiment", "layer", "concept_id", "top_N", "mode"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, experiment, layer, concept_id, top_N, mode = key_values
    else:
        return f"Keyword error. Available are {keys}", 404
        

    concept_statistics.apply_async((job, experiment, request.sid, concept_id, layer, mode, top_N), queue=experiment+"_cpu")



def get_attribution_graph(json_response):

    keys = ["job_id", "experiment", "layer", "concept_id", "target", "method", "index", "parent_c_id", "parent_layer", "abs_norm"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, experiment, layer, concept_id, target, method, index, parent_c_id, parent_layer, abs_norm = key_values
    else:
        return f"Keyword error. Available are {keys}", 404        

    device = EXPERIMENTS[experiment]

    compute_attribution_graph.apply_async(
        (job, experiment, device, request.sid, index, method, concept_id, layer, target, parent_c_id, parent_layer,
         abs_norm),
        queue=experiment+"_cuda")


def get_local_analysis(json_response):

    keys = ["job_id", "index", "experiment", "method", "layer", "abs_norm",
            "target",  "x", "y", "width", "height", "descending"]

    key_values = extract_json_keys(json_response, session, keys)
    if key_values:
        job, index, experiment, method, layer, abs_norm, \
            target, x, y, width, height, descending = key_values
    else:
        return f"Keyword error. Available are {keys}", 404

    device = EXPERIMENTS[experiment]

    compute_local_analysis.apply_async(
        (job, experiment, device, request.sid, index, target, method, layer, abs_norm, x, y, width, height, descending),
        queue=experiment+"_cuda")


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

    socketio.on("vis_max_reference")(vis_max_reference)
    socketio.on("vis_stats_reference")(vis_stats_reference)

    socketio.on("get_statistics")(get_statistics)

    socketio.on("vis_conditional_heatmaps")(vis_conditional_heatmaps)

    socketio.on("get_local_analysis")(get_local_analysis)
    socketio.on("get_global_analysis")(get_global_analysis)
    socketio.on("get_attribution_graph")(get_attribution_graph)

    EXPERIMENTS.update(experiments)

    socketio.run(app, host=host, debug=debug, port=port)


if __name__ == "__main__":

    run_socket_flask()
