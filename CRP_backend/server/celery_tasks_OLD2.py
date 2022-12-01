from tabnanny import verbose
from celery import Celery, Task
from flask_socketio import SocketIO
import torch
from CRP_backend.utils import get_interface
import CRP_backend.server.config as config
from typing import Tuple

import crp.helper

# celery -A celery_server worker --loglevel=INFO
# celery -A celery_server worker -P solo -l info

from celery import bootsteps
from click import Option

celapp = Celery(__name__) 
celapp.config_from_object(config)

celapp.user_options['worker'].add(
    Option(('--username',), is_flag=True, help='Enable custom option.')
)



class CustomArgs(bootsteps.Step):

    def __init__(self, worker, username, **options):
        super().__init__(worker, **options)

        print("hi", worker, username)

        GPUTask.testing_around = username


celapp.steps['worker'].add(CustomArgs)


class GPUTask(Task):

    def __init__(self) -> None:
        super().__init__()

        print("hasattr testing_around:", hasattr(self, "testing_around"))


    def initialize(self, name, device):

        print("hasattr interface:", hasattr(self, "interface"))
        print("hasattr testing_around:", hasattr(self, "testing_around"))

        if hasattr(self, "interface"):
            # already initialized
            return None

        print(f"model loooading {name}")
        self.interface = get_interface(name)
        self.model = self.interface.get_model(device)
        self.dataset = self.interface.get_dataset()
        self.layer_map = self.interface.get_layer_map(self.model)
        self.target_map = self.interface.get_target_map()
        self.composite_map = self.interface.get_composite_map()
        self.canonizers = self.interface.get_canonizers()
        self.attribution = self.interface.get_CondAttribution(self.model)
        self.fv = self.interface.get_FeatureVisualization(self.attribution, self.dataset, self.layer_map, device)

        self.plot_map = self.interface.get_ref_plot_map()
        
        single_sample = self.fv.get_data_sample(0)[0]
        self.attgraph = self.interface.get_AttributionGraph(self.attribution, single_sample, self.layer_map)

        self.socketio = SocketIO(message_queue='amqp://')


class CPUTask(Task):


    def initialize(self, name):

        if hasattr(self, "interface"):
            # already initialized
            return None

        print(f"data loooading {name}")
        self.interface = get_interface(name)
        self.dataset = self.interface.get_dataset()
        self.target_map = self.interface.get_target_map()
        self.composite_map = self.interface.get_composite_map()

        self.fv = self.interface.get_FeatureVisualization(None, self.dataset, None, "cpu")

        self.plot_map = self.interface.get_ref_plot_map()
        self.canonizers = self.interface.get_canonizers()

        self.socketio = SocketIO(message_queue='amqp://')


@celapp.task(bind=True, base=GPUTask)
def get_available_exp(self, name, device):

    self.initialize(name, device)
    layer_names = list(self.layer_map.keys())
    comp_names = list(self.composite_map.keys())
    plot_names = list(self.plot_map.keys())

    return {"layer_names": layer_names, "target_map": self.target_map, "methods": comp_names, "plot_modes": plot_names}
    

@celapp.task(bind=True, base=CPUTask, ignore_result=True)
def get_sample(self, job, name, sid, index, size):

    self.initialize(name)
    data, groud_truth = self.fv.get_data_sample(index, preprocessing=False)

    try:
        targets = self.fv.multitarget_to_single(groud_truth)
        if type(targets) is not list:
            targets = targets.tolist()
    except NotImplementedError:
        targets = [groud_truth]

    image = self.interface.visualize_sample(data, size)
    binary = self.interface.convert_to_binary(image)

    meta_data = {"index": index, "target": targets, "job_id": job,}

    self.socketio.emit("receive_sample", (binary, meta_data), to=sid)


@celapp.task(bind=True, base=GPUTask, ignore_result=True)
def calc_heatmap(self, job, name: str, device: str, sid, index: int, comp_name: str, target: int, size: int):

    self.initialize(name, device)

    data, _ = self.fv.get_data_sample(index, preprocessing=True)
    
    #EXP.XAI.set_zero_hook(zero_layer, zero_list_filter)

    composite = self.composite_map[comp_name](self.canonizers)
    conditions = [{"y": [target]}]
    record_layers = list(self.layer_map.keys())
    heat, activation, relevances, pred = self.attribution(data, conditions, composite, record_layer=record_layers)

    #EXP.XAI.remove_zero_hook()

    dec_pred = self.interface.decode_prediction(pred)
    image = self.interface.visualize_heatmap(heat, size)
    binary = self.interface.convert_to_binary(image)

    # sum of relevance in each layer as extra information
    rel_l = {}
    for l_name in relevances:
        rel_l[l_name] = str(torch.sum(relevances[l_name]).item())

    meta_data = {
        "index": index,
        "pred_names": dec_pred[0],
        "pred_confidences": dec_pred[1], 
        "rel_layer": rel_l, 
        "job_id": job,
        "target": target}

    self.socketio.emit("receive_heatmap", (binary, meta_data), to=sid)

@celapp.task(bind=True, base=GPUTask, ignore_result=True)
def attribute_concepts(self, job, name: str, device: str, sid, index, comp_name, target, layer_name, abs_norm, descending):

    self.initialize(name, device)

    data, _ = self.fv.get_data_sample(index, preprocessing=True)

    #EXP.XAI.set_zero_hook(zero_layer, zero_list_filter)
    ### TODO replace with caching
    composite = self.composite_map[comp_name](self.canonizers)
    conditions = [{"y": [target]}]
    record_layers = list(self.layer_map.keys())
    _, _, relevances, _ = self.attribution(data, conditions, composite, record_layers)
    ###
    #EXP.XAI.remove_zero_hook()

    rel_c = self.layer_map[layer_name].attribute(relevances[layer_name], abs_norm=abs_norm)[0]
    sorted_c_ids = torch.argsort(rel_c, descending=descending)

    meta_data = {
        "concept_ids": sorted_c_ids.tolist(),
        "relevance": {c_id.item(): rel_c[c_id].item() for c_id in sorted_c_ids},
        "job_id": job,
    }

    self.socketio.emit("receive_global_analysis", meta_data, to=sid)


@celapp.task(bind=True, base=CPUTask, ignore_result=False, serializer="pickle")
def load_cache(self, exp_name, key, l_name, mode, r_range, comp_name, rf, fn_name, plot_name):

    self.initialize(exp_name)

    plot_fn, cache = self.plot_map[plot_name]

    if cache is None:
        return False
    
    composite = self.composite_map[comp_name](self.canonizers)
    ref_c, _ = cache.load([key], l_name, mode, r_range, composite, rf, fn_name, plot_fn.__name__)
 
    if ref_c:
        # remove next celery task, which is 'get_max_reference' or 'get_stats_reference', of chain 
        # because reference samples are already found in cache
        # workaround to enable conditional celery chains that removes only the following task
        if self.request.chain is None:
            raise RuntimeError("'load_cache' must be used inside a celery chain")

        del self.request.chain[-1]
        return ref_c
    else:
        # call 'get_max_reference' or 'get_stats_reference' to compute reference samples since cache is empty 
        return False


@celapp.task(bind=True, base=CPUTask, ignore_result=True)
def save_cache(self, ref_c, exp_name, l_name, mode, r_range, comp_name, rf, fn_name, plot_name):

    self.initialize(exp_name)

    composite = self.composite_map[comp_name](self.canonizers)
    plot_fn, cache = self.plot_map[plot_name]

    cache.save(ref_c, l_name, mode, r_range, composite, rf, fn_name, plot_fn.__name__)

    
@celapp.task(bind=True, base=CPUTask, ignore_result=True)
def send_reference(self, ref_c, job, exp_name, sid, c_id, l_name, mode, fn_name, plot_name, size, target=None):

    self.initialize(exp_name)

    value = next(iter(ref_c.values()))
    
    if not isinstance(value, Tuple):
        value = (value,)

    binary_list = ([], [])
    for i, img_list in enumerate(value):
        for img in img_list:
            img = self.interface.visualize_sample(img, size, padding=True)
            binary = self.interface.convert_to_binary(img)
            binary_list[i].append(binary)

    meta_data = {
        "concept_id": c_id,
        "layer": l_name,
        "mode": mode,
        "job_id": job,
        "experiment": exp_name,
        "plot_mode": plot_name
    }

    if fn_name == "get_max_reference":
        route = "receive_max_reference",
    elif fn_name == "get_stats_reference":
        route = "receive_stats_reference"
        if target is None:
            raise ValueError("You must supply a valid 'target'")
        meta_data["target"] = target
    else:
        raise ValueError("You must supply a valid 'fn_name'")
            
    self.socketio.emit(route, (binary_list[0], binary_list[1], meta_data), to=sid)


@celapp.task(bind=True, base=GPUTask, ignore_result=False)
def get_max_reference(self, exp_name, device, c_id: int, l_name, mode, r_range, comp_name, rf, plot_name):

    self.initialize(exp_name, device)

    composite = self.composite_map[comp_name](self.canonizers)
    plot_fn, _ = self.plot_map[plot_name]
  
    ref_tensor = self.fv.get_max_reference([c_id], l_name, mode, r_range, composite, rf, plot_fn=None, batch_size=config.crp_batch_size)
    key = next(iter(ref_tensor.keys()))
    value = plot_fn(*ref_tensor[key], rf)

    return {key: value}

@celapp.task(bind=True, base=GPUTask, ignore_result=False)
def get_stats_reference(self, exp_name, device, c_id: int, l_name, target: int, mode, r_range, comp_name, rf, plot_name):

    self.initialize(exp_name, device)

    composite = self.composite_map[comp_name](self.canonizers)
    plot_fn, _ = self.plot_map[plot_name]

    ref_tensor = self.fv.get_stats_reference(c_id, l_name, [target], mode, r_range, composite, rf=rf, plot_fn=None, batch_size=config.crp_batch_size)
    key = next(iter(ref_tensor.keys()))
    value = plot_fn(*ref_tensor[key], rf)

    return {key: value}
    

@celapp.task(bind=True, base=CPUTask, ignore_result=True)
def concept_statistics(self, job, name, sid, c_id, layer_name, mode, top_N):

    self.initialize(name)

    targets, values = self.fv.compute_stats(c_id, layer_name, mode, norm=True, top_N=top_N)

    meta_data = {
        "concept_id": c_id,
        "layer": layer_name,
        "mode": mode,
        "targets": targets.tolist(),
        "values": values.tolist(),
        "job_id": job,
        "experiment": name
    }
    
    self.socketio.emit("receive_statistics", meta_data, to=sid)


@celapp.task(bind=True, base=GPUTask, ignore_result=True)
def concept_condional_heatmaps(self, job, name, device, sid, index, concept_ids, layer_name, target, comp_name, init_rel, size):

    self.initialize(name, device)

    data, _ = self.fv.get_data_sample(index, preprocessing=True)
    composite = self.composite_map[comp_name](self.canonizers)

    if init_rel == "relevance":
        conditions = [{layer_name: [c_id], self.attribution.MODEL_OUTPUT_NAME: [target]} for c_id in concept_ids] 
        start_layer = None
    elif init_rel == "activation": 
        conditions = [{layer_name: [c_id]} for c_id in concept_ids] 
        start_layer = layer_name
    else:
        raise ValueError("`mode` must be `activation` or `relevance`")

    binary_dict = {}
    for attr in self.attribution.generate(data, conditions, composite, start_layer=start_layer, verbose=False, batch_size=config.crp_batch_size):

        for i, img in enumerate(attr.heatmap):
            img = self.interface.visualize_heatmap(img, size, padding=False)
            binary = self.interface.convert_to_binary(img)
            binary_dict[concept_ids[i]] = binary

    meta_data = {
        "layer": layer_name,
        "init_rel": init_rel,
        "job_id": job,
        "list_concept_ids": concept_ids
    }
        
    self.socketio.emit("receive_conditional_heatmaps", (binary_dict, meta_data), to=sid)


def reduce_ch_accuracy(rel_layer, accuracy=0.90):
    """
    returns the most relevant channels so that the sum of their relevances is bigger or equal to accuracy * summed relevance
    relevance_layer without batch dimension
    """

    if 0 > accuracy or 1 < accuracy:
        raise ValueError("`accuracy` must be between 0 and 1.")

    abs_rel_layer = abs(rel_layer)
    rel_summed = torch.sum(abs_rel_layer)
    max_rel = rel_summed * accuracy

    indices = torch.argsort(abs_rel_layer, descending=True)

    rel = 0
    for i, id in enumerate(indices):
        rel += abs_rel_layer[id]
        if rel >= max_rel:
            return indices[:i + 1]

    return indices


@celapp.task(bind=True, base=GPUTask, ignore_result=True)
def compute_local_analysis(self, job, name, device, sid, index, target, comp_name, layer, abs_norm, x, y, width, height, descending):

    self.initialize(name, device)

    data, _ = self.fv.get_data_sample(index, preprocessing=True)
    composite = self.composite_map[comp_name](self.canonizers)

    #### TODO: replace with caching
    conditions = [{"y": [target]}]
    _, _, relevances, _ = self.attribution(data, conditions, composite, record_layer=[layer])
    rel_c = self.layer_map[layer].attribute(relevances[layer], abs_norm=False)[0]
    reduced_c_indices = reduce_ch_accuracy(rel_c)
    ###

    conditions = [{self.attribution.MODEL_OUTPUT_NAME: target, layer: id} for id in reduced_c_indices]

    rel_c = []
    for attr in self.attribution.generate(data, conditions, composite, batch_size=config.crp_batch_size, verbose=False):
        
        rel_c.append(self.interface.mask_input_attribution(attr.heatmap, x, y, width, height))

    rel_c = torch.cat(rel_c)
    c_indices = torch.argsort(rel_c, descending=descending)

    if abs_norm:
        rel_c = crp.helper.abs_norm(rel_c)

    meta_data = {
        "concept_ids": c_indices.tolist(),
        "relevance": rel_c[c_indices].tolist(),
        "job_id": job,
    }
    self.socketio.emit("receive_local_analysis", meta_data, to=sid)
    
    

@celapp.task(bind=True, base=GPUTask, ignore_result=True)
def compute_attribution_graph(self, job, name, device, sid, index: int, comp_name: str, c_id, layer: str, target: int, parent_c_id, parent_layer, abs_norm):

    self.initialize(name, device)
    
    data, _ = self.fv.get_data_sample(index, preprocessing=True)
    composite = self.composite_map[comp_name](self.canonizers)
    
    nodes, connections = self.attgraph(data, composite, c_id, layer, target, [4, 2], parent_c_id, parent_layer, abs_norm, config.crp_batch_size, False)

    js_nodes, js_links = [], []

    for n in nodes:
        js_nodes.append({"id": f"{n[0]}:{n[1]}", "layer_name": n[0], "concept_id": n[1]})

    for source in connections:
        for target in connections[source]:
            js_links.append({"source": f"{source[0]}:{source[1]}", "target": f"{target[0]}:{target[1]}", "value": target[2]})

    json_data = {
        "nodes" : js_nodes,
        "links": js_links,
        "job_id": job
    }
    self.socketio.emit("receive_attribution_graph", (json_data,), to=sid)