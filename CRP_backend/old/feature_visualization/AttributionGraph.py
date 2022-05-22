import json
from pathlib import Path
import os
import numpy as np
from typing import Union


from CRP_backend.core.model_utils import ModelGraph
from CRP_backend.datatypes.DataModelInterface import DataModelInterface
from CRP_backend.zennit_API.API import ZennitAPI
from CRP_backend.core.layer_specifics import get_neuron_selection_mask, sum_relevance
from CRP_backend.datatypes.data_utils import saveFile, loadFile

from CRP_backend.core.caching import GraphCache
from CRP_backend.core.Explainer import round_relevance


class AttributionGraph:

    def __init__(self, exp_name, MG: ModelGraph, DMI: DataModelInterface, ZAPI: ZennitAPI):

        self.MG = MG
        self.ZAPI = ZAPI
        self.DMI = DMI

        self.attr_graph = {}
        self.node_def = {}
        self.node_hiera = {}

        self.GC = GraphCache(DMI, exp_name)

    def add_result_to_dicts(self, parent_layer, parent_ch, child_layer, children_ch: Union[list, np.ndarray],
                            children_rel: Union[list, np.ndarray]):

        nodes = self.attr_graph["nodes"]
        links = self.attr_graph["links"]
        properties = self.attr_graph["properties"]

        children_ids = []  # for self.node_hiera

        source_id = str(self.add_to_node_def(self.node_def, parent_layer, parent_ch))
        self.append_to_list(nodes, {"id": source_id})
        self.append_to_dict(properties, source_id, {"layer": parent_layer, "filter_index": str(parent_ch)})
        for child, rel in zip(children_ch, children_rel):
            target_id = str(self.add_to_node_def(self.node_def, child_layer, child))
            self.append_to_list(nodes, {"id": target_id})
            connection = {"source": source_id, "target": target_id, "label": str(rel)}
            self.append_to_list(links, connection)
            self.append_to_dict(properties, target_id, {"layer": child_layer, "filter_index": str(child)})
            children_ids.append(target_id)

        self.node_hiera[source_id] = children_ids

    def append_to_list(self, myList: list, value):
        if value not in myList:
            myList.append(value)

    def append_to_dict(self, myDict: dict, key, value):
        if str(key) not in myDict:
            myDict[str(key)] = value

    def add_to_node_def(self, node_def, layer, channel):

        node_name = layer + ":" + str(channel)
        if node_name not in node_def:
            reserved_ids = len(node_def)
            node_def[node_name] = reserved_ids + 1

        # return node id
        return node_def[node_name]

    def analyze(self, data_index, input_data, model_relevance, method, target, parent_layer,
                parent_ch: Union[list, np.ndarray], depth: list):

        self.attr_graph, self.node_def, self.node_hiera = self.GC.load_json_files(data_index, method, target, parent_layer)

        # check if next children of parent_channel already calculated
        parent_key_name = f"{parent_layer}:{parent_ch[0]}"
        if parent_key_name in self.node_def:
            parent_id = str(self.node_def[parent_key_name])
            links = self.attr_graph["links"]
            top_n = depth[0]
            already_calc = 0
            for connection in links:
                if str(connection["source"]) == parent_id:
                    already_calc += 1
            if already_calc == top_n:
                # parent node already calculated
                return

        # parent node not calculated before
        # start recursive method
        self.walk_graph(input_data, model_relevance, method, parent_layer, parent_ch, depth)
        self.GC.save_json_files(data_index, method, target, parent_layer, self.attr_graph, self.node_def, self.node_hiera)

    def walk_graph(self, input_data, model_relevance, method, parent_layer,
                   parent_ch: Union[list, np.ndarray], depth: list):
        """
        recursive method that walks through the model graph and calculates the most relevant channel of each
        node passed.

        Parameter:
            input_data : torch.tensor input image
            model_relevance: dictionary containing the intermediate relevance of each layer
            method: explanation method
            parent_layer: layer name of parent channel
            parent_ch: parent channel, where recursive tree walk starts
            depth: list, each element signifies how many children per layer are returned
        """

        if len(depth) == 0:
            # condition to break recursion
            return

        # number of child channels to calculate
        n_top = depth.pop(0)

        for p_ch in parent_ch:

            children_dict = self.calc_next_relevance(input_data, model_relevance, method, parent_layer,
                                                     p_ch, top=n_top)
            # repeat for each child node in each layer
            for next_l in children_dict:
                next_parent_layer = next_l
                next_parent_ch = children_dict[next_l]
                self.walk_graph(input_data, model_relevance, method, next_parent_layer, next_parent_ch, depth=depth)

    def calc_next_relevance(self, input_data, model_relevance, method, parent_layer,
                            parent_ch, top=5):
        """
        method returns <top> child nodes per child layer connected to parent_layer.
        Results are saved in self.attr_graph and self.node_def
        """

        # select channel to analyze
        neuron_selection = get_neuron_selection_mask(self.MG.named_modules[parent_layer],
                                                     self.MG.output_shapes[parent_layer], [parent_ch])
        neuron_selection = neuron_selection * model_relevance[parent_layer]

        if neuron_selection.sum() == 0:
            # no relevance in child nodes
            print(f"no relevance in child nodes of {parent_ch}")
            return {}

        # receive intermediate relevance of all layers before
        #prep_data = self.DMI.preprocess_data_batch(input_data)

        for result in self.ZAPI.same_input_layer_attribution(input_data, parent_layer, neuron_selection.unsqueeze(0),
                                                                                            method, intermediate=True):
            inter_rel = result[1]

        # find layers connected to analyzed layer
        next_layers_list = self.MG.find_next_layers_of(parent_layer)

        # extract per following layer most relevant channels
        result = {}
        for next_l in next_layers_list:
            filter_rel = sum_relevance(self.MG.named_modules[next_l], inter_rel[next_l])[0]
            filter_rel = filter_rel / (sum(abs(filter_rel)) + 1e-10)  # percentage

            children_ch = np.flip(np.argsort(abs(filter_rel))[-top:])
            children_rel = filter_rel[children_ch]
            if abs(children_rel).sum() == 0:
                print(f"no relevance in child nodes of {parent_ch}")
                return {}  # no relevance in child nodes
            else:
                children_rel = round_relevance(children_rel)

            self.add_result_to_dicts(parent_layer, parent_ch, next_l, children_ch, children_rel)
            result[next_l] = children_ch

        return result

    def filter_selected_only(self, parent_layer: str, parent_ch: int, depth: int):
        """
        sends only the portion of the attribution graph that begins at the chosen channel
        """
        desired_ids = []

        try:
            parent_id = str(self.node_def[parent_layer + ":" + str(parent_ch)])
        except KeyError:
            # parent_ch has no relevance
            return {}

        self.walk_hierarchy(desired_ids, [parent_id], depth)

        # copy node ids and properties
        nodes = []
        properties = {}
        for n_id in desired_ids:
            nodes.append({"id": n_id})
            properties[n_id] = self.attr_graph["properties"][n_id]

        # extract node connections
        links = []
        for l in self.attr_graph["links"]:
            if l["source"] in desired_ids and l["target"] in desired_ids:
                links.append(l)

        small_attr_graph = {
            "nodes": nodes,
            "properties": properties,
            "links": links
        }

        return small_attr_graph

    def walk_hierarchy(self, desired_ids: list, parent_ids: list, depth: int):
        """
        recursive method to find the node ids of all children nodes beginning at parent node.
        This method traverses the self.node_hiera dictionary.

        Args:
            desired_ids: result list going to be filled
            parent_ids: starting point for recursion
            depth: integer! not list
        """

        if depth == 0:
            # condition to break recursion
            return

        for p_id in parent_ids:
            self.append_to_list(desired_ids, p_id)
            try:
                child_ids = self.node_hiera[p_id]
            except KeyError:
                continue
            self.walk_hierarchy(desired_ids, child_ids, depth - 1)
