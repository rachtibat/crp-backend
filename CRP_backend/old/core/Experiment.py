from distutils.errors import PreprocessError
import os
import importlib
import json
from CRP_backend.core.model_utils import ModelGraph
#from CRP_backend.datatypes.DataModelInterface import DataModelInterface
from CRP_backend.datatypes.SuperDataset import SuperDataset
from CRP_backend.zennit_API.API import ZennitAPI
from CRP_backend.core.model_utils import create_model_representation
from CRP_backend.core.caching import ImageCache
from CRP_backend.core.Explainer import Explainer
from CRP_backend.feature_visualization.AttributionGraph import AttributionGraph
from CRP_backend.feature_visualization.GradientAscent import GradientAscent
from CRP_backend.feature_visualization.ConceptNaming import ConceptNaming
import CRP_backend
from CRP_backend.datatypes.SuperDataset import SuperDataset

from pathlib import Path
import torch
import xml.etree.ElementTree as ET

e_path = os.path.dirname(CRP_backend.__file__) + "/experiments"


def load_model_and_data(name, device, config_message):
    all_names = os.listdir(e_path)

    if name not in all_names:
        raise ValueError(f"Experiment {name} not in folder <experiments>.")

    try:
        DMI_module = importlib.import_module(
            f"CRP_backend.experiments.{name}.DataModelInterface")
        DMI = getattr(DMI_module, "DataModelInterface")(device, config_message["model_path"], config_message["data_path"], config_message["extra_data_paths"])
        SDS = SuperDataset(DMI)
    except Exception as e:
        print(e)
        raise ImportError(
            f"Could not load file <DataModelInterface.py> from path {e_path}/{name}/.")

    return SDS
 


def load_model_graph(DMI):
    print("Creating Model Graph...", end='')
    dummy_data, _ = DMI.get_data_sample(0, preprocessing=True)
    MG = create_model_representation(DMI.model, dummy_data)
    print(" finished.")

    return MG


def load_config(name):
    """
    loads data_path and model_path arguments from config.json if available
    Args:
        name:

    Returns:

    """
    keys = {"data_path": None, "extra_data_paths": None, "model_path": None,
            "save_path": None, "threshold": None, "sigma": None, "wrt_all_class": "0",
            "select_neuron": "activation", "select_channel": "sum", "act_select_channel": "max",
            "normalize_rel": "1", "clip_act": "1", "normalize_act": "0", "multiply_rel": "0",
            "div_rel": 0}

    try:

        mytree = ET.parse(f"{e_path}/{name}/config.xml")
        myroot = mytree.getroot()

        for child in myroot:
            for child_child in child:
                value = child_child.text
                if value is not None and len(value) > 0:

                    if child_child.tag == "extra_data_paths":
                        value = value.split(",")
                    keys[child_child.tag] = value

    finally:
        return keys


def edit_config(name, tag, text):

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))

    mytree = ET.parse(f"{e_path}/{name}/config.xml", parser)
    myroot = mytree.getroot()

    for child in myroot:

        element = child.find(tag)
        if element is not None:
            element.text = text
            mytree.write(f"{e_path}/{name}/config.xml")
            return 1

    return 0


class Experiment:

    def __init__(self, exp_name: str, MG: ModelGraph, SDS: SuperDataset, ZAPI: ZennitAPI, config_message):

        self.exp_name = exp_name
        self.config_message = config_message

        self.SDS = SDS
        self.DMI = SDS.DMI
        self.MG = MG
        self.ZAPI = ZAPI
        self.XAI = Explainer(exp_name, MG, SDS, ZAPI)

        self.AG = AttributionGraph(exp_name, MG, self.DMI, ZAPI)
        self.GA = GradientAscent(MG, self.DMI)

        self.CN = ConceptNaming(f"{e_path}/{exp_name}")

        self.exp_path = self.get_experiment_path()
        self.layer_analyzed = self.get_analyzed_layers()
        self.layer_modes = self.get_analysis_modes_layer()

        self.class_to_indices = self.DMI.selection_of_indices_per_class()

        #self.IC = ImageCache(SDS, exp_name)

    def get_experiment_path(self):

        return e_path / Path(self.exp_name)

    def get_analyzed_layers(self):
        """
        :return all layers available for inspectation. Prio analysis of the model necessary, if layers do not exist.
        """

        layer_analyzed = []

        if not (self.exp_path / Path('ReceptiveField')).is_dir() or not (self.exp_path / Path('MaxActivation')).is_dir():
            raise RuntimeError("Experiment must be analyzed before using it!")

        for layer_name in self.MG.named_modules:

            if os.path.isfile(self.exp_path / Path('ReceptiveField') / Path("layer_" + layer_name + ".npy")):

                if os.path.isfile(self.exp_path / Path('MaxActivation') / Path(layer_name + "_a_value.p")):
                    layer_analyzed.append(layer_name)

        if len(layer_analyzed) == 0:
            raise RuntimeError("Experiment must be analyzed before using it!")

        return layer_analyzed

    #TODO: delete
    def get_cnn_layers_OLD(self):

        cnn_layers = {}
        for l in self.MG.named_modules:
            if isinstance(self.MG.named_modules[l], torch.nn.Conv2d):
                cnn_layers[l] = 1
            else:
                cnn_layers[l] = 0

        return cnn_layers
        

    def get_analysis_modes_layer(self):

        layer_modes = {}
        t_path = e_path + f"/{self.exp_name}/"

        for l in self.layer_analyzed:

            layer_modes[l] = {
                "max_activation" : 1,
                "max_relevance" : 0,
                "max_relevance_target": 0,
                "relevance_stats" : 0,
                "activation_stats" : 0,
                "cnn_activation": 0,
                "synthetic": 0
            }

            if isinstance(self.MG.named_modules[l], torch.nn.Conv2d):
                layer_modes[l]["cnn_activation"] = 1
                
            if self.GA.available:
                layer_modes[l]["synthetic"] = 1

            if os.path.isdir(t_path + "MaxRelevanceInterClass"):
                    layer_modes[l]["relevance_stats"], layer_modes[l]["max_relevance_target"] = 1, 1
            
            if os.path.isdir(t_path + "MaxActivationInterClass"):
                    layer_modes[l]["activation_stats"] = 1
            
            if os.path.isdir(t_path + "MaxRelevance"):
                    layer_modes[l]["max_relevance"] = 1

        return layer_modes