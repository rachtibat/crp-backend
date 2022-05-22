from pathlib import Path
import os
from typing import Union
import warnings
import numpy as np
import torch
import json

from CRP_backend.datatypes.data_utils import saveFile, loadFile

import CRP_backend

m_path = os.path.dirname(CRP_backend.__file__)


def match_size_parameter(files: list, size: int):
    """
    retrieve images within 20% of <size>

    Args:
        size: integer
        files: list array of filenames

    Returns:
        file name

    """

    all_sizes = np.array([int(file.split(".")[0]) for file in files])
    diff = all_sizes - size
    best_match = all_sizes[np.argmin(diff)]

    if size * 0.8 < best_match < size * 1.2 or size * 0.8 > best_match > size * 1.2:  # best_match might be negative
        return best_match

    return 0


class ImageCache:
    """
    cache for raw input images images, sample heatmaps
    """

    def __init__(self, DMI, experiment: str):

        self.path = Path(m_path + "/cache/" + experiment + "/b64images")
        self.path_label = Path(m_path + "/cache/" + experiment + "/label")
        self.path_heatmap = Path(m_path + "/cache/" + experiment + "/b64images/" + "heatmap/")
        self.path_pred = Path(m_path + "/cache/" + experiment + "/dec_pred")
        self.DMI = DMI

    def gen_path_filter_examples(self, mode, layer_name, ch_index, target_class):
        
        if mode == "max_activation":
            return self.path / f"{mode}/{layer_name}/{ch_index}"
        elif mode == "max_relevance_target":
            return self.path / f"{mode}/{target_class}/{layer_name}/{ch_index}"
        else:
            raise ValueError("wrong mode in caching module")


    def save_filter_examples(self, images, ch_index, layer_name, size, mode, target_class):

            t_path = self.gen_path_filter_examples(mode, layer_name, ch_index, target_class)
            saveFile(t_path, f"{size}.p", images)
    

    def load_filter_examples(self, ch_indices: Union[list, np.ndarray], layer_name: str, size: int, selected: tuple, mode: str, target_class: str):
        """
        Load cached b64 images results. Since the client requires different image <size>, only images within 20% error
        of <size> are loaded.
        Args:
            ch_indices: channel indices
            layer_name: layer name
            size: image size
            min_length: minimum number of images needed. If client requires 9 images, we have to make sure
                that at least 9 images are cached - otherwise it must be recalculated.

        Returns:
            ch_missing: (list) missing channel indices left to analyze
            ch_loaded: (dict) loaded cached b64 images per channel index

        """
        assert selected[0] < selected[1]

        ch_missing = []
        ch_loaded = {}

        for ch_index in ch_indices:
            
            t_path = self.gen_path_filter_examples(mode, layer_name, ch_index, target_class)
            if os.path.exists(t_path):
                # layer or channel cached before but <size> parameter unsure

                files = os.listdir(t_path)
                matching_name = match_size_parameter(files, size)

                if matching_name:
                    cached_img = loadFile(t_path, f"{matching_name}.p")
                    # make sure at least <selected[1]> images were cached
                    if len(cached_img) >= selected[1]:
                        ch_loaded[int(ch_index)] = cached_img[selected[0]:selected[1]]
                    else:
                        ch_missing.append(ch_index)
                else:
                    ch_missing.append(ch_index)

            else:
                # layer or channel never cached before
                ch_missing.append(ch_index)

        return ch_loaded, ch_missing

    def save_sample(self, image, data_index, size: int, label):

        filename = self.DMI.data_index_to_filename(data_index)
        saveFile(self.path / f"{filename}", f"{size}.p", image)
        saveFile(self.path_label / f"{filename}", "label.p", label)

    def load_sample(self, data_index, size: int):
        """
        Load cached b64 images results. Since the client requires different image <size>, only images within 20% error
        of <size> are loaded.
        Args:
            size: image size
        Returns:


        """

        filename = self.DMI.data_index_to_filename(data_index)
        t_path = self.path / f"{filename}"

        if os.path.exists(t_path):
            # imaged cached before but <size> parameter unsure

            files = os.listdir(t_path)
            matching_name = match_size_parameter(files, size)

            if matching_name:
                cached_img = loadFile(t_path, f"{matching_name}.p")
                label = loadFile(self.path_label / f"{filename}", "label.p")

            else:
                return 0

        else:
            return 0

        return cached_img, label

    def save_heatmap(self, image, data_index, size: int, method: str, target_class: str, pred):

        filename = self.DMI.data_index_to_filename(data_index)
        saveFile(self.path_heatmap / f"{filename}" / f"{method}" / f"{str(target_class)}", f"{size}.p", image)
        saveFile(self.path_pred / f"{filename}", "pred.p", pred)

    def load_heatmap(self, data_index, size: int, method: str, target_class):
        """
        Load cached b64 images results. Since the client requires different image <size>, only images within 20% error
        of <size> are loaded.
        Args:
            size: image size
        Returns:


        """

        filename = self.DMI.data_index_to_filename(data_index)
        t_path = self.path_heatmap / f"{filename}" / f"{method}" / f"{str(target_class)}"

        if os.path.exists(t_path):
            # imaged cached before but <size> parameter unsure

            files = os.listdir(t_path)
            matching_name = match_size_parameter(files, size)

            if matching_name:
                cached_img = loadFile(t_path, f"{matching_name}.p")
                pred = loadFile(self.path_pred / f"{filename}", "pred.p")

            else:
                return 0

        else:
            return 0

        return cached_img, pred


class AnalysisCache:

    def __init__(self, DMI, experiment):

        self.DMI = DMI
        self.global_an_path = m_path / Path("cache") / Path(experiment) / Path("global_analysis")

    def generate_path(self, data_index: int, method: str, target, layer_name: str, sorting: str):

        filename = self.DMI.data_index_to_filename(data_index)
        target_name = self.DMI.decode_target(target)

        return self.global_an_path / Path(f"{filename}/{method}/{str(target_name)}/{layer_name}/{sorting}")

    def save_global(self, data_index: int, method: str, target, layer_name: str, sorting: str, ch_indices, ch_r_rel):

        t_path = self.generate_path(data_index, method, target, layer_name, sorting)

        t_cache = {"indices": ch_indices, "rel_relevance": ch_r_rel}

        saveFile(t_path, "ch_indices.p", t_cache)

    def load_global(self, data_index: int, method: str, target, layer_name: str, sorting: str, selected: tuple):
        """
        returns cached analysis results if available.

        Returns:
            if exists:
                cached_res: (dict) contains global analysis results
            else:
                0
        """
        t_path = self.generate_path(data_index, method, target, layer_name, sorting)

        try:

            cached_res = loadFile(t_path, "ch_indices.p")
            ch_indices, rel_relevance = cached_res["indices"], cached_res["rel_relevance"]

            if len(ch_indices) >= selected[1]:

                return ch_indices[selected[0]:selected[1]], rel_relevance[selected[0]:selected[1]]

            else:
                return 0
        except FileNotFoundError as e:

            return 0


class AttributionCache:

    def __init__(self, DMI, experiment: str):

        self.DMI = DMI
        self.path = m_path / Path("cache/" + experiment + "/attributions/")
        self.channel_path = m_path / Path("cache/" + experiment + "/channel_attributions/")

    def save_attribution(self, data_index, method: str, target: torch.Tensor, inp_attrib, relevances, pred,
                         activations):

        filename = self.DMI.data_index_to_filename(data_index)
        target_name = self.DMI.decode_target(target)
        t_path = self.path / f"{filename}/{method}/{str(target_name)}"

        saveFile(t_path, "inp_attrib.p", inp_attrib)
        saveFile(t_path, "relevances.p", relevances)
        saveFile(t_path, "pred.p", pred)
        saveFile(t_path, "activations.p", activations)

    def load_attribution(self, data_index, method: str, target: torch.Tensor):
        """
        returns cached analysis results if available.

        Returns:
            if exists:
                cached_res: (dict) contains global analysis results
            else:
                0
        """

        filename = self.DMI.data_index_to_filename(data_index)
        target_name = self.DMI.decode_target(target)
        t_path = self.path / f"{filename}/{method}/{str(target_name)}"

        try:

            inp_attrib = loadFile(t_path, "inp_attrib.p")
            relevances = loadFile(t_path, "relevances.p")
            pred = loadFile(t_path, "pred.p")
            activations = loadFile(t_path, "activations.p")

            return inp_attrib, relevances, pred, activations

        except:

            return 0

    def generate_path(self, data_index: int, method: str, target, layer_name: str):

        filename = self.DMI.data_index_to_filename(data_index)
        target_name = self.DMI.decode_target(target)

        return self.channel_path / Path(f"{filename}/{method}/{str(target_name)}/{layer_name}")

    def load_channel_attr(self, data_index, method: str, target, layer_name: str, ch_to_analyze, attr_ch):

        t_path = self.generate_path(data_index, method, target, layer_name)

        index_index_left = []  # index of channel index inside ch_to_analyze

        for i, ch in enumerate(ch_to_analyze):

            try:
                t_attr = loadFile(t_path, f"ch_{ch}.p")
                attr_ch[i] = t_attr

            except FileNotFoundError as e:

                index_index_left.append(i)

        return index_index_left

    def save_channel_attr(self, data_index, method: str, target, layer_name: str, ch_indices, attr_ch):

        t_path = self.generate_path(data_index, method, target, layer_name)

        for i, ch in enumerate(ch_indices):
            saveFile(t_path, f"ch_{ch}.p", attr_ch[i])


class WatershedCache:

    def __init__(self, DMI, experiment):

        self.DMI = DMI
        self.main_path = m_path / Path("cache") / Path(experiment) / Path("watershed")

    def generate_path(self, data_index: int, method: str, target):

        filename = self.DMI.data_index_to_filename(data_index)
        target_name = self.DMI.decode_target(target)

        return self.main_path / Path(f"{filename}/{method}/{str(target_name)}")

    def save_masks(self, data_index: int, method: str, target, masks):

        t_path = self.generate_path(data_index, method, target)

        saveFile(t_path, "masks.p", masks)

    def load_masks(self, data_index: int, method: str, target):
        """
        returns cached analysis results if available.

        Returns:
            if exists:
                cached_res: (dict) contains global analysis results
            else:
                0
        """
        t_path = self.generate_path(data_index, method, target)

        try:
            cached = loadFile(t_path, "masks.p")
            return cached

        except FileNotFoundError as e:
            return 0


class GraphCache:

    def __init__(self, DMI, experiment):

        self.DMI = DMI
        self.main_path = m_path / Path("cache") / Path(experiment) / Path("graph")

    def generate_path(self, data_index: int, method: str, target, layer_name: str):

        filename = self.DMI.data_index_to_filename(data_index)
        target_name = self.DMI.decode_target(target)

        return self.main_path / Path(f"{filename}/{method}/{str(target_name)}/{layer_name}")

    def load_json_files(self, data_index, method, target, layer_name):
        """
        load json file if exists.
        If not create empty attr_graph and node_def dictionaries.

        """

        t_path = self.generate_path(data_index, method, target, layer_name)

        if not t_path.exists():
            os.makedirs(t_path)
            with open(t_path / Path("attribution_graph.json"), 'w', encoding='utf-8') as f:
                attr_graph = {
                    "nodes": [],
                    "properties": {},
                    "links": []
                }
                json.dump(attr_graph, f, ensure_ascii=False, indent=4)
            with open(t_path / Path("node_definitions.json"), 'w', encoding='utf-8') as f:
                node_def = {}
                json.dump(node_def, f, ensure_ascii=False, indent=4)
            with open(t_path / Path("node_hierarchy.json"), 'w', encoding='utf-8') as f:
                node_hiera = {}
                json.dump(node_hiera, f, ensure_ascii=False, indent=4)
        else:

            with open(t_path / Path("attribution_graph.json")) as json_file:
                attr_graph = json.load(json_file)
            with open(t_path / Path("node_definitions.json")) as json_file:
                node_def = json.load(json_file)
            with open(t_path / Path("node_hierarchy.json")) as json_file:
                node_hiera = json.load(json_file)

        return attr_graph, node_def, node_hiera

    def save_json_files(self, data_index, method, target, layer_name, attr_graph, node_def, node_hiera):

        t_path = self.generate_path(data_index, method, target, layer_name)

        with open(t_path / Path("attribution_graph.json"), 'w', encoding='utf-8') as f:
            json.dump(attr_graph, f, ensure_ascii=False, indent=4)

        with open(t_path / Path("node_definitions.json"), 'w', encoding='utf-8') as f:
            json.dump(node_def, f, ensure_ascii=False, indent=4)

        with open(t_path / Path("node_hierarchy.json"), 'w', encoding='utf-8') as f:
            json.dump(node_hiera, f, ensure_ascii=False, indent=4)





class AnalysisCache:

    def __init__(self, DMI, experiment):

        self.DMI = DMI
        self.global_an_path = m_path / Path("cache") / Path(experiment) / Path("global_analysis")

    def generate_path(self, data_index: int, method: str, target, layer_name: str, sorting: str):

        filename = self.DMI.data_index_to_filename(data_index)
        target_name = self.DMI.decode_target(target)

        return self.global_an_path / Path(f"{filename}/{method}/{str(target_name)}/{layer_name}/{sorting}")

    def save_global(self, data_index: int, method: str, target, layer_name: str, sorting: str, ch_indices, ch_r_rel):

        t_path = self.generate_path(data_index, method, target, layer_name, sorting)

        t_cache = {"indices": ch_indices, "rel_relevance": ch_r_rel}

        saveFile(t_path, "ch_indices.p", t_cache)

    def load_global(self, data_index: int, method: str, target, layer_name: str, sorting: str, selected: tuple):
        """
        returns cached analysis results if available.

        Returns:
            if exists:
                cached_res: (dict) contains global analysis results
            else:
                0
        """
        t_path = self.generate_path(data_index, method, target, layer_name, sorting)

        try:

            cached_res = loadFile(t_path, "ch_indices.p")
            ch_indices, rel_relevance = cached_res["indices"], cached_res["rel_relevance"]

            if len(ch_indices) >= selected[1]:

                return ch_indices[selected[0]:selected[1]], rel_relevance[selected[0]:selected[1]]

            else:
                return 0
        except FileNotFoundError as e:

            return 0


class SyntheticCache:

    def __init__(self, DMI, experiment: str):

        self.DMI = DMI
        self.path_base64 = m_path / Path("cache/" + experiment + "/synthetics/base64/")
        self.path_raw = m_path / Path("cache/" + experiment + "/synthetics/raw/")
        self.path_transform = m_path / Path("cache/" + experiment + "/synthetics/transform/")

        self.path_base = m_path / Path("cache/" + experiment + "/synthetics/")
    
    def generate_path(self, mode, layer_name, ch_index):

        if mode == "raw":
            t_path = self.path_raw / f"{layer_name}/"
        elif mode == "base64":
            t_path = self.path_base64 / f"{layer_name}/{ch_index}"
        else:
            raise ValueError("mode wrong value")

        return t_path

        
    def save_base64(self, images_ch, layer_name, size):

        for ch in images_ch:

            t_path = self.generate_path("base64", layer_name, ch)
            saveFile(t_path, f"{size}.p", images_ch[ch])


    def load_base64(self, ch_indices: Union[list, np.ndarray], layer_name: str, size: int):
      

        ch_missing = []
        ch_loaded = {}

        for ch_index in ch_indices:

            t_path = self.generate_path("base64", layer_name, ch_index)
            if os.path.exists(t_path):
                # layer or channel cached before but <size> parameter unsure

                files = os.listdir(t_path)
                matching_name = match_size_parameter(files, size)

                if matching_name:

                    cached_img = loadFile(t_path, f"{matching_name}.p")

                    ch_loaded[int(ch_index)] = cached_img

                else:
                    ch_missing.append(ch_index)

            else:
                # layer or channel never cached before
                ch_missing.append(ch_index)

        return ch_loaded, np.array(ch_missing)


    def save_raw(self, images, ch_indices, layer_name):

        assert len(images) == len(ch_indices)

        for i, ch_index in enumerate(ch_indices):

            t_path = self.generate_path("raw", layer_name, ch_index)
            saveFile(t_path, f"{ch_index}.p", images[i])


    def load_raw(self, ch_indices: Union[list, np.ndarray], layer_name: str, img_buffer):
      

        index_index_left = []  # index of channel index 

        for i_c, ch_index in enumerate(ch_indices):

            t_path = self.generate_path("raw", layer_name, ch_index)
            if os.path.isfile(t_path / Path(f"{ch_index}.p")):

                cached_img = loadFile(t_path, f"{ch_index}.p")

                img_buffer[i_c] = cached_img

            else:
                # layer or channel never cached before
                index_index_left.append(i_c)

        return index_index_left

    def warn_user(self):

        text = f"If you changed the transform for lucent, please delete the cache at {self.path_base}."
        warnings.warn(text)


    #TODO: delete
    def check_past_transform(self, transforms, steps):
        
        t_path = self.path_transform

        if os.path.isfile(t_path / Path("array.p")):
            
            try:
                past_transform = loadFile(t_path, "array.p")
                steps = loadFile(t_path, "steps.p")

                if past_transform != transforms:
                    warnings.warn(f"If you changed the transform for lucent, please delete the cache at {self.path_base}.")
        
            except:
                warnings.warn(f"If you changed the transform for lucent, please delete the cache at {self.path_base}.")
                
            
        saveFile(t_path, "array.p", transforms)
        saveFile(t_path, "steps.p", steps)
