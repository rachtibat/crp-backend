from CRP_backend.core.model_utils import ModelGraph
from CRP_backend.zennit_API.API import ZennitAPI
from CRP_backend.core.caching import AttributionCache, AnalysisCache, WatershedCache
from CRP_backend.core.layer_specifics import sum_relevance, get_neuron_selection_mask
from CRP_backend.core.XAI_utils import reduce_ch_accuracy
from CRP_backend.feature_visualization.utils import load_rel_statistics, load_act_statistics

import numpy as np
import torch
import math
import skimage.measure
from scipy.ndimage.filters import gaussian_filter
from torch._C import dtype


def norm_attribution(attribution, batch_dim=True):

    if batch_dim:
        attribution = attribution[0]  # remove batch dimension

    max_abs = abs(attribution).max() + 1e-10

    return attribution / max_abs


def round_relevance(relevance):
    return ["{:.2f}".format(x * 100) for x in relevance]

def clip_relevance(sorted_rel_relevance, sorted_ch_indices):
    """
    return only relevance != 0
    """

    for i in range(len(sorted_rel_relevance)):
        if sorted_rel_relevance[i] == 0:
            return sorted_rel_relevance[:i], sorted_ch_indices[:i]

    return sorted_rel_relevance, sorted_ch_indices


class Explainer:

    def __init__(self, exp_name, MG: ModelGraph, SDS, ZAPI: ZennitAPI):
        self.exp_name = exp_name

        self.MG = MG
        self.SDS = SDS
        self.DMI = SDS.DMI
        self.ZAPI = ZAPI

        self.ATTC = AttributionCache(self.DMI, exp_name)
        self.ANAC = AnalysisCache(self.DMI, exp_name)
        self.WAC = WatershedCache(self.DMI, exp_name)

        self.zero_hook = None # for method <set_zero_hook>
        self.caching_active = True # wether cache results or not

        # infer heatmap shape
        self.data_sample, target = self.DMI.get_data_sample(0, preprocessing=True)
        #self.data_sample = self.DMI.preprocess_data_batch(data_sample)
        s_target = self.DMI.select_standard_target(target)
        inp_heatmap, _, _, _ = ZAPI.calc_attribution(self.data_sample, [s_target], "all_flat")
        self.heatmap_shape = inp_heatmap.shape

    def calc_basic_attribution(self, data_batch, method, data_index, target, norm_attr=True):

        # if cached attribution results exist, load them
        if self.caching_active:
            cached = self.ATTC.load_attribution(data_index, method, target)
            if cached:
                inp_attrib, relevances, pred, activations = cached
                if norm_attr:
                    inp_attrib = norm_attribution(inp_attrib)  # TODO: remove, save with normalization instead in ATTC
                return inp_attrib, relevances, pred, activations

        #p_data = self.DMI.preprocess_data_batch(data_batch)

        inp_attrib, relevances, pred, activations = \
            self.ZAPI.calc_attribution(data_batch, [target], method, intermediate=True, invert_sign=False)

        if self.caching_active:
            self.ATTC.save_attribution(data_index, method, target, inp_attrib, relevances, pred, activations)

        if norm_attr:
            inp_attrib = norm_attribution(inp_attrib)  # TODO: remove, before ATTC instead

        return inp_attrib, relevances, pred, activations

    def find_relevant_in_sample(self, data_batch, data_index, layer_name, method, target, selected=(0, 10),
                                sorting="max", round_clip=True):

        if sorting != "max" and sorting != "min":
            raise ValueError("For keyword <sorting> are only max and min allowed")
        if selected[0] > selected[1] and selected[1] != -1:
            raise ValueError("first element of <selected> must be smaller or equal second element.")

        # if cached attribution results exist, load them
        if self.caching_active:
            cached = self.ANAC.load_global(data_index, method, target, layer_name, sorting, selected)
            if cached and round_clip:
                ch_indices, rel_relevance = cached
                return ch_indices, rel_relevance

        _, relevances, _, _ = self.calc_basic_attribution(data_batch, method, data_index, target)

        rel_layer = relevances[layer_name]

        layer = self.MG.named_modules[layer_name]
        rel_ch = sum_relevance(layer, rel_layer)[0]

        # relative relevance w.r.t to all absolute relevance
        rel_relevance = rel_ch / (np.sum(abs(rel_ch)) + 1e-10)

        # result = [index of filter with max relevance, ...., index of filter with min relevance]
        if sorting == "max":
            sorted_ch_indices = np.flip(np.argsort(abs(rel_ch)))  # abs -> independent of sign
        else:
            sorted_ch_indices = np.argsort(abs(rel_ch))

        sorted_rel_relevance = rel_relevance[sorted_ch_indices]
        if round_clip:
            sorted_rel_relevance, sorted_ch_indices = clip_relevance(sorted_rel_relevance, sorted_ch_indices)
            sorted_rel_relevance = round_relevance(sorted_rel_relevance)
            if self.caching_active:
                self.ANAC.save_global(data_index, method, target, layer_name, sorting, sorted_ch_indices, sorted_rel_relevance)

        return sorted_ch_indices[selected[0]:selected[1]], sorted_rel_relevance[selected[0]:selected[1]]

    def find_relevant_in_region(self, data: torch.Tensor, data_index, mask, layer_name, method, target: list,
                                selected=(0, 10),
                                sorting="max", accuracy=0.95, round=True, abs_sort=True, BATCH_SIZE=32):

        if sorting != "max" and sorting != "min":
            raise ValueError("For keyword <sorting> are only max and min allowed")
        if selected[0] > selected[1] and selected[1] != -1:
            raise ValueError("first element of <selected> must be smaller or equal second element.")

        layer = self.MG.named_modules[layer_name]

        inp_attr, relevances, _, _ = self.calc_basic_attribution(data, method, data_index, target)

        rel_layer = sum_relevance(layer, relevances[layer_name])[0]
        ch_to_analyze, rel_layer = reduce_ch_accuracy(rel_layer, accuracy)

        attr_ch = np.zeros((len(ch_to_analyze), *inp_attr.shape))  # results are saved here

        if self.caching_active:
            index_index_left = self.ATTC.load_channel_attr(data_index, method, target, layer_name, ch_to_analyze, attr_ch)

        #TODO: bug no preprocess?
        if not self.caching_active or (len(index_index_left) > 0):

            ch_left = ch_to_analyze[index_index_left]

            #self.batched_zennit_call(data, relevances, layer_name, method, ch_left, index_index_left, attr_ch, BATCH_SIZE)
            self.ZAPI.batched_same_input_layer_attribution(data, relevances, layer_name, method, ch_left,
                                                           get_neuron_selection_mask, index_index_left, attr_ch, BATCH_SIZE)

            self.ATTC.save_channel_attr(data_index, method, target, layer_name, ch_left, attr_ch)

        rel_in_mask_ch = self.DMI.extract_rel_in_region(attr_ch, mask)

        # relative relevance w.r.t to all absolute relevance
        rel_relevance = rel_in_mask_ch / (np.sum(abs(rel_in_mask_ch)) + 1e-10)

        # result = [index of filter with max relevance, ...., index of filter with min relevance]
        if sorting == "max":
            #TODO: why not_zero not here?
            if abs_sort:
                sorted_index_indices = np.flip(np.argsort(abs(rel_in_mask_ch)))  # abs -> independent of sign
            else:
                sorted_index_indices = np.flip(np.argsort(rel_in_mask_ch)) 
        else:
            not_zero = np.where(rel_in_mask_ch == 0, float("inf"), rel_in_mask_ch)
            sorted_index_indices = np.argsort(abs(not_zero))

        sorted_ch_indices = ch_to_analyze[sorted_index_indices]
        sorted_rel_relevance = rel_relevance[sorted_index_indices]
        if round:
            sorted_rel_relevance, sorted_ch_indices = clip_relevance(sorted_rel_relevance, sorted_ch_indices)
            sorted_rel_relevance = round_relevance(sorted_rel_relevance)

        return sorted_ch_indices[selected[0]:selected[1]], sorted_rel_relevance[selected[0]:selected[1]]


    def calc_attr_channel(self, filter_indices: list, data_batch: torch.Tensor, data_index, layer_name, method, target, weight_activation=False, normalize=True):
        """
        method computes the attribution heatmap for channels. If <weight_activation> is True,
        the relevance at the channel is initialized with its activation.

        Args:

            weight_activation: If true, relevance at the channel is initialized with its activation.

        """

        attr_ch = np.zeros((len(filter_indices), *self.heatmap_shape[1:]))
        attr_indices = np.arange(0, len(filter_indices))

        if self.caching_active and not weight_activation:
            # load heatmap from cache
            index_index_left = self.ATTC.load_channel_attr(data_index, method, target, layer_name, filter_indices, attr_ch)

            if len(index_index_left) > 0:
                # calculate remainder
                filter_indices = np.array(filter_indices)[index_index_left]
                attr_indices = index_index_left
            else:
                if normalize:
                    return np.array([norm_attribution(attr, False) for attr in attr_ch]) 
                else:
                    return attr_ch
        
        _, relevances, _, activations = self.calc_basic_attribution(data_batch, method, data_index, target)
        
        if weight_activation:
            # used to see where the channel activates
            weight_array = activations
        else:
            # used to see what of the activation is used to classify (why)
            weight_array = relevances
        #TODO: bug no preprocess?
        self.ZAPI.batched_same_input_layer_attribution(data_batch, weight_array, layer_name, method, filter_indices, get_neuron_selection_mask, attr_indices, attr_ch, BATCH_SIZE=10)

        if self.caching_active and not weight_activation:
            self.ATTC.save_channel_attr(data_index, method, target, layer_name, filter_indices, attr_ch)

        if normalize:
            return np.array([norm_attribution(attr, False) for attr in attr_ch]) # saved attribution must be not normed for <find_relevant_in_region>
        else:
            return attr_ch

    #TODO: remove
    def calc_watershed(self, data_batch, data_index, method, target, threshold=0.3, sigma=7):

        cached = self.WAC.load_masks(data_index, method, target)

        if cached != 0:
            return cached

        inp_heatmap, _,_,_ = self.calc_basic_attribution(data_batch, method, data_index, target)
        # independent of sign
        inp_heatmap = abs(inp_heatmap)  
        # remove small relevances
        max_v = inp_heatmap.max() * threshold
        th_heatmap = np.where(inp_heatmap > max_v, inp_heatmap, 0)
        # smooth relevances
        filtered_heatmap = gaussian_filter(th_heatmap.astype(np.float32), sigma=sigma)
        # remove noise and convert to binary uint8
        max_v = filtered_heatmap.max() * threshold
        binary = np.where(filtered_heatmap > max_v, 255, 0).astype(np.uint8)

        labeled_image, count = skimage.measure.label(binary, connectivity=2, return_num=True)

        masks = []
        for lab in range(1, count+1):
            masks.append((labeled_image == lab).astype(np.uint8))

        masks = np.array(masks)
        self.WAC.save_masks(data_index, method, target, masks)

        return masks
    
    def return_cnn_activations(self, data_batch, method, data_index, target, layer_name, filter_indices, norm=True):

        _, _, _, activations = self.calc_basic_attribution(data_batch, method, data_index, target)
        channel_activations = activations[layer_name][0][filter_indices]

        if norm:
            return np.array([norm_attribution(attr, False) for attr in channel_activations])
        else:
            return np.array([attr for attr in channel_activations])


    def calc_attr_examples(self, filter_index: int, data_indices: list, layer_name, method, weight_activation=True):
        """
        computes attribution for all data at <data_indices> for the filter <filter_index>.
        Used for visualizing heatmaps of a filter's example images.
        """

        data_batch, targets = self.SDS.get_data_concurrently(data_indices, preprocessing=True)
        #p_data = self.DMI.preprocess_data_batch(data_batch)

        layer, output_shape = self.MG.named_modules[layer_name], self.MG.output_shapes[layer_name]
        neuron_selection_mask = get_neuron_selection_mask(layer, output_shape, [filter_index])

        _, relevances, _, activations = self.ZAPI.calc_attribution(data_batch, targets, method=method, intermediate=True)
        
        if weight_activation:
            # used to see where the channel activates
            neuron_selection_mask = neuron_selection_mask * activations[layer_name]
        else:
            # used to see what of the activation is used to classify (why)
            neuron_selection_mask = neuron_selection_mask * relevances[layer_name]

        inp_heatmap, _, _ = self.ZAPI.calc_layer_attribution(data_batch, layer_name, neuron_selection_mask, method=method)

        return inp_heatmap


    def calc_N_top_rel(self, class_names, rel_value, filter_index, N=10):

        rel_class = np.zeros(len(class_names))
        for i, name in enumerate(class_names):
            rel_class[i] = rel_value[name][:N, filter_index].mean()

        return rel_class

    def get_statistics(self, layer_name, filter_index, n_classes=5, stats_mode="relevance_stats"):

        #TODO: remove mean_rl
        
        if stats_mode == "relevance_stats":
            rel_value, data_index, neuron_index, mean_rl = load_rel_statistics(self.exp_name, layer_name)
        elif stats_mode == "activation_stats":
            rel_value, data_index, neuron_index, mean_rl = load_act_statistics(self.exp_name, layer_name)
        else:
            raise ValueError("wrong <mode> value")

        class_names = list(rel_value.keys())

        rel_class = self.calc_N_top_rel(class_names, rel_value, filter_index)

        sorted_indices = np.flip(np.argsort(rel_class)[-n_classes:])
        sorted_rel_class = rel_class[sorted_indices]
        sorted_names = np.array(class_names)[sorted_indices]

        sorted_n_indices, sorted_d_indices, sorted_r_value = [], [], []
        for name in sorted_names:
            sorted_n_indices.append(neuron_index[name])
            sorted_d_indices.append(data_index[name])
            sorted_r_value.append(rel_value[name].tolist())

        #if stats_mode == "relevance_stats": TODO
         #   sorted_rel_class = round_relevance(sorted_rel_class)
        #else:
         #   sorted_rel_class = sorted_rel_class.tolist()

        #TODO: sorted_r_value normalize or remove completely?
    
        return sorted_names, sorted_rel_class, sorted_d_indices, sorted_n_indices, sorted_r_value


    def set_zero_hook(self, layer_name, filter_indices):
        """
        method applies torch hooks so that filter outputs of <filter_indices> are set to zero
        """
        if len(filter_indices) == 0 or layer_name == "":
            return

        def zero_hook(module, input, output):
            
            output[:, filter_indices] = output[:, filter_indices] * 0 
            # multiplication so that gradient history remains in tact

            return output

        layer = self.MG.named_modules[layer_name]
        self.zero_hook = layer.register_forward_hook(zero_hook)

        self.caching_active = False

    def remove_zero_hook(self):
        """
        method removes hooks created by <set_zero_filter> method
        """

        if self.zero_hook != None:
            self.zero_hook.remove()
            self.zero_hook = None
            self.caching_active = True

        







    