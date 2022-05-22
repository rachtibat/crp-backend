from CRP_backend.core.model_utils import ModelGraph
from CRP_backend.datatypes.DataModelInterface import DataModelInterface
from CRP_backend.zennit_API.API import ZennitAPI
from CRP_backend.core.layer_specifics import get_channel_neuron_count, sum_relevance
from CRP_backend.datatypes.data_utils import saveFile, loadFile

import torch
import numpy as np
import math
import shutil
from pathlib import Path

class MaxActivation:

    def __init__(self, MG: ModelGraph, SDS, ZAPI: ZennitAPI, save_path, config_message):

        self.most_act_value, self.most_data_index, self.most_neuron_index = {}, {}, {}
        self.MG = MG
        self.ZAPI = ZAPI
        self.DMI = SDS.DMI
        self.save_path = save_path / Path("MaxActivation/")
        self.save_path_tmp = self.save_path / Path("tmp/")
        self.N_IMAGES = 40

        self.set_and_verify_settings(config_message)

    def analyze_layer(self, pred, layer_name, data_indices):
        """
        Finds input samples that maximally activate each neuron in a layer
        """
        if self.clip_act:
            pred = np.clip(pred, 0, None)

        layer = self.MG.named_modules[layer_name]

        n_filters, n_neurons_ch = get_channel_neuron_count(layer, self.MG.output_shapes[layer_name])

        self.initialize_result_arrays(layer_name, n_filters)

        if len(pred) < self.N_IMAGES:
            tmp_n_images = len(pred)
        else:
            tmp_n_images = self.N_IMAGES

        # initialize final result of this function call
        sorted_act_value = np.zeros((tmp_n_images, n_filters))
        sorted_data_index = np.zeros((tmp_n_images, n_filters), dtype=np.int32)
        sorted_neuron_index = np.zeros((tmp_n_images, n_filters), dtype=np.int32)

        max_neuron_index_d = np.zeros((tmp_n_images, n_filters), dtype=np.int32) # used in MaxRelevance
        if self.select_channel == "sum":
            act_sum = sum_relevance(layer, pred)
            if self.normalize_act:
                act_sum = act_sum / (abs(act_sum).sum(-1).reshape(-1, 1) + 1e-10)

        for f in range(n_filters):
            pred_f = pred[:, f].reshape(-1, n_neurons_ch)

            n_index_max_b = np.argmax(pred_f, axis=-1)
            max_neuron_index_d[:, f] = n_index_max_b # used in MaxRelevance

            if self.select_channel == "sum":
                act_max_b = act_sum[:, f]
            else:
                # max
                row_indices = np.arange(len(pred_f))
                act_max_b = pred_f[row_indices, n_index_max_b]

            b_index_sorted = np.argsort(act_max_b)
            n_index_sorted = n_index_max_b[b_index_sorted]
            act_sorted = act_max_b[b_index_sorted]

            d_indices_sorted = data_indices[b_index_sorted]

            # put everything in one array with shape [most active images, filters]
            sorted_act_value[:, f] = act_sorted[-tmp_n_images:]
            sorted_data_index[:, f] = d_indices_sorted[-tmp_n_images:]
            sorted_neuron_index[:, f] = n_index_sorted[-tmp_n_images:]

        ########### filters  finished calculating ########

        self.concatenate_with_results(layer_name, sorted_act_value, sorted_data_index, sorted_neuron_index)
        self.sort_result_array(layer_name)

        return max_neuron_index_d

    def initialize_result_arrays(self, layer_name, n_filters):

        if layer_name not in list(self.most_act_value.keys()):
            # multiply by -inf for sort algorithm later
            self.most_act_value[layer_name] = np.ones((self.N_IMAGES, n_filters)) * float("-inf")
            self.most_data_index[layer_name] = np.zeros((self.N_IMAGES, n_filters), dtype=np.int32)
            self.most_neuron_index[layer_name] = np.zeros((self.N_IMAGES, n_filters), dtype=np.int32)

    def delete_result_arrays(self):

        self.most_act_value, self.most_data_index, self.most_neuron_index = {}, {}, {}

    def concatenate_with_results(self, layer_name, sorted_act_value, sorted_data_index, sorted_neuron_index):

        # concatenate results of all batches
        # in order to find maximal values of all past batches processed so far
        self.most_act_value[layer_name] = np.concatenate([sorted_act_value, self.most_act_value[layer_name]])
        self.most_data_index[layer_name] = np.concatenate([sorted_data_index, self.most_data_index[layer_name]])
        self.most_neuron_index[layer_name] = np.concatenate([sorted_neuron_index, self.most_neuron_index[layer_name]])

    def sort_result_array(self, layer_name):

        arg_sorted_act_value = np.argsort(self.most_act_value[layer_name], 0)
        arg_sorted_act_value = arg_sorted_act_value[-self.N_IMAGES:]  # take most active values only
        # loop through all indices (for loop only because of performance issues)
        # doing: most_act_value = tmp_most_value[arg_sorted_tmp_most_value]
        #       most_data_index = tmp_data_index[arg_sorted_tmp_most_value]
        #       ....

        for i, c in enumerate(arg_sorted_act_value.T):  # iterate over columns
            self.most_act_value[layer_name][-self.N_IMAGES:, i] = self.most_act_value[layer_name][c, i]
            self.most_data_index[layer_name][-self.N_IMAGES:, i] = self.most_data_index[layer_name][c, i]
            self.most_neuron_index[layer_name][-self.N_IMAGES:, i] = self.most_neuron_index[layer_name][c, i]

        # flip so that first element is maximal
        self.most_act_value[layer_name] = np.flip(self.most_act_value[layer_name][-self.N_IMAGES:], axis=0)
        self.most_data_index[layer_name] = np.flip(self.most_data_index[layer_name][-self.N_IMAGES:], axis=0)
        self.most_neuron_index[layer_name] = np.flip(self.most_neuron_index[layer_name][-self.N_IMAGES:], axis=0)

    def save_results(self, data_start, data_end):

        self.trim_result_arrays()

        for layer_name in self.MG.named_modules:
            saveFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_a_value.p", self.most_act_value[layer_name])
            saveFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_d_index.p", self.most_data_index[layer_name])
            saveFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_n_index.p", self.most_neuron_index[layer_name])

        self.delete_result_arrays()


    def trim_result_arrays(self):
        """
        remove unused parts of arrays
        """
        
        layer_names = list(self.MG.named_modules.keys())

        for layer_name in layer_names:
        
            end_indices = np.where(self.most_act_value[layer_name][:, 0] == float("-inf"))[0]

            if len(end_indices) > 0:
                end_index = end_indices.min()

                self.most_act_value[layer_name] = self.most_act_value[layer_name][:end_index]
                self.most_data_index[layer_name] = self.most_data_index[layer_name][:end_index]
                self.most_neuron_index[layer_name] = self.most_neuron_index[layer_name][:end_index]

    def command_to_parameters(self, command_argument):

        _, data_start, data_end, _, _ = command_argument.split(" ")

        data_start, data_end = map(int, (data_start, data_end))

        return data_start, data_end

    def collect_results(self, command_arguments):

        """
                Checks whether all neurons were analyzed. If so, concatenate the result for every layer.
        """

        print("MaxActivation: Check if all files were calculated according to argfile.txt...")

        files = []

        # get list of all files that should be calculated
        for command in command_arguments:

            data_start, data_end = self.command_to_parameters(command)

            for layer_name in self.MG.named_modules:

                # check if file exists

                file_name = f"{data_start}_{data_end}_{layer_name}_a_value.p"
                file_path = self.save_path_tmp / Path(file_name)

                if not file_path.exists():
                    print(f"At least file: {file_name} missing.")
                    return -1

            files.append([data_start, data_end])

        print("All files completed! Start collecting...")

        for layer_name in self.MG.named_modules:

            n_filters, _ = get_channel_neuron_count(self.MG.named_modules[layer_name],
                                                    self.MG.output_shapes[layer_name])

            self.delete_result_arrays()
            self.initialize_result_arrays(layer_name, n_filters)

            for data_start, data_end in files:
                # collect all samples

                act_value = loadFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_a_value.p")
                data_index = loadFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_d_index.p")
                neuron_index = loadFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_n_index.p")

                self.concatenate_with_results(layer_name, act_value, data_index, neuron_index)

            self.sort_result_array(layer_name)

            saveFile(self.save_path, f"{layer_name}_a_value.p", self.most_act_value[layer_name])
            saveFile(self.save_path, f"{layer_name}_d_index.p", self.most_data_index[layer_name])
            saveFile(self.save_path, f"{layer_name}_n_index.p", self.most_neuron_index[layer_name])

            print(f"MaxActivation: {layer_name} collected.")

        # delete all files afterwards
        
        shutil.rmtree(self.save_path_tmp)
        

    def set_and_verify_settings(self, config_message):

        self.select_channel = config_message["act_select_channel"]
        self.clip_act = int(config_message["clip_act"])
        self.normalize_act = int(config_message["normalize_act"])

        if self.select_channel != "sum" and self.select_channel != "max":
            raise ValueError("keyword <select_channel> in config.xml has wrong value.")
        if self.clip_act != 0 and self.clip_act != 1:
            raise ValueError("keyword <clip_act> in config.xml has wrong value.")
        if self.normalize_act != 0 and self.normalize_act != 1:
            raise ValueError("keyword <normalize_act> in config.xml has wrong value.")
