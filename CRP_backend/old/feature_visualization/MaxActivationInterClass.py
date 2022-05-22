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

class MaxActivationInterClass:

    def __init__(self, MG: ModelGraph, SDS, ZAPI: ZennitAPI, save_path, config_message):

        self.most_rel_value, self.most_data_index, self.most_neuron_index = {}, {}, {}
        self.mean_rel = {}

        self.MG = MG
        self.ZAPI = ZAPI
        self.DMI = SDS.DMI
        self.SDS = SDS
        self.save_path = save_path / Path("MaxActivationInterClass/")
        self.save_path_tmp = self.save_path / Path("tmp/")
        self.N_IMAGES = 16

        # settings
        self.set_and_verify_settings(config_message)

    def analyze_layer(self, pred, layer_name, data_indices, targets):
        """
        Finds input samples that are for each neuron in a layer
        """

        if self.clip_act:
            pred = np.clip(pred, 0, None)

        layer = self.MG.named_modules[layer_name]

        n_filters, n_neurons_ch = get_channel_neuron_count(layer, self.MG.output_shapes[layer_name])

        b_targets = np.unique(targets)

        if self.select_channel == "sum":
            act_sum = sum_relevance(layer, pred)

        for target in b_targets:

            label = self.DMI.decode_target(target)
            self.initialize_result_arrays(label, layer_name, n_filters)

            arg_targets = np.argwhere(targets == target).reshape(-1)

            # initialize final result of this function loop
            tmp_n_images = len(arg_targets)
            sorted_rel_value = np.zeros((tmp_n_images, n_filters))
            sorted_data_index = np.zeros((tmp_n_images, n_filters), dtype=np.int32)
            sorted_neuron_index = np.zeros((tmp_n_images, n_filters), dtype=np.int32)

            for f in range(n_filters):
                pred_f = pred[arg_targets, f].reshape(-1, n_neurons_ch)

                n_index_max_b = np.argmax(pred_f, axis=-1)

                if self.select_channel == "sum":
                    act_max_b = act_sum[arg_targets, f]
                else:
                    # max
                    row_indices = np.arange(len(pred_f))
                    act_max_b = pred_f[row_indices, n_index_max_b]

                b_index_sorted = np.argsort(act_max_b)
                n_index_sorted = n_index_max_b[b_index_sorted]
                act_sorted = act_max_b[b_index_sorted]

                d_indices_sorted = data_indices[arg_targets[b_index_sorted]]

                # put everything in one array with shape [most active images, filters]
                sorted_rel_value[:, f] = act_sorted[-tmp_n_images:]
                sorted_data_index[:, f] = d_indices_sorted[-tmp_n_images:]
                sorted_neuron_index[:, f] = n_index_sorted[-tmp_n_images:]

                mean_new = np.mean(pred_f)
                N_m_new = len(pred_f)
                self.mean_of_means(label, layer_name, f, mean_new, N_m_new)

            ########### filters  finished calculating ########
            self.concatenate_with_results(label, layer_name, sorted_rel_value, sorted_data_index, sorted_neuron_index)
            self.sort_result_array(label, layer_name)

    def initialize_result_arrays(self, target, layer_name, n_filters):

        if layer_name not in list(self.most_rel_value.keys()):
            self.most_rel_value[layer_name] = {}
            self.most_data_index[layer_name] = {}
            self.most_neuron_index[layer_name] = {}
            self.mean_rel[layer_name] = {}

        if target not in list(self.most_rel_value[layer_name].keys()):
            # multiply by -inf for sort algorithm later
            self.most_rel_value[layer_name][target] = np.ones((self.N_IMAGES, n_filters)) * float("-inf")
            self.most_data_index[layer_name][target] = np.zeros((self.N_IMAGES, n_filters), dtype=np.int32)
            self.most_neuron_index[layer_name][target] = np.zeros((self.N_IMAGES, n_filters), dtype=np.int32)

            self.mean_rel[layer_name][target] = {
                "value":  np.zeros(n_filters, dtype=np.float),
                "N": np.zeros(n_filters, dtype=np.int32)}

    def delete_result_arrays(self):

        self.most_rel_value, self.most_data_index, self.most_neuron_index, self.mean_rel = {}, {}, {}, {}

    def concatenate_with_results(self, target, layer_name, sorted_rel_value, sorted_data_index, sorted_neuron_index):

        # concatenate results of all batches
        # in order to find maximal values of all past batches processed so far
        self.most_rel_value[layer_name][target] = np.concatenate([sorted_rel_value, self.most_rel_value[layer_name][target]])
        self.most_data_index[layer_name][target] = np.concatenate([sorted_data_index, self.most_data_index[layer_name][target]])
        self.most_neuron_index[layer_name][target] = np.concatenate([sorted_neuron_index, self.most_neuron_index[layer_name][target]])

    def mean_of_means(self, target, layer_name, filter_index, mean_new, N_m_new):
        """
        method calculates the weighted mean (i.e. mathematical correct mean of means)
        """

        N_m_past = self.mean_rel[layer_name][target]["N"][filter_index]
        N_sum = N_m_new + N_m_past
        
        mean_past = self.mean_rel[layer_name][target]["value"][filter_index]
        self.mean_rel[layer_name][target]["value"][filter_index] =   N_m_new/N_sum * mean_new + N_m_past/N_sum * mean_past

        self.mean_rel[layer_name][target]["N"][filter_index] = N_sum


    def sort_result_array(self, target, layer_name):

        arg_sorted_act_value = np.argsort(self.most_rel_value[layer_name][target], 0)
        arg_sorted_act_value = arg_sorted_act_value[-self.N_IMAGES:]  # take most active values only
        # loop through all indices (for loop only because of performance issues)
        # doing: most_rel_value = tmp_most_value[arg_sorted_tmp_most_value]
        #       most_data_index = tmp_data_index[arg_sorted_tmp_most_value]
        #       ....

        for i, c in enumerate(arg_sorted_act_value.T):  # iterate over columns
            self.most_rel_value[layer_name][target][-self.N_IMAGES:, i] = self.most_rel_value[layer_name][target][c, i]
            self.most_data_index[layer_name][target][-self.N_IMAGES:, i] = self.most_data_index[layer_name][target][c, i]
            self.most_neuron_index[layer_name][target][-self.N_IMAGES:, i] = self.most_neuron_index[layer_name][target][c, i]

        # flip so that first element is maximal
        self.most_rel_value[layer_name][target] = np.flip(self.most_rel_value[layer_name][target][-self.N_IMAGES:], axis=0)
        self.most_data_index[layer_name][target] = np.flip(self.most_data_index[layer_name][target][-self.N_IMAGES:], axis=0)
        self.most_neuron_index[layer_name][target] = np.flip(self.most_neuron_index[layer_name][target][-self.N_IMAGES:], axis=0)

    def save_results(self, data_start, data_end):

        self.trim_result_arrays()

        for layer_name in self.MG.named_modules:
            saveFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_a_value.p", self.most_rel_value[layer_name])
            saveFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_d_index.p", self.most_data_index[layer_name])
            saveFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_n_index.p", self.most_neuron_index[layer_name])
            saveFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_mean.p", self.mean_rel[layer_name])

        self.delete_result_arrays()

    def trim_result_arrays(self):
        """
        remove unused parts of arrays
        """
        
        layer_names = list(self.MG.named_modules.keys())
        labels = list(self.most_rel_value[layer_names[0]].keys())

        for layer_name in layer_names:
            
            for lab in labels:
                end_indices = np.where(self.most_rel_value[layer_name][lab][:, 0] == float("-inf"))[0]

                if len(end_indices) > 0:
                    end_index = end_indices.min()

                    self.most_rel_value[layer_name][lab] = self.most_rel_value[layer_name][lab][:end_index]
                    self.most_data_index[layer_name][lab] = self.most_data_index[layer_name][lab][:end_index]
                    self.most_neuron_index[layer_name][lab] = self.most_neuron_index[layer_name][lab][:end_index]


    def command_to_parameters(self, command_argument):

        _, data_start, data_end, _, _ = command_argument.split(" ")

        data_start, data_end = map(int, (data_start, data_end))

        return data_start, data_end

    def collect_results(self, command_arguments):

        """
                Checks whether all neurons were analyzed. If so, concatenate the result for every layer.
        """

        print("MaxActivationInterClass: Check if all files were calculated according to argfile.txt...")

        files = []

        # get list of all files that should be calculated
        for command in command_arguments:

            data_start, data_end = self.command_to_parameters(command)
            # compute Statistics only for Base Dataset not Extra Dataset
            if self.SDS.regions[0] < data_start:
                continue

            for layer_name in self.MG.named_modules:

                # check if file exists

                file_name = f"{data_start}_{data_end}_{layer_name}_a_value.p"
                file_path = self.save_path_tmp / Path(file_name)

                if not file_path.exists():
                    print(f"At least file: {file_name} missing.")
                    return -1

            files.append([data_start, data_end])

        if len(files) > 0:
            print("All files completed! Start collecting...")
        else:
            print("No files analyzed.")
            return

        for layer_name in self.MG.named_modules:

            n_filters, _ = get_channel_neuron_count(self.MG.named_modules[layer_name],
                                                    self.MG.output_shapes[layer_name])

            self.delete_result_arrays()

            for data_start, data_end in files:
                # collect all samples

                rel_value = loadFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_a_value.p")
                data_index = loadFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_d_index.p")
                neuron_index = loadFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_n_index.p")
                mean_value = loadFile(self.save_path_tmp, f"{data_start}_{data_end}_{layer_name}_mean.p")

                labels = list(rel_value.keys())
                for lab in labels:
                    self.initialize_result_arrays(lab, layer_name, n_filters)
                    self.concatenate_with_results(lab, layer_name, rel_value[lab], data_index[lab], neuron_index[lab])

                    for f in range(n_filters):
                        self.mean_of_means(lab, layer_name, f, mean_value[lab]["value"][f], mean_value[lab]["N"][f])
            
            labels = list(self.most_rel_value[layer_name].keys())
            for lab in labels:
                self.sort_result_array(lab, layer_name)


            saveFile(self.save_path, f"{layer_name}_r_value.p", self.most_rel_value[layer_name])
            saveFile(self.save_path, f"{layer_name}_d_index.p", self.most_data_index[layer_name])
            saveFile(self.save_path, f"{layer_name}_n_index.p", self.most_neuron_index[layer_name])
            saveFile(self.save_path, f"{layer_name}_mean.p", self.mean_rel[layer_name])

            print(f"MaxActivationInterClass: {layer_name} collected.")

        # delete all files afterwards
        shutil.rmtree(self.save_path_tmp)


    def set_and_verify_settings(self, config_message):

        self.select_channel = config_message["act_select_channel"]
        self.clip_act = int(config_message["clip_act"])

        if self.select_channel != "sum" and self.select_channel != "max":
            raise ValueError("keyword <select_channel> in config.xml has wrong value.")
        if self.clip_act != 0 and self.clip_act != 1:
            raise ValueError("keyword <clip_act> in config.xml has wrong value.")

