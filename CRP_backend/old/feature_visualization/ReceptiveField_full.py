import torch
import numpy as np
import math
from pathlib import Path

from CRP_backend.core.model_utils import ModelGraph
from CRP_backend.datatypes.DataModelInterface import DataModelInterface
from CRP_backend.zennit_API.API import ZennitAPI
from CRP_backend.core.layer_specifics import get_channel_neuron_count, get_rf_neuron_selection_mask
from CRP_backend.datatypes.data_utils import saveFile, loadFile
import shutil
import warnings

class ReceptiveField:

    def __init__(self, MG: ModelGraph, DMI: DataModelInterface, ZAPI: ZennitAPI, save_path):

        self.data_sample = DMI.get_data_sample(0)[0]
        self.data_sample = DMI.preprocess_data_batch(self.data_sample)

        self.MG = MG
        self.ZAPI = ZAPI
        self.DMI = DMI
        self.save_path = save_path / Path("ReceptiveField")
        self.save_path_tmp = self.save_path / Path("tmp/")

    def run_analysis(self, layer_start, neuron_start, layer_end, neuron_end, BATCH_SIZE=10):

        for i_l in range(layer_start, layer_end + 1):  # +1 to include "layer_end" index

            print(f"Analyze Layer {i_l + 1}/{layer_end - layer_start}")
            layer_name = list(self.MG.named_modules.keys())[i_l]

            n_neurons, neurons_to_analyze, neuron_start_layer, neuron_end_layer = self.get_neuron_indices_layer(i_l,
                                                                    layer_start, layer_end, neuron_start, neuron_end)

            if n_neurons == 0:
                raise ValueError("n_neurons == 0. Should not happen! Raise issue on github please.")

            # receptive field per neuron for one channel and one layer
            rf_n_l = np.zeros((n_neurons, *self.data_sample.shape[2:]))
            indices_rf = np.arange(0, n_neurons)

            self.ZAPI.batched_same_input_layer_attribution(self.data_sample, None, layer_name, "all_flat",
                                                           neurons_to_analyze, get_rf_neuron_selection_mask,
                                                           indices_rf, rf_n_l, BATCH_SIZE)

            rf_n_l = self.norm_rf_heatmaps(rf_n_l)

            saveFile(self.save_path_tmp, f"{i_l}_{neuron_start_layer}_{neuron_end_layer}.p", rf_n_l)

    def norm_rf_heatmaps(self, heatmaps):

        # normalize between 0 and 255
        result = np.zeros_like(heatmaps, dtype=np.uint8)

        for i, r in enumerate(heatmaps):

            if r.max() != 0:
                r = r / r.max()
                r = r * 255
            else:
                warnings.warn("Receptive field is for one neuron zero.")

            result[i] = r

        return result

    def get_neuron_indices_layer(self, i_l, layer_start, layer_end, neuron_start, neuron_end):
        """
        This method returns all the neurons in a layer "i_l" that have to be analyzed. This is necessary as
        it depends on the layer type and also on the method's parameters.
        """

        # we calculate now the receptive field for a neuron for the first channel only
        # and then extrapolate it for the other channels as they have the same receptive field
        layer_name = list(self.MG.named_modules.keys())[i_l]
        layer = self.MG.named_modules[layer_name]
        out_shape = self.MG.output_shapes[layer_name]

        _, n_neurons = get_channel_neuron_count(layer, out_shape)

        if layer_start == layer_end:
            # only one layer to analyze
            n_neurons = neuron_end - neuron_start
            neuron_end_layer = neuron_end
            neuron_start_layer = neuron_start

        elif i_l == layer_end:
            # last layer
            n_neurons = neuron_end
            neuron_end_layer = neuron_end
            neuron_start_layer = 0

        elif i_l == layer_start:
            # first layer
            neuron_end_layer = n_neurons
            n_neurons = n_neurons - neuron_start
            neuron_start_layer = neuron_start

        else:
            # layer in the middle
            neuron_start_layer = 0
            neuron_end_layer = n_neurons

        neurons_to_analyze = np.arange(neuron_start_layer, neuron_end_layer)

        return n_neurons, neurons_to_analyze, neuron_start_layer, neuron_end_layer

    def divide_processes(self, n_processes):

        """
        Parameter:
            n_processes : divide dataset analysis into maximal number of processes
            ignore_channel : index only neurons for one channel as receptive field is identical for all channels
            max_channel : maximal number of channels to include, if ignore_channel = False
        Writes command line arguments for ReceptiveField. Divides data analysis into n_processes.
        """

        command_argument = []
        n_neurons_l = {}  # number neurons in one layer
        n_neurons = 0  # number of neurons to analyze in whole model

        for layer_name in self.MG.named_modules:  # TODO: remove last layer?

            layer = self.MG.named_modules[layer_name]
            out_shape = self.MG.output_shapes[layer_name]

            _, n_neurons_ch = get_channel_neuron_count(layer, out_shape)

            # start neuron index, end neuron index
            n_neurons_l[layer_name] = n_neurons, n_neurons + n_neurons_ch

            n_neurons += n_neurons_ch

        # calculate how many neurons each subprocess calculates
        chunk_p = int(math.ceil(n_neurons / n_processes))

        if n_neurons / n_processes <= 1:
            # too many processes
            n_processes = n_neurons
            chunk_p = 1

            print(f"To many processes for neuron size {n_neurons}. Use now {n_processes} processes.")

        spawned = 0
        while spawned * chunk_p < n_neurons:

            if (n_neurons - spawned * chunk_p) <= chunk_p:
                if (n_neurons - spawned * chunk_p) == 0:
                    # nothing left
                    break

                # last process to spawn
                start = self.convert_neuron_id_to_layer(chunk_p * spawned, n_neurons_l)
                end_l, end_n = self.convert_neuron_id_to_layer(n_neurons - 1, n_neurons_l)
                # -1 in convert_neuron_id_to_layer because last neuron_index = n_neurons - 1

                # but {end} is always exclusive. So we have to add 1 to the index although it does not exist.
                # Thus, +1 in end_n.
                command_argument.append(f"{start[0]} {start[1]} {end_l} {end_n + 1}")

                break

            # normal iteration
            start = self.convert_neuron_id_to_layer(chunk_p * spawned, n_neurons_l)
            end = self.convert_neuron_id_to_layer(chunk_p * (spawned + 1), n_neurons_l)
            command_argument.append(f"{start[0]} {start[1]} {end[0]} {end[1]}")

            spawned += 1

        return command_argument

    def convert_neuron_id_to_layer(self, neuron_index, n_neurons_l):
        """
        converts the neuron index between [0, maximal number of neurons in model] to (layer, neuron index in layer)

        Parameter:
            neuron_index : neuron index between [0, maximal number of neurons in model]
            n_neurons_l : helper dictionary contains tuple of (neurons summed layer before,
                        neurons summed until this layer)


        """

        for i_l, layer_name in enumerate(self.MG.named_modules):  # TODO: without last layer ?

            n_neurons = n_neurons_l[layer_name][1]

            if neuron_index < n_neurons:
                # return layer_name as integer for simplicity later on
                # layer index, neuron index in layer
                return i_l, neuron_index - n_neurons_l[layer_name][0]

        raise ValueError("Index not found")

    def command_to_parameters(self, command_argument):

        _, layer_start, neuron_start, layer_end, neuron_end = command_argument.split(" ")

        layer_start, neuron_start, layer_end, neuron_end = map(int, (layer_start, neuron_start, layer_end, neuron_end))

        return layer_start, neuron_start, layer_end, neuron_end

    def collect_results(self, command_arguments):
        """
        Checks whether all neurons where analyzed. If so, concatenate the result for every layer.
        """
        print("Check if all files were calculated...")
        files = []

        # get list of all files that should be calculated
        for command in command_arguments:

            layer_start, neuron_start, layer_end, neuron_end = self.command_to_parameters(command)

            for i_l in range(layer_start, layer_end + 1):  # +1 to include layer_end

                _, _, neuron_start_layer, neuron_end_layer = self.get_neuron_indices_layer(i_l, layer_start, layer_end,
                                                                                                            neuron_start,
                                                                                                            neuron_end)

                files.append([i_l, neuron_start_layer, neuron_end_layer])

                # check if file exists

                file_name = Path(f"{i_l}_{neuron_start_layer}_{neuron_end_layer}.p")
                file_path = (self.save_path_tmp / file_name)

                if not file_path.exists():
                    print(f"At least file: {i_l}_{neuron_start_layer}_{neuron_end_layer}.p missing")
                    return -1

        print("All files completed! Start collecting ...")

        k = 0  # index of last concatenated files
        for i in range(len(files)):

            if i == len(files) - 1 or files[i + 1][0] != files[i][0]:
                # last file or next file in queue is for another layer
                # concatenate all files from one layer
                rf_tmp = np.array([])
                files_to_delete = []
                for q in range(k, i + 1):
                    file_name = f"{files[q][0]}_{files[q][1]}_{files[q][2]}.p"
                    files_to_delete.append(file_name)
                    loaded_file = loadFile(self.save_path_tmp, file_name)

                    rf_tmp = np.concatenate((rf_tmp, loaded_file)) if len(rf_tmp) > 0 else loaded_file

                k = i + 1

                layer_name = list(self.MG.named_modules.keys())[files[i][0]]
                saveFile(self.save_path, f"layer_{layer_name}.p", rf_tmp)
                print(f"Layer_{layer_name} collected")

        shutil.rmtree(self.save_path_tmp)
