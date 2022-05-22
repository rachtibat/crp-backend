from numpy.core.numeric import ones_like
from feature_visualization.MaxRelevanceInterClass import MaxRelevanceInterClass
from CRP_backend.core.model_utils import ModelGraph
from CRP_backend.datatypes.DataModelInterface import DataModelInterface
from CRP_backend.datatypes.SuperDataset import SuperDataset
from CRP_backend.zennit_API.API import ZennitAPI
from CRP_backend.core.layer_specifics import get_channel_neuron_count
from CRP_backend.datatypes.data_utils import saveFile, loadFile

from CRP_backend.feature_visualization.MaxActivation import MaxActivation
from CRP_backend.feature_visualization.MaxRelevanceInterClass import MaxRelevanceInterClass
from CRP_backend.feature_visualization.MaxActivationInterClass import MaxActivationInterClass
from CRP_backend.feature_visualization.MaxRelevance import MaxRelevance 

import torch
import numpy as np
import math
import shutil
from pathlib import Path

class ConceptAnalysis:

    def __init__(self, MG: ModelGraph, SDS: SuperDataset, ZAPI: ZennitAPI, save_path, modes: list, config_message):

        self.most_act_value, self.most_data_index, self.most_neuron_index = {}, {}, {}
        self.MG = MG
        self.ZAPI = ZAPI
        self.SDS = SDS

        # necessary and always first element in self.ANALYZER. 
        self.MACT = MaxActivation(MG, SDS, ZAPI, save_path, config_message) 
        self.MREL_inter, self.MACT_inter = None, None

        self.RELMAX = MaxRelevance(MG, SDS.DMI, ZAPI, save_path, config_message) 

        self.set_and_verify_settings(config_message)
        self.max_neuron_index = None
        self.compute_flag = False 
        # optional 
        if "Max Relevance Inter Class" in modes: #TODO: add max act
            self.MREL_inter = MaxRelevanceInterClass(MG, SDS, ZAPI, save_path, config_message)
            self.compute_flag = True 
        if "Max Activation Inter Class" in modes: 
            self.MACT_inter = MaxActivationInterClass(MG, SDS, ZAPI, save_path, config_message)
            self.compute_flag = True 

    def run_analysis(self, data_start, data_end, method, BATCH_SIZE=32):
        
        n_samples = data_end - data_start  # data_end counted exclusively
        samples = np.arange(start=data_start, stop=data_end)

        if self.wrt_all_class:
            # compute w.r.t each target for each sample
            all_targets = self.SDS.DMI.get_all_targets()
            samples = np.repeat(samples, len(all_targets)) #repeat interleaved
            targets_wrt_all = np.tile(all_targets, n_samples)
            n_samples = n_samples * len(all_targets)

        if n_samples > BATCH_SIZE:
            batches = math.ceil(n_samples / BATCH_SIZE)
            batch_size_tmp = BATCH_SIZE
        else:
            batch_size_tmp = n_samples
            batches = 1

        for b in range(batches):
            print(f"Run Zennit on Sample Batch {b + 1}/{batches}")

            samples_batch = samples[b * batch_size_tmp: (b + 1) * batch_size_tmp]

            data_batch, targets_samples = self.SDS.get_data_concurrently(samples_batch)
            targets_samples = np.array(targets_samples) # numpy operation needed
            data_batch = self.SDS.DMI.preprocess_data_batch(data_batch)

            if self.wrt_all_class:
                targets = targets_wrt_all[b * batch_size_tmp: (b + 1) * batch_size_tmp]
            else:
                targets = targets_samples

            _, relevances, _, activations = \
            self.ZAPI.calc_attribution(data_batch, targets, method, intermediate=True, invert_sign=False)  

            print(f"Analyze Sample Batch {b + 1}/{batches}")
            for i, layer_name in enumerate(self.MG.named_modules):
                print(f"Analyze Layer {i + 1}/{len(self.MG.named_modules)}")

                act_l, rel_l = activations[layer_name], relevances[layer_name]

                self.analyze_layer(act_l, rel_l, layer_name, samples_batch, targets, targets_samples, all_targets)

        self.MACT.save_results(data_start, data_end)
        self.RELMAX.save_results(data_start, data_end)

        # compute Statistics only for Base Dataset not Extra Dataset
        if self.SDS.regions[0] > data_start:
            if self.MREL_inter:
                self.MREL_inter.save_results(data_start, data_end)
            if self.MACT_inter:
                self.MACT_inter.save_results(data_start, data_end)

        
    def run_analysis_old(self, data_start, data_end, method, BATCH_SIZE=32):

        n_samples = data_end - data_start  # data_end counted exclusively
        samples = np.arange(start=data_start, stop=data_end)

        if n_samples > BATCH_SIZE:
            batches = math.ceil(n_samples / BATCH_SIZE)
            batch_size_tmp = BATCH_SIZE
        else:
            batch_size_tmp = n_samples
            batches = 1

        for b in range(batches):
            print(f"Run Zennit on Sample Batch {b + 1}/{batches}")

            samples_batch = samples[b * batch_size_tmp: (b + 1) * batch_size_tmp]

            data_batch, targets = self.SDS.get_data_concurrently(samples_batch)
            data_batch = self.SDS.DMI.preprocess_data_batch(data_batch)

            past_n_samples = b * batch_size_tmp + data_start

            _, relevances, _, activations = \
            self.ZAPI.calc_attribution(data_batch, targets, method, intermediate=True, invert_sign=False)

            print(f"Analyze Sample Batch {b + 1}/{batches}")
            for i, layer_name in enumerate(self.MG.named_modules):
                print(f"Analyze Layer {i + 1}/{len(self.MG.named_modules)}")

                self.analyze_layer(activations[layer_name], relevances[layer_name], layer_name, past_n_samples, targets)

        self.MACT.save_results(data_start, data_end)
        self.RELMAX.save_results(data_start, data_end)

        # compute Statistics only for Base Dataset not Extra Dataset
        if self.SDS.regions[0] > data_start:
            if self.MREL_inter:
                self.MREL_inter.save_results(data_start, data_end)
            if self.MACT_inter:
                self.MACT_inter.save_results(data_start, data_end)

    def analyze_layer(self, pred, rel, layer_name, data_indices, targets, sample_targets, all_targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """

        if self.wrt_all_class:
            # activation analysis only for one sample
            unique_indices = np.unique(data_indices, return_index=True)[1]
            if len(unique_indices) > 0:
                d_unqiue = data_indices[unique_indices]
                p_unique = pred[unique_indices]
                t_unique = sample_targets[unique_indices] # used in MACT_inter
                
                self.max_neuron_index = self.MACT.analyze_layer(p_unique, layer_name, d_unqiue)
                if self.select_neuron == "activation":
                    # self.max_neuron_index used in RELMAX. Since max_neuron_index contains only unique
                    # data indices, it must be broadcasted for wrt_all_class computation
                    self.max_neuron_index = np.repeat(self.max_neuron_index, len(all_targets), axis=0)[:len(data_indices)]
        else:
            self.max_neuron_index = self.MACT.analyze_layer(pred, layer_name, data_indices)

        self.RELMAX.analyze_layer(rel, self.max_neuron_index, layer_name, data_indices)

        # compute Statistics only for Base Dataset not Extra Dataset
        if self.compute_flag:

            above = np.where(data_indices >= self.SDS.regions[0])[0]
            if len(above) > 0:
                # cut out portion not in Extra Dataset
                below = np.where(data_indices < self.SDS.regions[0])
                self.compute_flag = False 
                pred = pred[below]
                rel = rel[below]
                targets = targets[below]
                data_indices = data_indices[below]
                
            if self.MREL_inter:
                self.MREL_inter.analyze_layer(self.max_neuron_index, rel, layer_name, data_indices, targets)
            if self.MACT_inter:
                if self.wrt_all_class:
                    self.MACT_inter.analyze_layer(p_unique, layer_name, d_unqiue, t_unique)
                else:
                    self.MACT_inter.analyze_layer(pred, layer_name, data_indices, targets)


    def analyze_layer_old(self, pred, rel, layer_name, past_n_samples, targets, compute_act=True):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """

        if compute_act:
            self.max_neuron_index = self.MACT.analyze_layer(pred, layer_name, past_n_samples)
        self.RELMAX.analyze_layer(rel, self.max_neuron_index, layer_name, past_n_samples)

        # compute Statistics only for Base Dataset not Extra Dataset
        if self.compute_flag:
            if past_n_samples + len(pred) > self.SDS.regions[0]:
                self.compute_flag = False 
                left = self.SDS.regions[0] - past_n_samples 
                if left <= 0:
                    # already over self.SDS.regions[0]
                    return
                pred = pred[:left]
                rel = rel[:left]
                targets = targets[:left]
                
            if self.MREL_inter:
                self.MREL_inter.analyze_layer(self.max_neuron_index, rel, layer_name, past_n_samples, targets)
            if self.MACT_inter and compute_act:
                self.MACT_inter.analyze_layer(pred, layer_name, past_n_samples, targets)
         

    def collect_results(self, command_args):

        self.MACT.collect_results(command_args)
        self.RELMAX.collect_results(command_args)
        
        if self.MREL_inter:
            self.MREL_inter.collect_results(command_args)
        if self.MACT_inter:
            self.MACT_inter.collect_results(command_args)

    def command_to_parameters(self, command_argument):

        _, data_start, data_end = command_argument.split(" ")

        data_start, data_end = map(int, (data_start, data_end))

        return data_start, data_end


    def divide_processes(self, n_processes):
        """
            Parameter:
                n_processes : divide dataset analysis into maximal number of processes

            Writes command line arguments for SamplePatternSearch. Divides data analysis into n_processes.
            """

        command_argument = []

        samples = len(self.SDS)

        ##calculate how many samples each subprocess calculates
        chunk_p = int(math.ceil(samples / n_processes))

        if samples / n_processes <= 1:
            # too many processes
            n_processes = samples
            chunk_p = 1

            print(f"To many processes for sample size {samples}. Use now {n_processes} processes.")

        spawned = 0
        while spawned * chunk_p < samples:

            if (samples - spawned * chunk_p) <= chunk_p:
                if (samples - spawned * chunk_p) == 0:
                    # nothing left
                    break

                # last process to spawn
                start = chunk_p * spawned
                end = samples
                command_argument.append(f"{start} {end}")

                break

            # normal iteration
            start = chunk_p * spawned
            end = chunk_p * (spawned + 1)
            command_argument.append(f"{start} {end}")

            spawned += 1

        return command_argument


    def set_and_verify_settings(self, config_message):

        self.wrt_all_class = int(config_message["wrt_all_class"])
        self.select_neuron = config_message["select_neuron"]

        if self.wrt_all_class != 0 and self.wrt_all_class != 1:
            raise ValueError("keyword <wrt_all_class> in config.xml has wrong value.")
        if self.select_neuron != "activation" and self.select_neuron != "relevance":
            raise ValueError("keyword <select_neuron> in config.xml has wrong value.")