import torch
from pathlib import Path
import numpy as np
from typing import Dict, List, Iterable, Tuple, Any, Union, Iterator
import concurrent.futures

class SuperDataset():
    """
    Contains the base dataset and all extra datasets.
    The SuperDataset has len(all datasets) with indices [0 .... len(first dataset) ... len(last dataset)]
    """

    def __init__(self, DMI, analysis_mode="efficient") -> None:
        """
        The analysis_mode changes the targets returned from dataset. In efficient mode is the ground truth label for
        base dataset returned and for the extra datasets "max" (max value of prediction).
        Is the mode not efficient, is every datapoint N_classes times replicated and for each replication the target set
        to class target. This way, Relevance Maximization can be computed for all classes once. This inefficient mode is
        only relevant during Concept Analysis.
        """
        
        self.DMI = DMI
        self.base_dataset = DMI.dataset

        self.base_targets = self.DMI.get_all_targets()
        self.n_class = len(self.base_targets)

        self.extra_datasets = []
        self.extra_DIs = DMI.extra_DIs
        for obj in self.extra_DIs:
            self.extra_datasets.append(obj.dataset)

        self.all_DIs = [self.DMI] + self.extra_DIs
        self.set_analysis_mode(analysis_mode)

    def __set_regions(self):
        """
        returns list that marks at which virtual index each dataset ends according to
        indices = [0 .... len(first dataset) ... len(last dataset)]
        """
        self.regions = []
        if self.analysis_mode == "efficient":
            self.regions.append(len(self.base_dataset))
        else:
            # replicate datsets N_classes times
            self.regions.append(len(self.base_dataset)*self.n_class)

        for i, set in enumerate(self.extra_datasets):
            if self.analysis_mode == "efficient":
                self.regions.append(self.regions[i-1] + len(set))
            else:
                self.regions.append(self.regions[i-1] + len(set)*self.n_class)
        
        self.regions = np.array(self.regions)

    def __len__(self):
        
        return self.regions[-1]

    def set_analysis_mode(self, new_mode="efficient"):
        
        self.analysis_mode = new_mode
        self.__set_regions()

    #TODO: test preprocessing
    def get_data_sample(self, index, preprocessing=False):

        arg = np.argwhere(index < self.regions)[0,0]

        if arg == 0:
            if self.analysis_mode == "efficient":
                return self.DMI.get_data_sample(index, preprocessing)
            else:
                real_index = int(index / self.n_class)
                class_index = index % self.n_class
                virtual_target = self.base_targets[class_index]
                return self.DMI.get_data_sample(real_index, preprocessing)[0], virtual_target
        else:
            index = index - self.regions[arg-1] 
            if self.analysis_mode == "efficient":
                return self.extra_DIs[arg-1].get_data_sample_no_target(index, preprocessing), "max"
            else:
                real_index = int(index / self.n_class)
                class_index = index % self.n_class
                virtual_target = self.base_targets[class_index]
                return self.extra_DIs[arg-1].get_data_sample_no_target(real_index, preprocessing), virtual_target

    #TODO: not needed
    def __preprocess_data_batch(self, data_batch, indices: list):
        """

        Parameter:
            indices: index in SuperDataset corresponding to datapoint in databatch
        """
        
        # split indices to dataset
        arg_indices = {}
        for d_i, index in enumerate(indices):
            arg = np.argwhere(index < self.regions)[0]
            if arg not in arg_indices:
                arg_indices[arg] = []
            arg_indices[arg].append(d_i)
        
        # compute for each dataset the corresponding data
        result = []
        for arg in arg_indices:
            data = data_batch[arg_indices[arg]]
            data = self.all_DIs[arg].preprocess_data_batch(data)
            result.append(data)
        
        return torch.tensor(result)


    def data_index_to_filename(self, index):
      
        arg = np.argwhere(index < self.regions)[0,0]
        if arg == 0:
            if self.analysis_mode == "efficient":
                real_index = index
            else:
                real_index = int(index / self.n_class)
            return self.DMI.data_index_to_filename(real_index), self.DMI.get_name_dataset()

        else:
            index = index - self.regions[arg-1] 
            DI = self.extra_DIs[arg-1]
            if self.analysis_mode == "efficient":
                real_index = index
            else:
                real_index = int(index / self.n_class)
            return DI.data_index_to_filename(real_index), DI.get_name_dataset()
            

    def data_filename_to_index(self, filename, dataset_name):
        
        if self.analysis_mode != "efficient":
            raise ValueError("Function only usable in analysis_mode efficient")

        for i, DI in enumerate(self.all_DIs):
            if dataset_name == DI.get_name_dataset():
                if i == 0:
                    return DI.data_filename_to_index(filename)
                else:
                    return DI.data_filename_to_index(filename) + self.regions[i-1]

        raise ValueError(f"Name of Dataset {dataset_name} not found.")


    
    def get_data_concurrently(self, indices: Union[List, np.ndarray, torch.tensor], preprocessing=False):

        if len(indices) == 1:
            data, label = self.get_data_sample(indices[0], preprocessing)
            return data, label

        threads = []
        data_returned = []
        labels_returned = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for index in indices:
                future = executor.submit(self.get_data_sample, index, preprocessing)
                threads.append(future)

        for t in threads:
            single_data = t.result()[0]
            single_label = t.result()[1]
            data_returned.append(single_data)
            labels_returned.append(single_label)

        data_returned = torch.cat(data_returned, dim=0)
        return data_returned, labels_returned


