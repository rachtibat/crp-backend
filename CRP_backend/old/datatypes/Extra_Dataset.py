import torch
import torch
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Any, Union, Iterator
import numpy as np
import torch.utils.data


class Extra_Dataset:

    def __init__(self, device, data_path):
        """

        Args:
            device: "cpu" or "cuda:integer" corresponding to torch.device
        """
        self.device = device

        print(f"Loading dataset {self.get_name_dataset()}...", end='')
        self.dataset = self.load_dataset(data_path)
        print(" finished.")

    def get_name_dataset(self) -> str:
        """
        In order to enable caching of intermediate results as well as differentiating this dataset from
        the base and other extra datasets, a unqiue name must be given.
        """

        return "ExtraDataset_1"
    
    def load_dataset(self, data_path=None) -> torch.utils.data.Dataset:
        """
        !Please do not preprocess the data in <self.load_dataset>. Preprocessing is done in
        <self.preprocess_data_batch>!

        Parameter:
            data_path: (pathlib.Path object) path to data. You do not need to use it, if you hardcode it below.
        Returns:
             Map-style datasets see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
        """

        raise NotImplementedError

    def preprocess_data(self, data: torch.tensor) -> torch.tensor:
        """
        method implements preprocessing of data for the model.
        !Please do not preprocess the data in <self.load_dataset>!

        Parameter:
            data: (torch.tensor) input data for model

        Returns:
            torch.tensor
        """

        raise NotImplementedError

    def get_data_sample_no_target(self, index, preprocessing=False) -> Tuple[Any, Any]:
        """
        !Note that the target value is unimportant"
        Interface describes how a single data sample is loaded from the dataset.
        If method not edited by user, returns single data sample using direct indexing of the dataset.

        Parameter:
            index value of data samples (integer)

        Returns:
            tuple, where first element is data sample at <index> (torch.tensor) and second element is the target.

        """
        data, _ = self.dataset[index]
        
        if preprocessing:
            data = self.preprocess_data(data)
        
        return data.unsqueeze(0).to(self.device)


    def data_index_to_filename(self, index: int) -> Union[str, int]:
        """
        the analysis results are saved for each data sample independently. Every data sample is identified with its index
        inside the dataset. If the ordering of the dataset is changed, the program can NOT find the cached results and
        wrong data is returned!
        To allow the user, to change the data ordering, he can define the method <data_index_to_name> and
        <data_name_to_index>. If the indices change, but the file names stay the same, no corruption of the analysis
        results occurs.
        Default behavior: file name is equal to index
        !Note, if extra datasets are used, please make sure that each data name is unique for all datasets!
        If filename not found, please raise an error in your implementation.

        Args:
            index: dataset index

        Returns:
            filename as string or integer

        """

        return index

    def data_filename_to_index(self, filename: Union[str, int]) -> int:
        """
        see <data_index_to_name>

        Args:
            filename as str or int

        Returns:
            index as integer
        """

        return int(filename)
