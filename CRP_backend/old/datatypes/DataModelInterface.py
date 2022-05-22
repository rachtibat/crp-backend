import torch
from pathlib import Path

from typing import List, Iterable, Tuple, Any
import torch.utils.data

from CRP_backend.datatypes.Image_2D_Dataset import Image_2D_Dataset

###  inputs --> input_to_iterable --> zennit --> adjust_inputs_relevance --> visualize
###  pred --> pred_to_iterable -------


class DataModelInterface(Image_2D_Dataset):

    def __init__(self, device):
        super(DataModelInterface, self).__init__(device)

        self.file_path = Path(__file__).parent

