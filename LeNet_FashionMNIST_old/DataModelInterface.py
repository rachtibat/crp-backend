
import torch
from pathlib import Path

from typing import List, Iterable, Tuple, Any, Union, Iterator
import numpy as np
import torch.utils.data
import concurrent.futures

from CRP_backend.datatypes.Image_2D_Dataset import Image_2D_Dataset

import torchvision
import torchvision.transforms as T
from CRP_backend.experiments.LeNet.FashionLeNet import FashionLeNet
import torch.utils.data
from CRP_backend.feature_visualization.utils import load_max_activation, load_receptive_field
from CRP_backend.datatypes.data_utils import *


class DataModelInterface(Image_2D_Dataset):

    def __init__(self, device, model_path, data_path, extra_paths: list):
        """

        Args:
            device: "cpu" or "cuda:integer" corresponding to torch.device
        """

        self.file_path = Path(__file__).parent

        self.classes = {
            0: "T - shirt / top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

        self.class_to_target_dict = {v: k for k, v in self.classes.items()}

        self.softmax = torch.nn.Softmax(dim=-1)

        super(DataModelInterface, self).__init__(device, model_path, data_path, extra_paths)

    def build_model(self, model_path=None) -> torch.nn.Module:
        """
        method initializes model (loading of weights, .eval(), shifting model to gpu etc.).
        Please make sure, that the last layer is not a softmax activation.

        Returns:
            torch.nn.Module without softmax activation at output
        """

        model = FashionLeNet()
        model.load_state_dict(torch.load(self.file_path / Path("FashionLeNet.p")))
        model.eval()

        return model.to(self.device)

    def load_dataset(self, data_path=None) -> torch.utils.data.Dataset:
        """
        Parameter:
            data_path: (pathlib.Path object) path to data. You do not need to use it, if you hardcode it below.
        Returns:
             Map-style datasets see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
        """

        test_set = torchvision.datasets.FashionMNIST(
            root=self.file_path / Path('data/FashionMNIST'),
            train=False,
            download=True,
            transform=T.Compose([
                T.ToTensor()
            ])
        )

        return test_set

    def preprocess_data(self, data: torch.tensor) -> torch.tensor:

        return data
       
    def get_all_targets(self):

        return np.arange(0, 10)
    
    def load_extra_dataset_objects(self, data_paths: list):

        return []

    def decode_pred(self, pred: torch.tensor, N: int) -> Tuple[str, int]:
        """
        method returns an understandable representation for the user. Might be string values, images, audio etc.
        For now only string values are supported

        Args:
            pred: (torch.tensor) model output

        Returns:
            ?
        """

        pred_s = self.softmax(pred)[0]
        indices = torch.flip(torch.argsort(pred_s)[-N:], dims=[0]).detach().cpu().numpy()
        pred_np = pred_s.detach().cpu().numpy()

        return [self.classes[i] for i in indices], pred_np[indices]

    def decode_class_name(self, class_name: str):
 
        return self.class_to_target_dict[class_name]

    def get_all_classes(self):

        return list(self.class_to_target_dict.keys())

    def decode_target(self, target: int) -> str:
        """
        method returns an understandable representation for the user. Might be string values, images, audio etc.
        For now only string values are supported

        Args:
            target/label: (torch.tensor) used in loss function

        Returns:
            ?
        """

        return self.classes[target]

    def data_index_to_filename(self, index: int) -> Union[str, int]:
        """
        the analysis results are saved for each data sample independently. Every data sample is identified with its index
        inside the dataset. If the ordering of the dataset is changed, the program can NOT find the cached results and
        wrong data is returned!
        To allow the user, to change the data ordering, he can define the method <data_index_to_name> and
        <data_name_to_index>. If the indices change, but the file names stay the same, no corruption of the analysis
         results occurs.
         Default behavior: file name is equal to index

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

    def pred_to_iterable(self, pred) -> Iterable:
        """
        method receives model output prediction and returns an "iterable" object, where each object
        is used to calculate the attribution inside the model.
        Zennit calculates the attribution by backpropagating gradients starting at each object. The gradients of each
        object are added together in each neuron.

        This method is useful, if the model produces several output objects and you either want to analyse all object or
        only a subset of them.

        Args:
            pred:

        Returns:
            iterable object

        """

        if type(pred) == torch.Tensor:
            return [pred]
        else:
            return pred

    def input_to_iterable(self, inputs) -> Iterable[torch.tensor]:
        """
        method receives data input and returns an "iterable" object, where each object
        is used to calculate the attribution inside the model.

        This method is useful, if the model has several inputs and you either want to see the heatmap of all inputs or
            only a subset of them.

        Args:
            pred:

        Returns:
            iterable object

        """

        if type(inputs) == torch.Tensor:
            return [inputs]
        else:
            return inputs

    def visualize_data_sample(self, image_np: np.ndarray, size: int, padding=None) -> np.ndarray:
        """
        method receives first tuple output of self.get_data_sample and returns a numpy array on "cpu" device so that
        the image can be visualized.
        Args:
            data: self.get_data_sample[0]

        Returns:
            numpy array on cpu device
        """

        image_np = np.moveaxis(image_np, 0, -1)  # convert to channel last
        loaded_image = convert_image_type(image_np, 0, 255, np.uint8)
        loaded_image = rescale_image(loaded_image, size)

        return loaded_image


    def adjust_input_relevance(self, inputs: List[torch.tensor]) -> torch.tensor:
        """
        method receives list of input relevances as torch.tensors and applies optionl transformations on the generated
        attribution.
        For example, it is good practice to add up all channels in heatmap images if you are not interested in the
        detailed distribution of the relevances between each image channel.

        Args:
            inputs:

        Returns:

        """

        # remove channel dimension for single image, keep batch dimension
        return inputs[0].sum(1)



    def init_relevance_of_target(self, targets, pred) -> Iterable[torch.tensor]:
        """
        method is used to initialize the relevence in zennit if the attribution should be analyzed
        with respect to the target/label
        Args:
            target: self.get_data_sample[1]
            pred: model prediction

        Returns:
            same shape and datatype as output of self.pred_to_iterable

        """

        r = torch.zeros_like(pred).to(self.device)
        batch_indices = np.arange(0, len(targets))
        r[batch_indices, targets] = 1 #pred[batch_indices, targets] 
        return [r]


