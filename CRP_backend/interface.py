import torch
from PIL import Image
import io
from typing import List, Tuple, Dict, Union
import zennit

from zennit.composites import COMPOSITES

from crp.attribution import CondAttribution
from crp.graph import trace_model_graph
from crp.image import imgify


class Interface:

    def visualize_sample(self):

        raise NotImplementedError("Interface must be implemented!")

    def visualize_heatmap(self):

        raise NotImplementedError("Interface must be implemented!")

    def convert_to_binary(self):

        raise NotImplementedError("Interface must be implemented!")
        
    def mask_input_attribution(self):
        raise NotImplementedError("Interface must be implemented!")
    
    def get_model(self, device):
        """
        method initializes model (loading of weights, .eval(), shifting model to device etc.).
        Please make sure, that the last layer is not a softmax activation.

        Returns:
            torch.nn.Module without softmax activation at output
        """

        raise NotImplementedError("Interface must be implemented!") 

    def get_dataset(self) -> torch.utils.data.Dataset:
        """
        Returns:
             Map-style datasets see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
        """

        raise NotImplementedError("Interface must be implemented!")  

    def get_canonizers(self) -> Union[List, None]:
        """
        Returns:
            zennit canonizer that is applied to the model as list.
            If no canonization should be applied, return None.
        """

        raise NotImplementedError("Interface must be implemented!")  

    def get_composite_map(self):
        """
        Returns:
            dictionary where keys are str names and values are zennit.composites
        """

        raise NotImplementedError("Interface must be implemented!") 

    def get_target_map(self) -> Dict[int, str]:
        """
        method returns an dictionary where keys are integer single target and values 
        understandable string representations for the user. 
        """

        raise NotImplementedError("Interface must be implemented!") 

    def decode_prediction(self, prediction, top_N: int) -> Tuple[List[str], List[float]]:
        """
        method returns an understandable string representation for the user.

        Parameters:
            pred: model output
            top_N: integer. Number of top-N class prediction

        Returns:
            class names and confidence values as tuple. Please make sure not to return
            torch.Tensors or numpy.ndarrays as they are not serializable in json.
        """

        raise NotImplementedError("Interface must be implemented!") 

    def get_layer_map(self, model):

        raise NotImplementedError("Interface must be implemented!")        

    def get_CondAttribution(self, model):

        raise NotImplementedError("Interface must be implemented!") 

    def get_FeatureVisualization(self, attribution: CondAttribution, dataset, layer_map, device):

        raise NotImplementedError("Interface must be implemented!") 

    def get_ReceptiveField(self, attribution: CondAttribution, single_sample: torch.Tensor):

        raise NotImplementedError("Interface must be implemented!") 

    def get_AttributionGraph(self, attribution: CondAttribution, single_sample: torch.Tensor, layer_map):
        
        raise NotImplementedError("Interface must be implemented!") 


class Image2D(Interface):

    def convert_to_binary(self, image: Image):
        
        buffer = io.BytesIO()
        image.save(buffer, format="png", compress_level=1)

        return buffer.getvalue()

    def visualize_heatmap(self, tensor, size, padding=False):

        return imgify(tensor, "bwr", symmetric=True, resize=(size, size), padding=padding) 

    def visualize_sample(self, tensor, size, padding=True):
        
        # remove batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        return imgify(tensor, resize=(size, size), padding=padding)

    def mask_input_attribution(self, heatmap, x, y, width, height):

        inp_shape = heatmap.shape
        x, y = int(x * inp_shape[2]), int(y * inp_shape[1])
        width, height = int(width * inp_shape[2]), int(height * inp_shape[1])

        return torch.sum(heatmap[:, y:y + height, x:x + width], dim=(1, 2))



    

