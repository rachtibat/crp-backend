import torch.nn as nn
import numpy as np
import torch
from typing import Union, List

### <LAYER_TYPES> defined below ###
#class Inspection2D(nn.Identity):

 #   def __init__(self):
  #      super().__init__()


class SubclassMeta(type):
    ### code adopted from "Zennit" ###
    '''Meta class to bundle multiple subclasses.'''
    def __instancecheck__(cls, inst):
        """Implement isinstance(inst, cls) as subclasscheck."""
        return cls.__subclasscheck__(type(inst))

    def __subclasscheck__(cls, sub):
        """Implement issubclass(sub, cls) with by considering additional __subclass__ members."""
        candidates = cls.__dict__.get("__subclass__", tuple())
        return type.__subclasscheck__(cls, sub) or issubclass(sub, candidates)

class D3_Layer(metaclass=SubclassMeta):
    '''Abstract base class that describes modules that have a 1-dimensional output like conv2D without batch dimension'''

    __subclass__ = (
        torch.nn.modules.conv.Conv2d,
       # Inspection2D,
    )

    @staticmethod
    def sum_relevance(relevance):
        if len(relevance.shape) != 4:
            raise ValueError("Report to GitHub please")
        return np.sum(relevance, axis=(2, 3))

    @staticmethod
    def get_channel_neuron_count(output_shape):
        n_ch = output_shape[0]
        n_neurons_ch = output_shape[1] * output_shape[2]
        return n_ch, n_neurons_ch

    @staticmethod
    def get_neuron_selection_mask(output_shape, indices):

        neurons_to_analyze = []

        for ch in indices:
            neurons = torch.zeros(1, *output_shape)
            neurons[0, ch] = 1

            neurons_to_analyze = torch.cat((neurons_to_analyze, neurons), dim=0) if len(
                neurons_to_analyze) > 0 else neurons

        return neurons_to_analyze

    @staticmethod
    def get_rf_neuron_selection_mask(output_shape, neuron_indices):

        neurons_to_analyze = []

        # the receptive field for all channels is identical, thus channel 0 sufficient
        for index in neuron_indices:
            neurons = torch.zeros(1, *output_shape)
            neurons[0, 0].view(-1)[index] = 1

            neurons_to_analyze = torch.cat((neurons_to_analyze, neurons), dim=0) if len(
                neurons_to_analyze) > 0 else neurons

        return neurons_to_analyze

    @staticmethod
    def get_max_abs(relevance):
        return np.max(abs(relevance), axis=(2,3))


class D2_Layer(metaclass=SubclassMeta):
    '''Abstract base class that describes modules that have a 2-dimensional output like conv1D without batch dimension'''
    __subclass__ = (
        torch.nn.modules.conv.Conv1d,
    )

    @staticmethod
    def sum_relevance(relevance):
        if len(relevance.shape) != 3:
            raise ValueError("Report to GitHub please")
        return np.sum(relevance, axis=2)

    @staticmethod
    def get_channel_neuron_count(output_shape):
        n_ch = output_shape[0]
        n_neurons_ch = output_shape[1]
        return n_ch, n_neurons_ch

    @staticmethod
    def get_neuron_selection_mask(output_shape, indices):

        neurons_to_analyze = []

        for ch in indices:
            neurons = torch.zeros(1, *output_shape)
            neurons[0, ch] = 1

            neurons_to_analyze = torch.cat((neurons_to_analyze, neurons), dim=0) if len(
                neurons_to_analyze) > 0 else neurons

        return neurons_to_analyze

    @staticmethod
    def get_rf_neuron_selection_mask(output_shape, neuron_indices):

        neurons_to_analyze = []

        # the receptive field for all channels is identical, thus channel 0 sufficient
        for index in neuron_indices:
            neurons = torch.zeros(1, *output_shape)
            neurons[0, 0].view(-1)[index] = 1

            neurons_to_analyze = torch.cat((neurons_to_analyze, neurons), dim=0) if len(
                neurons_to_analyze) > 0 else neurons

        return neurons_to_analyze

    @staticmethod
    def get_max_abs(relevance):
        return np.max(abs(relevance), axis=2)


class D1_Layer(metaclass=SubclassMeta):
    '''Abstract base class that describes modules that have a 1-dimensional output like linear without batch dimension'''
    __subclass__ = (
        torch.nn.modules.linear.Linear,
    )

    @staticmethod
    def sum_relevance(relevance):
        if len(relevance.shape) != 2:
            raise ValueError("Report to GitHub please")
        return relevance

    @staticmethod
    def get_channel_neuron_count(output_shape):
        n_ch = output_shape[-1]
        n_neurons_ch = 1
        return n_ch, n_neurons_ch

    @staticmethod
    def get_neuron_selection_mask(output_shape, indices):
        neurons_to_analyze = []

        for ch in indices:
            neurons = torch.zeros(1, *output_shape)
            neurons[0, ch] = 1

            neurons_to_analyze = torch.cat((neurons_to_analyze, neurons), dim=0) if len(
                neurons_to_analyze) > 0 else neurons

        return neurons_to_analyze

    @staticmethod
    def get_rf_neuron_selection_mask(output_shape, neuron_indices):

        neurons_to_analyze = []

        # the receptive field for all neurons is identical, thus neuron 0 sufficient
        for index in neuron_indices:
            neurons = torch.zeros(1, *output_shape)
            neurons[0, 0] = 1

            neurons_to_analyze = torch.cat((neurons_to_analyze, neurons), dim=0) if len(
                neurons_to_analyze) > 0 else neurons

        return neurons_to_analyze

    @staticmethod
    def get_max_abs(relevance):
        return abs(relevance)


LAYER_TYPES = [D1_Layer, D2_Layer, D3_Layer]


#TODO: description is wrong
def sum_relevance(layer, relevance):
    """
    function sums up relevance of a layer so that a one dimensional array is returned, where
    each row corresponds to a channel/neuron.

    Parameters:
        layer : torch.nn layer
        relevance : numpy array output of dictionary of <calc_attribution>
    Returns:
        relevance : one dimensional numpy array

    """
    for l_type in LAYER_TYPES:
        if issubclass(layer.__class__, l_type):
            return l_type.sum_relevance(relevance)

    raise KeyError(f"{layer} not an analyzable type. Please choose only of {LAYER_TYPES}.")


def get_channel_neuron_count(layer, output_shape):
    """
        function returns the number of channels in a layer and neurons per channel.

        Parameters:
            layer : torch.nn layer
            output_shape : tuple shape of output tensor of layer
        Returns:
            n_ch : number of channels
            n_neurons_ch: number of neurons in a channel

    """
    for l_type in LAYER_TYPES:
        if issubclass(layer.__class__, l_type):
            return l_type.get_channel_neuron_count(output_shape)

    raise KeyError(f"{layer} not an analyzable type. Please choose only of {LAYER_TYPES}.")


def get_neuron_selection_mask(layer, output_shape, indices: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
    """
       function returns a neuron selection mask with the output shape of the layer at the channel/neuron at index set to
       1.

       Parameters:
           indices: channel/neuron index depending on layer
           layer : torch.nn layer
           output_shape : tuple shape of output tensor of layer
       Returns:
           neuron_selection_mask: if tensor than, shape is prediction output shape with one extra dimension at 0
                if list, then each element's shape must be like prediction output shape


    """
    if type(indices) == torch.Tensor:
        indices = indices.long()

    for l_type in LAYER_TYPES:
        if issubclass(layer.__class__, l_type):
            return l_type.get_neuron_selection_mask(output_shape, indices)

    raise KeyError(f"{layer} not an analyzable type. Please choose only of {LAYER_TYPES}.")


def get_rf_neuron_selection_mask(layer, output_shape, neuron_indices: Union[List, torch.Tensor]) -> Union[
    List, torch.Tensor]:
    """
       function returns a neuron selection for ReceptiveField calculations.
       It sets all neurons to 1 at neuron_indices depending on layer type.

       Parameters:
           indices: channel/neuron index depending on layer
           layer : torch.nn layer
           output_shape : tuple shape of output tensor of layer
       Returns:
           neuron_selection_mask: if tensor than, shape is prediction output shape with one extra dimension at 0
                if list, then each element's shape must be like prediction output shape


    """
    if type(neuron_indices) == torch.Tensor:
        neuron_indices = neuron_indices.long()

    for l_type in LAYER_TYPES:
        if issubclass(layer.__class__, l_type):
            return l_type.get_rf_neuron_selection_mask(output_shape, neuron_indices)

    raise KeyError(f"{layer} not an analyzable type. Please choose only of {LAYER_TYPES}.")


def get_max_abs(layer, relevance):
    """
    function retrieves maximal absolute value of each channel in layer

    Parameters:
        layer : torch.nn layer
        relevance : numpy array output of dictionary of <calc_attribution>
    Returns:
        relevance : two dimensional numpy array

    """
    for l_type in LAYER_TYPES:
        if issubclass(layer.__class__, l_type):
            return l_type.get_max_abs(relevance)

    raise KeyError(f"{layer} not an analyzable type. Please choose only of {LAYER_TYPES}.")
