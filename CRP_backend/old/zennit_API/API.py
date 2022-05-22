import torch
from CRP_backend.zennit_API.API_utils import *
from zennit.composites import *
from CRP_backend.core.model_utils import ModelGraph
from typing import Union
import math
import numpy as np

#TODO: add flat rule to zennit

class ZennitAPI:

    def __init__(self, DMI, MG: ModelGraph, device):

        self.DMI = DMI
        self.MG = MG
        self.model = MG.model
        self.device = device

        self.add_composites()
        self.canonizer = DMI.define_canonizer()

        
    def add_composites(self):
        """
        method adds composites to zennit
        """

        @register_composite('all_epsilon')
        class EpsilonPlusFlat(LayerMapComposite):
            '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
            layers and the epsilon rule for all other fully connected layers.
            '''

            def __init__(self, canonizers=None):
                layer_map = LAYER_MAP_BASE + [
                    (Convolution, Epsilon()),
                    (torch.nn.Linear, Epsilon()),
                ]

                super().__init__(layer_map, canonizers=canonizers)


        @register_composite('all_flat')
        class AllFlatComposite(LayerMapComposite):
            '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
            layers and the epsilon rule for all other fully connected layers.
            '''

            def __init__(self, canonizers=None):
                layer_map = [
                    (Linear, Flat()),
                    (AvgPool, Flat()),
                    (torch.nn.modules.pooling.MaxPool2d, Flat()),
                    (Activation, Pass()),
                    (Sum, Norm()),
                ]

                super().__init__(layer_map, canonizers=canonizers)

    def same_input_layer_attribution(self, same_input: torch.tensor, layer_name: str,
                                     neuron_selection: Union[list, iter], method="epsilon_plus_flat",
                                     intermediate=False):
        """
        special case of <calc_layer_attribution> where the same input is used several times for different masks in
        <neuron selection>. This implementation takes advantages of torch's multiple backpropagation utility and is
        this way 2x more efficient than the standard <calc_layer_attribution> function.
        """

        composite, same_input = self.check_method_data(method, same_input)

        try:
            selected_layer = self.MG.named_modules[layer_name]
        except KeyError:
            raise KeyError(f"{layer_name} is not analyzable or does not exist.")

        # apply hook at selected_layer
        sel_layer_out = []

        def get_tensor_hook(module, input, output):
            sel_layer_out.append(output)

        h = selected_layer.register_forward_hook(get_tensor_hook)

        relevances = {}
        activations = {}

        with composite.context(self.model) as modified:

            # hooks to get intermediate attributions and activations
            if intermediate:
                layer_out, forward_hooks = append_all_layers_forward_hooks(self.MG, intermediate)

            _ = modified(same_input)
            layer_pred = sel_layer_out[0]

            for mask in neuron_selection:
                torch.autograd.backward((layer_pred,), (mask.to(self.device),), retain_graph=True)

                #TODO: unify all 3 lines with method that takes in (model, data_batch, pred)
                inp_list = self.DMI.input_to_iterable(same_input)  # specifies which data's heatmap to compute
                inp_rel = [inp.grad.detach().cpu().numpy() for inp in inp_list]
                inp_heatmap = self.DMI.adjust_input_relevance(inp_rel)

                # save intermediate attributions and activations
                if intermediate:
                    relevances = get_attributions_from_hooks(self.MG, layer_out)
                    activations = save_activations_as_numpy(layer_out)

                yield inp_heatmap, relevances, activations

                reset_gradients(self.model, inp_list)

            h.remove()
            if intermediate:
                [hf.remove() for hf in forward_hooks]


    def calc_attribution(self, data_batch: torch.tensor, target: Union[list, np.ndarray], method="epsilon_plus_flat",
                         intermediate=False, invert_sign=False):

        composite, data_batch = self.check_method_data(method, data_batch)

        relevances = {}
        activations = {}

        with composite.context(self.model) as modified:

            # hooks to get intermediate attributions and activations
            if intermediate:
                layer_out, forward_hooks = append_all_layers_forward_hooks(self.MG, intermediate)

            # compute attributions
            pred = modified(data_batch)

            pred_list = self.DMI.pred_to_iterable(pred)
            mask_list = self.DMI.init_relevance_of_target(target, pred)
            for p, m in zip(pred_list, mask_list):
                m = m.to(self.device)
                if invert_sign:
                    m = torch.where(pred < 0, torch.abs(m) * -1, torch.abs(m))

                torch.autograd.backward((p,), (m,))

            #TODO: unify all 3 lines with method that takes in (model, data_batch, pred)
            inp_list = self.DMI.input_to_iterable(data_batch)  # specifies which data's heatmap to compute
            inp_rel = [inp.grad.detach().cpu().numpy() for inp in inp_list]
            inp_heatmap = self.DMI.adjust_input_relevance(inp_rel)

            # save intermediate attributions and activations
            if intermediate:
                relevances = get_attributions_from_hooks(self.MG, layer_out)
                activations = save_activations_as_numpy(layer_out)
                [h.remove() for h in forward_hooks]

            pred = pred.detach()#.cpu().numpy()

        return inp_heatmap, relevances, pred, activations


    def calc_layer_attribution(self, data_batch: torch.tensor, layer_name: str,
                                     neuron_selection: Union[list, iter], method="epsilon_plus_flat",
                                     intermediate=False):
        """
        calculates attribution beginning at intermediate layer inside network.
        """

        composite, data_batch = self.check_method_data(method, data_batch)

        # apply hook at selected_layer
        sel_layer_out = []

        try:
            selected_layer = self.MG.named_modules[layer_name]
        except KeyError:
            raise KeyError(f"{layer_name} is not analyzable or does not exist.")

        def get_tensor_hook(module, input, output):
            sel_layer_out.append(output)

        h = selected_layer.register_forward_hook(get_tensor_hook)

        relevances = {}
        activations = {}

        with composite.context(self.model) as modified:

            # hooks to get intermediate attributions and activations
            if intermediate:
                layer_out, forward_hooks = append_all_layers_forward_hooks(self.MG, intermediate)

            _ = modified(data_batch)
            layer_pred = sel_layer_out[0]

            torch.autograd.backward((layer_pred,), (neuron_selection.to(self.device),))

            inp_list = self.DMI.input_to_iterable(data_batch)  # specifies which data's heatmap to compute
            inp_rel = [inp.grad.detach().cpu().numpy() for inp in inp_list]
            inp_heatmap = self.DMI.adjust_input_relevance(inp_rel)

            # save intermediate attributions and activations
            if intermediate:
                relevances = get_attributions_from_hooks(self.MG, layer_out)
                activations = save_activations_as_numpy(layer_out)

        h.remove()
        if intermediate:
            [hf.remove() for hf in forward_hooks]

        return inp_heatmap, relevances, activations


    def batched_same_input_layer_attribution(self, data, relevances, layer_name, method, neuron_indices, get_mask_fct, result_indices, result_array, BATCH_SIZE=10):
        """
        Very efficient implementation of zennit forward and backward pass. Uses batched input data for same_input_layer_attribution.
        Args:

            neuron_indices -> get_mask_fct -> neuron_mask * relevances -> zennit -> result_array at result_indices

            data:
            relevances: torch array or None. Weighting of neuron_selection_mask.
            layer_name: string
            method: lrp method
            get_mask_fct: pointer to get_neuron_selection_mask function
            result_indices: indices of result_array, where output of neuron index is mapped to.
            result_array: where to save heatmaps
            BATCH_SIZE: of zennit forward and backward pass

        Returns:

        """

        n_neurons = len(neuron_indices)
        if n_neurons > BATCH_SIZE:
            batches = math.ceil(n_neurons / BATCH_SIZE)
            batch_size_tmp = BATCH_SIZE
        else:
            batch_size_tmp = n_neurons
            batches = 1

        # stack same images BATCH_SIZE times
        repeat_shape = [1] * len(data.shape[1:])
        data_batch = data.repeat(batch_size_tmp, *repeat_shape).detach()

        # gather indices for all batches before using zennit
        all_result_indices = []
        for b in range(batches-1):  # exclude last batch
            result_indices_batch = result_indices[b * batch_size_tmp: (b + 1) * batch_size_tmp]
            all_result_indices.append(result_indices_batch)

        # using a generator is much more memory efficient than saving every array in a list prior zennit call
        def neuron_mask_generator():
            for b in range(batches-1):  # exclude last batch
                neuron_indices_batch = neuron_indices[b * batch_size_tmp: (b + 1) * batch_size_tmp]
                neuron_selection_mask_batch = get_mask_fct(self.MG.named_modules[layer_name],
                                                                        self.MG.output_shapes[layer_name], neuron_indices_batch)
                if relevances:
                    neuron_selection_mask_batch = neuron_selection_mask_batch * relevances[layer_name]

                yield neuron_selection_mask_batch

        # apply zennit on gathered indices
        k_step = 0
        for result in self.same_input_layer_attribution(data_batch, layer_name, neuron_mask_generator(), method):
            result_array[all_result_indices[k_step]] = result[0]
            print(f"Batch {k_step+1}/{batches} finished.")
            k_step += 1

        # in last batch, BATCH_SIZE is most of the time different!
        neuron_indices_batch = neuron_indices[(batches-1) * batch_size_tmp: batches * batch_size_tmp]
        neuron_selection_mask_batch = get_mask_fct(self.MG.named_modules[layer_name],
                                                                self.MG.output_shapes[layer_name], neuron_indices_batch)
        if relevances:
            neuron_selection_mask_batch = neuron_selection_mask_batch * relevances[layer_name]
        data_batch = data_batch[:len(neuron_indices_batch)].detach()
        result_indices_batch = result_indices[(batches-1) * batch_size_tmp : (batches-1) * batch_size_tmp + len(neuron_indices_batch)]

        for result in self.same_input_layer_attribution(data_batch, layer_name, [neuron_selection_mask_batch],
                                                             method):
            t_attr = result[0]
            result_array[result_indices_batch] = t_attr
            print(f"Batch {batches}/{batches} finished.")


    def check_method_data(self, method: str, data_batch: torch.Tensor):
        """
        method returns zennit composite according to method name.
        data_batch has to be a leaf node since zennit is differentiating the model and saving the attribution inside
        the .grad attribute.
        """


        try:
            composite = COMPOSITES[method](canonizers=self.canonizer)
        except KeyError:
            raise KeyError(f"Either Method {method} or Canonizer not defined. Available methods are {list(COMPOSITES.keys())}.")

        if data_batch.is_leaf is not True:
            raise ValueError("input tensor must be a leaf node.")
        try:
            data_batch.requires_grad = True
        except:
            pass

        return composite, data_batch



