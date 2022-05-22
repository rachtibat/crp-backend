from urllib.request import HTTPPasswordMgrWithDefaultRealm
import torch
from CRP_backend.core.layer_specifics import LAYER_TYPES


class LayerNode:
    """
    contains meta information about each layer in pytorch model
    """

    def __init__(self, node):
        self.id = node.__repr__()
        self.scopeName = node.scopeName()
        if len(self.scopeName) == 0:
            self.scopeName = node.kind()
        self.next = []
        self.before = []
        self.layer_name = None


class ModelGraph:
    """
    As pytorch does not trace the model structure like keras/tensorflow, we need to do it ourselves.
    Thus, this class contains meta information about layer connections inside the pytorch model
    """

    def __init__(self, model, input_nodes):

        self.id_layer_dict = {}
        self.all_layer_nodes = []
        self.named_modules = {}
        self.output_shapes = {}
        self.model = model

        for node in input_nodes:
            layer_node = LayerNode(node)
            self.id_layer_dict[node.__repr__()] = layer_node
            self.all_layer_nodes.append(layer_node)

    def add_connection(self, input_node, end_node):
        """
        Add an entry for the connection between input_node and end_node
        """

        input_id = input_node.__repr__()
        end_id = end_node.__repr__()

        if input_id not in self.id_layer_dict:
            raise KeyError("Input Node does not exist")

        layer_node_in = self.id_layer_dict[input_id]

        if end_id in self.id_layer_dict:
            layer_node_end = self.id_layer_dict[end_id]
            if (layer_node_in in layer_node_end.before) and (layer_node_end in layer_node_in.next):
                # Node already traversed!
                return False
        else:
            layer_node_end = LayerNode(end_node)
            self.id_layer_dict[end_id] = layer_node_end
            self.all_layer_nodes.append(layer_node_end)

        layer_node_end.before.append(layer_node_in)
        layer_node_in.next.append(layer_node_end)

        return True

    def add_analyzable_module_names(self, model, layers_to_analyze=None):
        """
        function finds all intermediate layers (modules with no child module).
        creates dictionary with layer names as keys and corresponding module as value)

        param:
            layers_to_analyze: list or None. User can choose here which layers to analyze only.

        """

        all_modules = {}
        for name, layer in model.named_modules():

            if list(layer.children()) == []:
                for l_type in LAYER_TYPES:
                    if issubclass(layer.__class__, l_type):
                        # layer is analyzable subclass
                        if layers_to_analyze:
                            # only layers in <layers_to_analyze> are analyzed
                            if name in layers_to_analyze:
                                all_modules[name] = layer
                                break # for more efficiency
                        else:
                            # all layers with correct subclass are analyzed
                            all_modules[name] = layer
                            break # for more efficiency

        self.named_modules = all_modules

    def find_next_layers_of(self, layer_name):
        """
        find next analyzable layer starting at "layer_name"
        returns list
        """

        layer_node = None
        for ln in self.all_layer_nodes:
            # if layer_name == ln.scopeName.split("__module.")[-1]:
            if ln.scopeName.endswith(layer_name):
                layer_node = ln
                break

        if layer_node is None:
            raise KeyError("Layer not in Model Structure!")

        found_layers = []
        self._recursive_search(found_layers, layer_node)

        return found_layers

    def _recursive_search(self, found_layers, layer_node):
        """
        helper function. Traverses model graph to find next analyzable layer BEFORE
        """

        model_layer_names = list(self.named_modules.keys())
        found_bool = False

        for nt in layer_node.before:

            for layer_name in model_layer_names:

                if nt.scopeName.endswith(layer_name):
                    # found an analyzable layer
                    if layer_name not in found_layers:  # no duplicates
                        found_layers.append(layer_name)
                    found_bool = True
                    break

            # if found not analyzable layer, go deeper into this node
            if found_bool is False:
                self._recursive_search(found_layers, nt)

    def print(self):
        """
        print model structure
        """

        for layer in self.all_layer_nodes:
            print(layer.scopeName, "->", end="")
            for next_l in layer.next:
                print(next_l.scopeName + ", ", end="")

            if len(layer.next) == 0:
                print(" end")
            else:
                print("")


###################
################### functions
##################

def create_model_representation(model, dummy_input, layers_to_analyze=None, debug=False):
    """"
    MAIN function of model_utils.py.
    returns a ModelGraph object that contains meta information about the connections inside a pytorch model

    model: pytorch model
    dummy_input: data batch to trace the model. Has to be same shape as usual input to the model.

    """

    model.eval()
    
    #TODO: delete
   # from CRP_backend.zennit.composites import COMPOSITES
    #from CRP_backend.zennit.torchvision_2 import ResNetCanonizer
    #modified = COMPOSITES("epsilon_plus")(canonizers=[ResNetCanonizer()]).context(model)
    #model = modified.model

    # we use torch.jit to record the connections of all tensors
    traced = torch.jit.trace(model, (dummy_input,), check_trace=False)
    # inlined_graph returns a suitable presentation of the traced model
    graph = traced.inlined_graph

    if debug is True:
        dump_pytorch_graph(graph)

    # we concatenate all input and output tensor ids for each node as they are spread out in the original
    # torch.jit presentation
    node_inputs, node_outputs = collect_node_inputs_and_outputs(graph)

    # we search for all input nodes where we could start a recursive travers through the network graph
    input_nodes = get_input_nodes(graph, node_inputs, node_outputs)

    # initialize a model representation where we save the results
    Model_Graph = ModelGraph(model, input_nodes)

    # start recursive decoding of torch.jit record
    build_graph_recursive(Model_Graph, graph, input_nodes)

    # free gpu/cpu
    del traced, graph

    # add output shape information to ModelGraph using hooks as torch.jit presentation
    # has no (simple) shape information
    output_shapes = get_output_shapes(model, dummy_input)
    Model_Graph.output_shapes = output_shapes

    # assign module names and input/output layers
    Model_Graph.add_analyzable_module_names(model, layers_to_analyze)

    if debug is True:
        Model_Graph.print()

    return Model_Graph


def build_graph_recursive(Model_Graph, graph, input_nodes):
    """
    recursive function traverses the graph constructed by torch.jit.trace
    and records the graph structure inside our ModelGraph Class
    """

    for in_node in input_nodes:

        node_outputs = [i.unique() for i in in_node.outputs()]
        next_layers = find_next_layers(graph, node_outputs)

        if len(next_layers) == 0:
            return

        for next_node in next_layers:
            if not Model_Graph.add_connection(in_node, next_node):
                # Node already traversed!
                return

        build_graph_recursive(Model_Graph, graph, next_layers)


def find_next_layers(graph, node_outputs):
    """
    find layers where node_output is node_input
    """

    next_layers = []

    for node in graph.nodes():
        # if "aten" in node.kind():

        node_inputs = [i.unique() for i in node.inputs()]
        if set(node_inputs) & set(node_outputs):
            next_layers.append(node)

    return next_layers


def get_input_nodes(graph, layer_inputs, layer_outputs):
    """
    finds all input layers of the pytorch model.
    Uses output of collect_node_inputs_and_outputs and torch.jit.trace->inlined_graph
    """

    input_nodes = []

    for node in graph.nodes():
        # "aten" describes all real layers
        if "aten" in node.kind():

            name = node.scopeName()

            node_inputs = layer_inputs[name]
            if not find_overlap_with_output(node_inputs, layer_outputs):
                input_nodes.append(node)

    return input_nodes


def collect_node_inputs_and_outputs(graph):
    """
    helper function to beautify the output of torch.jit.trace and to calculate all input nodes of the model.
    this function merges all input tensors of a node and all output tensors into an array respectively
    """

    layer_inputs = {}
    layer_outputs = {}

    for node in graph.nodes():
        # "aten" nodes are real layers
        if "aten" in node.kind():

            name = node.scopeName()

            if name not in layer_inputs:
                layer_inputs[name] = []
                layer_outputs[name] = []

            [layer_inputs[name].append(i.unique()) for i in node.inputs()]
            [layer_outputs[name].append(i.unique()) for i in node.outputs()]

    return layer_inputs, layer_outputs


def find_overlap_with_output(node_inputs: list, layer_outputs):
    """
    used only in find_next_layers.
    Helper function to find a valid connection between two nodes
    """

    hit_bool = False

    for name in layer_outputs:

        node_outputs = layer_outputs[name]
        if set(node_inputs) & set(node_outputs):
            # if hit, not input node
            hit_bool = True
            break

    return hit_bool


def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph.
    Source: https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/pytorch_builder.py
    """

    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))


def get_output_shapes(model, data_batch: torch.tensor):
    """
    calculates the output shape of each layer using a forward pass.
    returns dictionary with layer names as keys and output shape without batch_dimension as values.

    """

    output_shapes = {}

    def generate_hook(name):

        def shape_hook(module, input, output):
            output_shapes[name] = output.shape[1:]

        return shape_hook

    hooks = []
    for name, layer in model.named_modules():

        if list(layer.children()) == []:
            for l_type in LAYER_TYPES:  # if analyzable leaf node
                if issubclass(layer.__class__, l_type):
                    shape_hook = generate_hook(name)
                    hooks.append(layer.register_forward_hook(shape_hook))
                    break

    data_batch.requires_grad = False
    _ = model(data_batch)
    try:
        # integer, long data types do not support gradients
        data_batch.requires_grad = True
    except:
        pass

    [h.remove() for h in hooks]

    return output_shapes

#TODO: memory leak fixing

if __name__ == "__main__":
    from pathlib import Path

    DML = DataModelLoader()
    model = DML.build_model().to("cuda")
    dataset = DML.loadImages(Path(""))
    images = dataset[0][0].to("cuda")

    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print("memory reserved ", r, "memory allocated ", a)

    MS = create_model_representation(model, images.unsqueeze(0), debug=False)

    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print("memory reserved ", r, "memory allocated ", a)
