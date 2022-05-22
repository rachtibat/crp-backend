import torch


def reset_gradients(model, data):
    """
    custom zero_grad() function
    """

    for p in model.parameters():
        p.grad = None

    if type(data) == torch.Tensor:
        data.grad = None
    else:
        # data is a iterable
        for d in data:
            d.grad = None


def append_all_layers_forward_hooks(MG, intermediate=False):
    """
    function appends hooks to all layers so that intermediate activations and gradients can be saved
    """
    layer_out = {}
    hooks = []

    def generate_hook(name):
        def get_tensor_hook(module, input, output):
            layer_out[name] = output
            if intermediate is True:
                output.retain_grad()

        return get_tensor_hook

    for layer_name in MG.named_modules:
        layer = MG.named_modules[layer_name]
        hooks.append(layer.register_forward_hook(generate_hook(layer_name)))

    return layer_out, hooks


def get_attributions_from_hooks(MG, layer_out):
    """
    Function returns the intermediate relevance of the output of the layer (normal forward direction).
    If no relevance was found, None is returned. Otherwise a torch.tensor with output shape of layer is returned.
    """

    relevances = {}
    for name in MG.named_modules:
            if name in layer_out and layer_out[name].grad is not None:
                relevances[name] = layer_out[name].grad.detach().cpu().numpy()
                layer_out[name].grad = None

    return relevances


def save_activations_as_numpy(layer_out):
    """
    Function returns intermediate activations as numpy arrays in layer_out dictionary
    """
    activations = {}
    for key in layer_out:
        activations[key] = layer_out[key].detach().cpu().numpy()

    return activations
