
import numpy as np
import torch
from torch.nn.modules.loss import MSELoss, L1Loss
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from CRP_backend.core.layer_specifics import sum_relevance, get_neuron_selection_mask

from CRP_backend.zennit_API.API import ZennitAPI
from CRP_backend.core.model_utils import create_model_representation
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
import torch
import warnings
from lucent.optvis import param


class GradientAscent:

    def __init__(self, MG, DMI):

        self.available = True # whether generation of synthetic images is available for this experiment

        self.DMI = DMI
        self.MG = MG

        self.image_shape = self.get_image_shape()
        #self.layer_name_to_lucid = self.compare_lucent_layers()

        #self.lucent_transforms, self.lucent_steps = DMI.get_lucent_transform()

    def get_image_shape(self):

        image, _ = self.DMI.get_data_sample(0)

        if len(image.shape[1:]) != 3 or image.shape[1] == 1:
            self.available = False
            warnings.warn("Generation of synthetic samples for this data shape not supported.")
            return

        channel, height, width = image.shape[1:]
        return height, width, channel


    def gen_synthetic(self, layer_name, filter_indices: list, iterations=200):

        model = self.MG.model
        for p in model.parameters():
            p.requires_grad = False

        filter_indices = np.array(filter_indices) # for np functionality

        params, optimized_image = param.image(self.image_shape[0], self.image_shape[1], batch=len(filter_indices))

        optimizer = torch.optim.Adam(params, lr=0.05)
        hyper = self.DMI.get_synthetic_transform()
        transforms, iterations = hyper[0], hyper[1]
        transforms.append(
                    torch.nn.Upsample(size=(self.image_shape[0], self.image_shape[1]), mode="bilinear", align_corners=True)
                )
        transforms = T.Compose(transforms)

        output_layer = [0]
        def my_hook(self, input, output):
            
            output_layer[0] = output

        h = self.MG.named_modules[layer_name].register_forward_hook(my_hook)

        #mask_batch = get_neuron_selection_mask(self.MG.named_modules[layer_name],
         #                                                               self.MG.output_shapes[layer_name], filter_indices)
        
        batch_indices = np.arange(0, len(filter_indices))
        debug_info = iterations/10
        for e in range(iterations):

            synthetic_image = optimized_image()
            synthetic_image = self.DMI.preprocess_data_batch(synthetic_image)
            synthetic_image = transforms(synthetic_image)
            _ = model(synthetic_image)

            layer_out = output_layer[0] 

            loss = -layer_out[batch_indices, filter_indices].sum()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if e % debug_info == 0:

                yield optimized_image().detach().cpu().numpy(), "{:.2f}".format(e/iterations)
            
        yield optimized_image().detach().cpu().numpy(), "1.00"

        h.remove()
        for p in model.parameters():
            p.requires_grad = True





if __name__ == "__main__":

    def save_input_image(image: torch.tensor, e):

        picture = image.cpu().detach().numpy()
        picture = np.moveaxis(picture, 0, -1)
        plt.imsave(f"{e}_input.png", picture)


    device = "cuda:0"
    experiment = "VGG16_ImageNet"
    config_message = load_config(experiment)
    DMI = load_model_and_data(experiment, device, config_message)
    MG = create_model_representation(DMI.model, DMI.get_data_sample(0)[0])
    ZAPI = ZennitAPI(DMI, MG, device)


    GA = GradientAscent(MG, DMI)

    for imgs, e in GA.gen_synthetic("features.28", [0, 10, 22]):
        k = 0
        for img in imgs:
            save_input_image(img, f"{k}_{e}")
            k += 1

