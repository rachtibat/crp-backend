import torch

from CRP_backend.tests.test_model.ImageDataset import ImageDataset

from CRP_backend.zennit_API.API import ZennitAPI
import matplotlib.pyplot as plt

from CRP_backend.core.layer_specifics import *
from CRP_backend.core.model_utils import create_model_representation

if __name__ == "__main__":

    ID = ImageDataset("cpu")

    dataset = ID.load_dataset()
    model = ID.build_model()


    image, target = dataset[0]
    image_batch = image.unsqueeze(0)
    image_batch = ID.preprocess_data_batch(image_batch)

    pred = model(image_batch)

    ################

    MG = create_model_representation(model, image_batch)
    ZAPI = ZennitAPI(ID, MG, "cpu")

    heatmap = ZAPI.calc_attribution(image_batch, [target], method="epsilon_plus")

    plt.title("heatmap")
    plt.imshow(heatmap)

    ####################


    channels = list(range(120))
    neuron_selection_mask = get_neuron_selection_mask(MS.named_modules["l6"], MS.output_shapes["l6"], channels)


    i = 0
    for result in ZAPI.same_input_layer_attribution(image_batch, MS.named_modules["l6"], neuron_selection_mask):
        #plt.figure(i)
        #plt.imshow(result)
        #plt.title(i)
        print(result.shape)
        i += 1


    #
    # for k in [0,1,2]:
    #     for result in ZAPI.same_input_layer_attribution(image_batch, MS.named_modules["l8"], [neuron_selection_mask[k]]):
    #         #plt.figure(k+10)
    #         #plt.imshow(result)
    #         plt.title(k)
    #         print(result.shape)
    #         k += 1


