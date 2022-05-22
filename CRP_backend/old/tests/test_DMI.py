from CRP_backend.datatypes.test_interface import *

import torch

from test_model.ImageDataset import ImageDataset

from CRP_backend.zennit_API.API import ZennitAPI


if __name__ == "__main__":

    ID = ImageDataset("cpu")

    dataset = ID.dataset
    model = ID.model

    image, label = dataset[0]
    image_batch = image.unsqueeze(0)
    image_batch = ID.preprocess_data_batch(image_batch)

    ###############################

    test_softmax_layer(model)
    test_dataset(ID)
    test_model_inference(model, ID)
    test_to_iterable(model, ID)

