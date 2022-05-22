import torch

from CRP_backend.tests.test_model.ImageDataset import ImageDataset

from CRP_backend.zennit_API.API import ZennitAPI
import matplotlib.pyplot as plt
from CRP_backend.feature_visualization.FilterVisualization import FilterVisualization

from CRP_backend.core.layer_specifics import *
from CRP_backend.core.model_utils import create_model_representation

if __name__ == "__main__":

    ID = ImageDataset("cpu")

    dataset = ID.load_dataset()
    model = ID.build_model()

    image, target = dataset[0]
    image_batch = image.unsqueeze(0)
    image_batch = ID.preprocess_data_batch(image_batch)

    MG = create_model_representation(model, image_batch)
    ZAPI = ZennitAPI(ID, MG, "cpu")

    FV = FilterVisualization(MG, ID, ZAPI)

    FV.run_analysis(0, 100)