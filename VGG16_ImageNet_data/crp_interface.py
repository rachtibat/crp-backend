from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
import torchvision
import torch.nn as nn
import torch
from pathlib import Path
import json

from crp.attribution import CondAttribution, AttributionGraph
from crp.graph import trace_model_graph
from crp.visualization import FeatureVisualization
from crp.receptive_field import ReceptiveField
from crp.helper import get_layer_names
from crp.concepts import ChannelConcept

from CRP_backend.interface import Image2D

from zennit.composites import COMPOSITES
from zennit.canonizers import SequentialMergeBatchNorm



class CRP_Interface(Image2D):

    def get_model(self, device):

        model = vgg16_bn(True).to(device)
        model.eval()
        return model

    def get_dataset(self):

        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

        path = "/home/achtibat/PycharmProjects/pytorch-lrp-meats-feature-visualization/model_data/ImageNet"
        imagenet_data = torchvision.datasets.ImageNet(path, transform=transform, split="val")   

        return imagenet_data

    def get_layer_map(self, model):

        layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])

        layer_map = {l_name: ChannelConcept() for l_name in layer_names}

        return layer_map

    def get_target_map(self):

        path = "/home/achtibat/PycharmProjects/CRP_backend_vigitlab/VGG16_ImageNet/label_map.json"
        with open(path, "r") as f:
            label_map = json.load(f)

        taget_map = {}
        for val in label_map.values():
            taget_map[val["label"]] = val["name"]

        return taget_map

    def decode_prediction(self, pred):

        return (["decode_prediction working"], [0.9])

    def get_composite_map(self):

        return COMPOSITES

    def get_canonizers(self):
        
        return [SequentialMergeBatchNorm()]

    def get_CondAttribution(self, model):

        return CondAttribution(model)


    def get_FeatureVisualization(self, attribution: CondAttribution, dataset, layer_map, device):

        path = "/home/achtibat/PycharmProjects/CRP_backend_vigitlab/VGG16_ImageNet"
        preprocess_fn = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocess_fn, path=path, device=device)

    def get_ReceptiveField(self, attribution: CondAttribution, single_sample: torch.Tensor):
        
        path = "/home/achtibat/PycharmProjects/CRP_backend_vigitlab/VGG16_ImageNet"
        return ReceptiveField(attribution, single_sample, path=path)


    def get_AttributionGraph(self, attribution: CondAttribution, single_sample: torch.Tensor, layer_map):

        graph = trace_model_graph(attribution.model, single_sample, list(layer_map.keys()))
        return AttributionGraph(attribution, graph, layer_map)