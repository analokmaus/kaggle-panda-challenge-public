import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from efficientnet_pytorch import EfficientNet
from .xception import *
from .senet import *

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def resnet_mod(base_model, in_channel=3, num_classes=1000, pretrained=False):
    '''
    '''
    model = base_model(pretrained=pretrained)
    model_name = model.__class__.__name__

    # 3ch to 6ch
    if in_channel != 3:
        model.conv1 = nn.Conv2d(in_channel, 64,
                                kernel_size=7, stride=2, padding=3,
                                bias=False)

        if pretrained:  # adjust weight dim
            with torch.no_grad():
                trained_weight = model.conv1.weight
                model.conv1.weight[:, :] = torch.stack(
                    [torch.mean(trained_weight, 1)] * in_channel, dim=1)

    # modify output classes
    if num_classes is not None:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def densenet_mod(base_model, in_channel=3, num_classes=1000, pretrained=False):
    '''
    '''
    model = base_model(pretrained=pretrained)
    model_name = model.__class__.__name__

    if in_channel != 3:
        model.features.conv0 = nn.Conv2d(in_channel, 64,
                                        kernel_size=7, stride=2, padding=3,
                                        bias=False)
        if pretrained:  # adjust weight dim
            with torch.no_grad():
                trained_weight = model.features.conv0.weight
                model.features.conv0.weight[:, :] = torch.stack(
                    [torch.mean(trained_weight, 1)] * in_channel, dim=1)

    # modify output classes
    if num_classes is not None:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model


def enet_mod(model_size=6, in_channel=3, num_classes=1000):
    '''
    '''
    model = EfficientNet.from_pretrained(
        f'efficientnet-b{model_size}', num_classes=num_classes)
    model_name = model.__class__.__name__
    
    if in_channel != 3:  # adjust weight dim
        model._conv_stem = nn.Conv2d(
            in_channel, model._bn0.num_features, kernel_size=3, stride=2, bias=False)

        with torch.no_grad():
            trained_weight = model._conv_stem.weight
            model._conv_stem.weight[:, :] = torch.stack(
                [torch.mean(trained_weight, 1)] * in_channel, dim=1)

    return model


def xception_mod(in_channel=3, num_classes=1000, pretrained=False):
    '''
    '''
    model = xception(pretrained=pretrained)
    model_name = model.__class__.__name__

    if in_channel != 3:  # adjust weight dim
        model.conv1 = nn.Conv2d(in_channel, 32, kernel_size=7,
                                stride=2, padding=3, bias=False)

        with torch.no_grad():
            trained_weight = model.conv1.weight
            model.conv1.weight[:, :] = torch.stack(
                [torch.mean(trained_weight, 1)] * in_channel, dim=1)
    
    if num_classes != 1000:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def senet_mod(base_model, in_channel=3, num_classes=1000, pretrained=False):
    '''
    '''
    _pretrained = 'imagenet' if pretrained else None
    model = base_model(pretrained=_pretrained)
    model_name = model.__class__.__name__

    if in_channel != 3:
        model.layer0.conv1 = nn.Conv2d(in_channel, 64,
                                    kernel_size=7, stride=2, padding=3,
                                    bias=False)

        if pretrained:  # adjust weight dim
            with torch.no_grad():
                trained_weight = model.layer0.conv1.weight
                model.layer0.conv1.weight[:, :] = torch.stack(
                    [torch.mean(trained_weight, 1)] * in_channel, dim=1)

    # modify output classes
    if num_classes is not None:
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)

    return model
