import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from models.backbones import *
from models.senet import *
from models.activation import *
from models.layers import *


'''
Modified backbones
'''
def se_resnext50_32x4d_downsample():
    model = se_resnext50_32x4d(pretrained='imagenet')
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.layer0.add_module(
        'conv2', nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False))
    model.layer0.add_module('bn2', nn.BatchNorm2d(64))
    model.layer0.add_module('relu2', nn.ReLU(inplace=True))

    return model


class FeatureEfficientNet(EfficientNet):

    def forward(self, inputs):
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)
        return x


def drop_fc(model):
    if model.__class__.__name__ == 'FeatureEfficientNet':
        new_model = model
        nc = model._fc.in_features
    elif model.__class__.__name__ == 'RegNetX':
        new_model = nn.Sequential(*list(model.children())[0])[:-1]
        nc = list(model.children())[0][-1].fc.in_features
    elif model.__class__.__name__ == 'DenseNet':
        new_model = nn.Sequential(*list(model.children())[:-1])
        nc = list(model.children())[-1].in_features
    else:
        new_model = nn.Sequential(*list(model.children())[:-2])
        nc = list(model.children())[-1].in_features
    return new_model, nc


'''
New models
'''

class PatchPoolModel2(nn.Module):

    def __init__(self, base_model, patch_total=64, num_classes=6):
        super(PatchPoolModel2, self).__init__()

        self.N = patch_total
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, _, ch, w, h = x.shape
        x = x.view(bs*self.N, ch, w, h) # x: N bs x C x W x W
        x = self.encoder(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, self.N, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, self.N*w2, h2) # x: bs x C' x N W'' x W''
        x = self.head(x)

        return x

    def __repr__(self):
        return f'PatchPoolModel2({self.model_name})'


class IterativeSelfLearningModel(nn.Module):
    '''
    Implementation of 
    Deep Self-Learning From Noisy Labels: 
    https://arxiv.org/pdf/1908.02160.pdf
    '''

    def __init__(self, base_model, patch_total=64, num_classes=6, 
                 p=5, m=500, n_jobs=8, debug=False):
        super(IterativeSelfLearningModel, self).__init__()

        self.N = patch_total
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.flatten = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes)
        )

        self.p = p # p prototypes per class
        self.m = m # m samples to calc prototypes
        self.n_jobs = n_jobs
        self.debug = debug

    def init_features(self, train_labels, valid_labels):
        self.train_features = torch.zeros((len(train_labels), self.head[0].in_features))
        self.valid_features = torch.zeros((len(valid_labels), self.head[0].in_features))
        self.train_labels = torch.from_numpy(train_labels).float()
        self.valid_labels = torch.from_numpy(valid_labels).float()
        self.train_pseudo_labels = torch.from_numpy(train_labels).float()
        self.valid_pseudo_labels = torch.from_numpy(valid_labels).float()

    def feature(self, x, y=None, indices=None):
        # x: bs x N x C x W x W
        bs, _, ch, w, h = x.shape
        x = x.view(bs*self.N, ch, w, h)  # x: N bs x C x W x W
        x = self.encoder(x)  # x: N bs x C' x W' x W'
        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, self.N, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, self.N*w2, h2)  # x: bs x C' x N W'' x W''
        x = self.flatten(x)
        if indices is not None: # Update features
            if self.training:
                self.train_features[indices] = x.detach().clone().cpu()
            else:
                self.valid_features[indices] = x.detach().clone().cpu()
        return x

    def forward(self, x, y=None, indices=None):
        return self.head(self.feature(x, y, indices))

    def correct_labels(self):
        kmeans = KMeans(n_clusters=self.p, n_jobs=self.n_jobs)
        train_labels = self.train_labels.numpy()
        valid_labels = self.valid_labels.numpy()
        train_pseudo_labels = np.zeros(len(train_labels))
        valid_pseudo_labels = np.zeros(len(self.valid_labels))
        train_features = self.train_features.numpy()
        valid_features = self.valid_features.numpy()
        prototypes = []

        # Get prototypes
        for t in range(6):
            target_idx = np.where(train_labels==t)[0]
            if len(target_idx) > self.m:
                target_idx = np.random.choice(target_idx, self.m, replace=False)
            kmeans.fit(train_features[target_idx])
            prototypes.append(kmeans.cluster_centers_)

        # Correct labels
        for i, f in enumerate(train_features):
            scores = [cosine_similarity(f.reshape(1, -1), ps)[0].mean() for ps in prototypes]
            train_pseudo_labels[i] = np.argmax(scores)
        for i, f in enumerate(valid_features):
            scores = [cosine_similarity(f.reshape(1, -1), ps)[0].mean() for ps in prototypes]
            valid_pseudo_labels[i] = np.argmax(scores)
        if self.debug:
            print(f'train: {(train_labels!=train_pseudo_labels).sum()} labels replaced.')
            print(f'valid: {(valid_labels!=valid_pseudo_labels).sum()} labels replaced.')

        self.train_pseudo_labels = torch.from_numpy(train_pseudo_labels).float()
        self.valid_pseudo_labels = torch.from_numpy(valid_pseudo_labels).float()

    def __repr__(self):
        return f'IterativeSelfLearningModel({self.model_name})'
