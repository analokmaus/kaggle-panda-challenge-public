import numpy as np
import pandas as pd
from pathlib import Path
from copy import copy
import random

import skimage.io
import cv2
from PIL import Image
# import openslide

import torch
import torch.utils.data as D
from torchvision import transforms as T
# from configs import INPUT_PATH


class PandaDataset(D.Dataset):
    def __init__(self, images, labels, insts=None, img_size=2, transform=None, bin_label=False,
                 root_path='', istest=False, return_index=True,
                 use_cache=False, mixup=False, mixup_alpha=1.0, separate_image=False, cat_insts=False):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.insts = insts
        self.img_size = img_size
        self.transform = transform
        self.bin_label = bin_label
        self.root = root_path
        self.istest = istest
        self.return_index = return_index
        self.use_cache = use_cache
        self.cache = {}
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.separate_image = separate_image
        self.cat_insts = cat_insts

    def __len__(self):
        return len(self.images)

    def _load_data(self, idx):
        cache_loaded = False
        if idx in self.cache.keys():
            image = self.cache[idx]
            cache_loaded = True
        else:
            if self.istest:
                fpath = str(self.root/'test_images'/f'{self.images[idx]}.tiff')
            else:
                fpath = str(self.root/'train_images'/f'{self.images[idx]}.tiff')
            image = skimage.io.MultiImage(fpath)[self.img_size]
            if self.use_cache and not self.separate_image:
                self.cache[idx] = image

        if self.transform:
            if self.separate_image:
                if cache_loaded:
                    pass
                else:
                    assert 'tile' in self.transform.keys()
                    image = self.transform['tile'](image=image)['image']  # N x 3 x W x H
                    if self.use_cache:
                        self.cache[idx] = image
                output = []
                for tile in image:
                    output.append(self.transform['augmentation'](image=tile)['image'])
                output = torch.stack(output)
            else:
                output = self.transform(image=image)['image']

        label = self.labels[idx]

        if self.insts is not None and self.cat_insts:
            output = torch.flatten(output)
            insts = self.insts[idx]
            if insts == 'karolinska':
                insts = torch.tensor([0.0])
            elif insts == 'radboud':
                insts = torch.tensor([1.0])
            output = torch.cat((output, insts))
            
        return output, label

    def __getitem__(self, idx):
        image, label = self._load_data(idx)

        if self.mixup:
            idx2 = np.random.randint(0, len(self.images))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            image2, label2 = self._load_data(idx2)
            image = torch.cat([torch.Tensor([lam]), image.view(-1), image2.view(-1)])
            label = lam * label + (1 - lam) * label2

        if self.bin_label:
            if self.mixup:
                label_dec = label - label_int
                label2 = torch.zeros(5)
                label2[:label_int] = 1.0
                if label_int < 5:
                    label2[label_int] = label_dec
            else: 
                label2 = torch.zeros(5)
                label2[:label] = 1
        else:
            label2 = label

        if self.return_index:
            return image, label2, idx
        else:
            return image, label2
