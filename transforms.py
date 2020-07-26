import numpy as np
import random
import cv2
from skimage import morphology
from skimage.color import rgb2hed, hed2rgb
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform


class MakePatches(DualTransform):

    def __init__(self, patch_size=128, patch_dim=4, concat=True, 
                 criterion='darkness', dropout_ratio=0.0,
                 always_apply=True, p=1.0):
        super(MakePatches, self).__init__(always_apply, p)
        self.patch_size = patch_size 
        if isinstance(patch_dim, (list, tuple)):
            self.patch_row = patch_dim[0]
            self.patch_total = patch_dim[0] * patch_dim[1]
        else:
            self.patch_row = patch_dim
            self.patch_total = patch_dim ** 2
        self.concat = concat
        self.criterion = criterion
        self.dropout = dropout_ratio
        self.idxs = None

    def get_blue_ratio(self, img):
        # (N, C, W, H) image
        rgbs = img.transpose(0, 3, 1, 2).mean(2).mean(2)  # N, C
        br = (100 + rgbs[:, 2]) * 256 / \
            (1 + rgbs[:, 0] + rgbs[:, 1]) / (1 + rgbs.sum(1))
        return br

    def tiles(self, img, sz=128, concat=True):
        w, h, ch = img.shape
        pad0, pad1 = (sz - w % sz) % sz, (sz - h % sz) % sz
        padding = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
        img = np.pad(img, padding, mode='constant', constant_values=255)
        img = img.reshape(img.shape[0]//sz, sz, img.shape[1]//sz, sz, ch)
        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, ch)
        valid_count = len(img)
        if len(img) < self.patch_total:
            padding = [[0, self.patch_total-len(img)], [0, 0], [0, 0], [0, 0]]
            img = np.pad(img, padding, mode='constant', constant_values=255)
        if self.criterion == 'darkness':
            idxs = np.argsort(img.reshape(
                img.shape[0], -1).sum(-1))[:self.patch_total]
        elif self.criterion == 'blue_ratio':
            idxs = np.argsort(self.get_blue_ratio(img) * -1)[:self.patch_total]
        self.idxs = idxs
        if concat:
            img = cv2.hconcat(
                [cv2.vconcat([_img for _img in img[idxs[i:i+self.patch_row]]]) \
                    for i in np.arange(0, self.patch_total, self.patch_row)])
        else:
            img = img[idxs]
            if self.dropout > 0:
                valid_count = min(valid_count, self.patch_total)
                drop_count = round(valid_count * self.dropout)
                if drop_count > 0:
                    drop_index = random.sample(range(valid_count), drop_count)
                    img[drop_index] = img[drop_index].mean()

        return img

    def mask_tiles(self, img, sz=128, concat=True):
        assert self.idxs is not None
        w, h, ch = img.shape
        pad0, pad1 = (sz - w % sz) % sz, (sz - h % sz) % sz
        padding = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
        img = np.pad(img, padding, mode='constant', constant_values=0)
        img = img.reshape(img.shape[0]//sz, sz, img.shape[1]//sz, sz, ch)
        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, ch)
        if len(img) < self.patch_total:
            padding = [[0, self.patch_total-len(img)], [0, 0], [0, 0], [0, 0]]
            img = np.pad(img, padding, mode='constant', constant_values=0)
        if concat:
            img = cv2.hconcat(
                [cv2.vconcat([_img for _img in img[self.idxs[i:i+self.patch_row]]]) \
                    for i in np.arange(0, self.patch_total, self.patch_row)])
        else:
            img = img[self.idxs]
        return img

    def apply(self, img, **params):
        return self.tiles(img, self.patch_size, self.concat)

    def apply_to_mask(self, img, **params):
        return self.mask_tiles(img, self.patch_size, self.concat)

    def get_transform_init_args_names(self):
        return ('patch_size', 'patch_total', 'concat', 'criterion', 'dropout')


class Scale(ImageOnlyTransform):

    def __init__(self, factor=0.5, always_apply=True, p=1.0):
        super(Scale, self).__init__(always_apply, p)
        self.factor = factor

    def apply(self, img, **params):
        return cv2.resize(img, (int(img.shape[1]*self.factor), int(img.shape[0]*self.factor)))

    def get_transform_init_args_names(self):
        return {'scale': self.factor}


class StainAug(ImageOnlyTransform):

    def __init__(self, factor=0.03, always_apply=False, p=0.5):
        super(StainAug, self).__init__(always_apply, p)
        self.factor = factor
    
    def stain_aug(self, image, k=0.03):
        # input must be rgb
        hed_image = rgb2hed(image)
        w = 1 - k + np.random.rand(3) * k * 2
        b = - k + np.random.rand(3) * k * 2
        for i in range(3):
            hed_image[:, :, i] = w[i] * hed_image[:, :, i] + b[i]
        return (hed2rgb(hed_image) * 255).astype(np.uint8)

    def apply(self, img, **params):
        return self.stain_aug(img, self.factor)

    def get_transform_init_args_names(self):
        return {'factor': self.factor}


class FlipColor(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):
        super(FlipColor, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if np.max(img) > 1.0:
            return 255 - img
        else:
            return 1.0 - img

    def get_transform_init_args_names(self):
        return ()


class DropTile(ImageOnlyTransform):

    def __init__(self, fill='mean', always_apply=False, p=0.1):
        super(DropTile, self).__init__(always_apply, p)
        self.fill = fill

    def apply(self, img, **params):
        if self.fill == 'mean':
            return np.full_like(img, int(np.mean(img)), dtype=np.uint8)
        elif self.fill == 'white':
            return np.full_like(img, 255)

    def get_transform_init_args_names(self):
        return {'fill': self.fill}
