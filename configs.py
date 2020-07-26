from pathlib import Path
from pprint import pprint

# import segmentation_models_pytorch as smp
from albumentations import *
from albumentations.pytorch import ToTensor, ToTensorV2

from kuma_utils.nn.training import *
from kuma_utils.metrics import QWK

from transforms import *
from models.backbones import *
from models.group_norm import convert_groupnorm
from models.batch_renorm import convert_batchrenorm
from models.noisy_loss import *
from utils import ClassificationEvent
from panda_models import *
from metrics import CustomQWK, CustomAccuracy


INPUT_PATH = Path('~/shared/kaggle-panda-challenge/').expanduser()
USE_PRETRAINED = True


'''
Misc.
'''

def print_config(cfg):
    items = [
        'name', 
        # general
        'patch_size', 'patch_dim',
        'resume', 'img_size', 'batch_size', 'lr', 'epochs', 'CV', 'seed',
        # dataset
        'use_cache', 'separate_image', 'return_index', 'bin_label', 'mixup',
        # 
        'model', 'criterion', 'metric', 'log_metrics', 'stopper', 'event', 'transform',
    ]
    print(f'\n----- Config -----')
    for key in items:
        try:
            value = eval(f'cfg.{key}')
            print(f'{key}: {value}')
        except:
            print('{key}: ERROR')
    print(f'----- Config -----\n')


'''
Configs start here
'''

class Baseline:

    name = 'baseline'

   # General
    img_size = 2
    patch_size = 56
    patch_dim = 8
    batch_size = 12
    lr = 2e-4
    CV = 5
    epochs = 60
    pretrained_path = None
    resume = False
    seed = 2020

    # Dataset
    use_cache = True
    separate_image = False
    return_index = False
    bin_label = False
    mixup = False
    cat_insts = False

    model = senet_mod(se_resnext50_32x4d, in_channel=3,
                      num_classes=6, pretrained=USE_PRETRAINED)
    criterion = nn.CrossEntropyLoss()
    metric = QWK(6).torch
    log_metrics = [CustomQWK(6), CustomAccuracy(6)]
    stopper = EarlyStopping(patience=15, maximize=True)
    event = NoEarlyStoppingNEpochs(20)
    my_scheduler = None
    transform = {
        'train': Compose([
            ShiftScaleRotate(scale_limit=0.0625, rotate_limit=15, p=0.75),
            MakePatches(patch_size, patch_dim, always_apply=True),
            Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                      [0.36357649, 0.49984502, 0.40477625], always_apply=True),
            HorizontalFlip(p=0.5), VerticalFlip(p=0.5),
            ToTensor()
        ]),
        'test': Compose([
            MakePatches(patch_size, patch_dim, always_apply=True),
            Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                      [0.36357649, 0.49984502, 0.40477625], always_apply=True),
            ToTensor()
        ]),
    }


class PatchBinClassification(Baseline):

    name = 'patch_bin_classification'

    # General
    img_size = 1
    patch_size = 224
    patch_dim = 8
    batch_size = 12
    lr = 2e-4
    CV = 5
    epochs = 60
    resume = False

    # Dataset
    use_cache = False
    separate_image = True
    return_index = False
    bin_label = True

    model = PatchPoolModel2(
        base_model=senet_mod(se_resnext50_32x4d, pretrained=USE_PRETRAINED),
        patch_total=patch_dim**2, num_classes=5,
    )
    criterion = OUSMLoss(k=1, loss='Coral', trigger=1)
    metric = CustomQWK(6, loss='Coral', p=1-1/batch_size, optim=False)
    stopper = EarlyStopping(15, maximize=True)
    event = ClassificationEvent(stopper=30)
    transform = {
        'train': {
            'tile': Compose([
                ShiftScaleRotate(scale_limit=0.0625, rotate_limit=15, p=0.5),
                MakePatches(patch_size, patch_dim, concat=False, dropout_ratio=0.05,
                            always_apply=True)
            ]),
            'augmentation': Compose([
                ShiftScaleRotate(scale_limit=0.0625, rotate_limit=15, p=0.5),
                HorizontalFlip(p=0.5), VerticalFlip(p=0.5), 
                Normalize([0.910, 0.819, 0.878],
                          [0.363, 0.499, 0.404], always_apply=True),
                ToTensor()
            ])
        },
        'test': {
            'tile': Compose([
                MakePatches(patch_size, patch_dim, concat=False, always_apply=True)
            ]),
            'augmentation': Compose([
                Normalize([0.910, 0.819, 0.878],
                          [0.363, 0.499, 0.404], always_apply=True),
                ToTensor()
            ])
        }
    }
