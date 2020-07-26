import os
import sys
import time
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import copy, deepcopy
from pprint import pprint
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as D
from torchvision import transforms as T
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
try:
    from apex import amp
    USE_APEX = True
except:
    USE_APEX = False

from kuma_utils.nn.training import TorchTrainer
from kuma_utils.nn.logger import Logger
from kuma_utils.nn.snapshot import *
from kuma_utils.metrics import *
from kuma_utils.training import StratifiedGroupKFold

from configs import *
from metrics import sigmoid
from datasets import PandaDataset
from utils import MyScheduler, analyse_results


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="Specify config name in configs.py")
    parser.add_argument("--data", type=str, default='both',
                        help="Train on which dataset")
    parser.add_argument("--n_cpu", type=int, default=4,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only one fold")
    parser.add_argument("--drop_last", action='store_true', 
                        help="drop last data to avoid batchnorm error")
    parser.add_argument("--log", action='store_true',
                        help="write tensorboard log")
    parser.add_argument("--fp16", action='store_true',
                        help="train on fp16")
    parser.add_argument("--filter", type=str, default='',
                        help="filter data on specific columns")
    parser.add_argument("--default_train", type=str, default='train.csv',
                        help="default train path")
    opt = parser.parse_args()
    pprint(opt)
    N_CPU = opt.n_cpu

    cfg = eval(opt.config)
    print_config(cfg)

    if Path(opt.default_train).exists():
        train_df = pd.read_csv(opt.default_train)
    else:
        train_df = pd.read_csv(INPUT_PATH/'train.csv')

    if opt.filter != '':
        train_df = train_df.loc[train_df[opt.filter] == 1]
        print(f'filtered train_df by {opt.filter}. ({len(train_df)})')
    if opt.data == 'karolinska':
        train_df = train_df.loc[train_df.data_provider == 'karolinska']
        suffix = f'_{opt.data}'
    elif opt.data == 'radboud':
        train_df = train_df.loc[train_df.data_provider == 'radboud']
        suffix = f'_{opt.data}'
    else:
        suffix = ''
    print(f'Train on {opt.data} institutions.')

    images = train_df.image_id.values
    labels = train_df.isup_grade.values
    insts = train_df.data_provider.values
    if 'groups' in train_df.columns:
        groups = train_df.groups.values
    else:
        groups = None

    if opt.log:
        logger = Logger(f'results/{cfg.name}{suffix}')
    else:
        logger = DummyLogger('')

    if opt.fp16 and not USE_APEX:
        EPS = 1e-6
    else:
        EPS = 1e-8

    seed_everything(cfg.seed)

    if groups is None:
        skf = StratifiedKFold(n_splits=cfg.CV, shuffle=True, random_state=cfg.seed)
    else:
        skf = StratifiedGroupKFold(n_splits=cfg.CV, random_state=cfg.seed)

    scores = np.zeros(cfg.CV, dtype=np.float16)
    
    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_df, labels, groups)):

        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            # skip fold
            continue

        train_ds = PandaDataset(
            images=images[train_idx], labels=labels[train_idx], insts=insts[train_idx], 
            img_size=cfg.img_size, transform=cfg.transform['train'],
            use_cache=cfg.use_cache, return_index=cfg.return_index, bin_label=cfg.bin_label,
            mixup=cfg.mixup, mixup_alpha=8.0, separate_image=cfg.separate_image, cat_insts=cfg.cat_insts,
            root_path=INPUT_PATH)
        valid_ds = PandaDataset(
            images=images[valid_idx], labels=labels[valid_idx], insts=insts[valid_idx],
            img_size=cfg.img_size, transform=cfg.transform['test'],
            use_cache=cfg.use_cache, return_index=cfg.return_index, bin_label=cfg.bin_label,
            separate_image=cfg.separate_image, cat_insts=cfg.cat_insts,
            root_path=INPUT_PATH)

        train_loader = D.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, 
            num_workers=N_CPU, drop_last=opt.drop_last)
        valid_loader = D.DataLoader(
            valid_ds, batch_size=cfg.batch_size, shuffle=False, 
            num_workers=N_CPU, drop_last=opt.drop_last)
        
        model = deepcopy(cfg.model)
        if cfg.pretrained_path is not None:
            load_snapshots_to_model(cfg.pretrained_path, model=model)
            print(f'{cfg.pretrained_path} is loaded.')
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=EPS) # bigger eps for half precision
        if cfg.my_scheduler is None:
            scheduler = ReduceLROnPlateau(
                optimizer, 'min', factor=0.5, patience=2, cooldown=1, verbose=True, eps=EPS, min_lr=cfg.lr/200)
        else:
            scheduler = MyScheduler(optimizer, config=cfg.my_scheduler)

        if cfg.criterion.__class__.__name__  == 'JointOptimizationLoss':
            cfg.criterion.init_targets(labels[train_idx])
        
        if cfg.model.__class__.__name__ == 'IterativeSelfLearningModel':
            model.init_features(labels[train_idx], labels[valid_idx])
            model.n_jobs = N_CPU
            cfg.criterion.model = model
            
        NN_FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
            'loader_test': None,
            'criterion': cfg.criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'num_epochs': cfg.epochs,
            'stopper': cfg.stopper, 
            'event': cfg.event, 
            'logger': logger,
            'snapshot_path': Path(f'results/{cfg.name}{suffix}/fold{fold}.pt'),
            'eval_metric': cfg.metric,
            'log_metrics': cfg.log_metrics, 
            'info_format': '[epoch] time data loss metric logmetrics earlystopping',
            'info_train': True,
            'info_interval': 1,
            'resume': cfg.resume,
        }
        trainer = TorchTrainer(
            model, serial=f'fold{fold}', fp16=opt.fp16)
        trainer.apex_opt_level = 'O1'
        if cfg.model.__class__.__name__ == 'IterativeSelfLearningModel':
            trainer.all_inputs_to_model = True
        trainer.fit(**NN_FIT_PARAMS)

        if fold == 0 or fold == opt.limit_fold:
            if len(trainer.oof.shape) > 1:
                oof = np.zeros(
                    (train_df.shape[0], *trainer.oof.shape[1:]), dtype=np.float16)
            else:
                oof = np.zeros(train_df.shape[0], dtype=np.float16)
        
        oof[valid_idx] = trainer.oof
        if cfg.bin_label:
            pred = sigmoid(oof[valid_idx]).sum(1)
            scores[fold] = QWK(6)(labels[valid_idx], pred)
            analyse_results(np.clip(pred.round(), 0, 5), labels[valid_idx], insts[valid_idx])
        else:
            scores[fold] = QWK(6)(labels[valid_idx], oof[valid_idx])
            if oof.shape[1] > 1: # classification
                analyse_results(np.clip(oof[valid_idx, :6].argmax(1), 0, 5), labels[valid_idx], insts[valid_idx])
            else:
                analyse_results(np.clip(oof[valid_idx].round(), 0, 5), labels[valid_idx], insts[valid_idx])
    if opt.limit_fold >= 0:
        np.save(f'results/{cfg.name}{suffix}/oof{opt.limit_fold}', oof)
    else:
        np.save(f'results/{cfg.name}{suffix}/oof', oof)
    print(f'Scores: {scores}')
