# Overview
- Pytorch and our toolkit: kuma_utils(https://github.com/analokmaus/kuma_utils)
- manage experiments using config class (`configs.py`) 


# What worked for us
- Tile based model like Iafoss's kernel (`panda_models/PatchPoolModel2`)
    - Tile setting is 224 x 224 x 64 
    - We solve the problem as ordinal regression(Coral loss)
    - se-resnext50 is always the best backbone 
    - public LB: ~0.87
- The biggest challenge is how to deal with noisy labels
    - We read and implemented lots of papers (`models/noisy_loss.py`)
    - Online Uncertaity Sample Mining (OUSM) worked best (`models/noisy_loss/OUSMLoss`)
        - Data with noisy label should give big loss value
        - Exclude k samples with biggest loss in a mini batch during training can prevent overfitting to noisy samples
        - In order to learn general features, first 5 epochs w/o OUSM
    - public LB: ~0.89
- Data augmentation has two types: slide aug and tile aug
    - Slide aug: ShiftScaleRotate
    - Tile aug: ShiftScaleRotate, Flip, Dropout
    - public LB: ~0.90
- Ensemble noisy detection
    - LB score is very unstable depending on the seed value
    - We trained our previous SOTA setting with 10 different seeds, and detected noise labels based on loss
        - Noise flags are added to `train.csv`
    - Noise ratio 0.10 + OUSM(k=1) worked the best
    - publicLB: ~0.91


# Instructions
- fp16 training on nvidia apex is recommended (https://github.com/NVIDIA/apex)
- Configure `INPUT_PATH` in `configs.py` for your own environment
- Make sure all requirements in `requirements.txt` met
- Train with config `PatchBinClassification`
```
python3 train.py --config PatchBinClassification --fp16
```
Results(checkpoint, oof) will be in `results/(config name)/`