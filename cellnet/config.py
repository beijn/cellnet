import os, json
from types import SimpleNamespace as obj

DRAFT = True if os.uname().nodename == 'eli' else os.getenv('CELLNET_DRAFT', False)
CONFIG = os.getenv('CONFIG', 'default')
CROSSVAL = False if DRAFT else os.getenv('CELLNET_CROSSVAL', False)

__data_config = json.load(open('data/config.json'))

__cfg_base = dict(
  **__data_config,

  xnorm_type='imagenet',  # TODO: check 'image_per_channel' as well
  xnorm_params={},

  model_architecture='smp.UnetPlusPlus:attention',
  model_encoder='timm-mobilenetv3_large_100',

  epochs=351,
  lossf='MSE+BCE',
  lr_gamma=0.1,
  lr_steps=2.5,

  cropsize=256,
  batch_size=16,
  
  maxdist=26, 
  sigma=5.0,
)

__cfg_draft = dict(
  image_paths=__cfg_base['image_paths'][:1], #type:ignore
  model_architecture='smp.Unet',
  model_encoder='resnet34',
  epochs=2,
  cropsize=128,
  batch_size=2,
)

__cfg = __cfg_base
if CONFIG!='default': __cfg |= json.load(open(f'configs/{CONFIG}.json'))
if DRAFT: __cfg |= __cfg_draft
cfg = obj(**__cfg)
