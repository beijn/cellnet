# %% [markdown]
# # CellNet on LIVECell
# drafted from train.py
# %% # Config and Imports

# TODO: refactor config setup, so its a standard config |= an experiment specific settings dict. Implment gridsearch via seperate notbook
# after experiment successful, summarize findings in respective ipynb and integrate into defaults
experiments = dict(
  default =     ('default', True), 
  draft =       ('epochs', 2),
  
  release =    ('epochs', 501),
  release_big =    ('epochs', 501),

  architecture = ('model_architecture', ['smp.Unet', 'smp.Unet:attention', 'smp.UnetPlusPlus']),  # NOTE: 'smp.Unet:deeper' fails, likely due to a bug in smp :[
  encoder_all =     ('model_encoder', ['resnet34', 'resnext50_32x4d', 'timm-resnest50d', 'timm-res2net50_26w_4s', 'timm-regnetx_064', 'timm-gernet_l', 'se_resnext50_32x4d', 'timm-skresnext50_32x4d', 'densenet161', 'xception', 'timm-efficientnet-b5', 'timm-mobilenetv3_large_100', 'dpn68b', 'vgg19_bn', 'mit_b2', 'mobileone_s4']),
  encoder_adaptedbestsmall =     ('model_encoder', ['resnet152', 'resnet34', 'timm-efficientnet-b5', 'timm-mobilenetv3_large_100', 'timm-resnest50d', 'xception', 'vgg19_bn']),
  encoder_big = ('model_encoder', ['resnet152', 'timm-efficientnet-l2', 'timm-efficientnet-b8', 'timm-mobilenetv3_large_100', 'timm-resnest200e', 'timm-resnest269e', 'inceptionresnetv2', 'inceptionv4', 'xception', 'vgg19_bn']),
  xnorm_per_channel = ('xnorm_type', 'image_per_channel'),
  loss =        ('lossf', ['MSE','BCE', 'MSE+BCE']), # 'KLD', 'MSE+BCE+KLD', 'Focal', 'MCC', 'Dice' Focal and MCC are erroneous (maybe logits vs probs). Dice is bad.  TODO Be inspired by arxiv:1907.02336
  # TODO: if linearcomb is better, then grid search weight
  sigma =       ('sigma', [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]),
  fraction =    ('fraction', [0.1, 0.25, 0.5, 0.75, 0.875, 1.0]),
  sparsity =    ('sparsity', [0.1, 0.25, 0.5, 0.75, 0.875, 1.0]),
)

from cellnet import data
import pandas as pd, json


dataset_experiment = 'livecell/0_train2percent.json'
image_paths = ['livecell/images/livecell_train_val_images/'+ann["file_name"] for ann in json.load(open(dataset_experiment))['images']][:3]

crossval_vals = data.imgid(image_paths[-1])

import os, torch, json
from types import SimpleNamespace as obj

EXPERIMENT = os.getenv('EXPERIMENT', 'default')
CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu'); print('device =', device)

DRAFT_MODE = os.getenv('CELLNET_DRAFT_MODE', False)

P, ps = experiments[EXPERIMENT]

modes = ['release', 'crossval', 'draft']
MODE = os.getenv('RELEASE_MODE', 'draft')
if MODE not in modes: MODE = 'crossval' 
# MODE undefined => interactive execution with draft; uknown custom tag defaults to crossval


if os.uname().nodename == 'eli': 
  MODE = 'draft'
  DRAFT_MODE = True
  data.DRAFT_MODE = True
  # TODO refactor

if MODE=='demo': 
  image_paths = [image_paths[0]]
  data_splits = [([image_paths[0]], [])]
elif MODE=='draft': data_splits = [([image_paths[0]], [image_paths[-1]])]
elif MODE=='release': data_splits = [(image_paths, [])]
else: #MODE=='crossval': 
  crossval_vals = [s.split(' ') for s in crossval_vals.split(' | ')]
  data_splits = [([p for p in image_paths if data.imgid(p) not in vs], 
                  [p for p in image_paths if data.imgid(p)     in vs]) 
                  for vs in crossval_vals]


CFG = obj(**(dict(
  EXPERIMENT=EXPERIMENT,
  cropsize=int(256*1.5),
  batch_size=int(16*1.5),
  data_splits=data_splits,
  device=f'{device}',
  epochs=1 if MODE in ('demo', 'draft') else int(251*2.5),
  fraction=1, 
  image_paths=image_paths,
  lossf='MSE+BCE',
  lr_gamma=0.1,
  lr_steps=2.5,
  maxdist=int(26*1.5), 
  MODE=MODE,
  model_architecture='smp.Unet' if MODE=='draft' else 'smp.UnetPlusPlus:attention',
  model_encoder='resnet34' if MODE=='draft' else 'timm-mobilenetv3_large_100',
  param=P,
  sigma=5*1.5,  # NOTE: do grid search again later when better convergence 
  sparsity=1,
  xnorm_params={},
  xnorm_type='imagenet',  # TODO: check 'image_per_channel' as well
  ) | {P: ps[-1] if type(ps) is list else ps}))


import torch
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

import albumentations as A; from albumentations.pytorch import ToTensorV2

import os, json
from types import SimpleNamespace as obj

from cellnet.data import *
import cellnet.plot as plot
import cellnet.debug as debug
from cellnet.data import key2text

# save the config to disk
with open('cfg.json', 'w') as f:  f.write(json.dumps(CFG.__dict__, indent=2))

# %% # Load Data

XNorm, CFG.xnorm_params = mk_XNorm(CFG, image_paths)

def mkAugs(mode):
  C = CFG.cropsize
  T = lambda ts:  A.Compose(transforms=[
    A.PadIfNeeded(C,C, border_mode=0, value=0),
    *ts,
    A.PadIfNeeded(C,C, border_mode=0, value=0),
    ToTensorV2(transpose_mask=True, always_apply=True)], 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True) 
  )

  vals = [A.D4(),
          ]

  return dict(
    demo  = T([]),
    test  = T([XNorm()]),
    val   = T([A.RandomCrop(C,C, p=1),
               *vals, XNorm()]),
    train = T([A.RandomCrop(C,C, p=1),
               A.RandomBrightnessContrast(p=1, brightness_limit=0.25, contrast_limit=0.25),
               #A.RandomSizedCrop(p=1, min_max_height=(CROPSIZE//2, CROPSIZE*2), height=CROPSIZE, width=CROPSIZE),  # NOTE: issue with resize is that the keypoint sizes will not be updated
               #A.Rotate(),
               #A.AdvancedBlur(),
               #A.Equalize(),
               #A.ColorJitter(), 
               #A.GaussNoise(),
               *vals, XNorm()])
  )[mode]

# %% # Plot data 
if MODE in ('draft', 'demo'): 
  def norm01(x):
    x = x.transpose(1,2,0)
    x = x/(x.max((0,1))-x.min((0,1)))
    return (x-x.min((0,1))).transpose(2,0,1)
  kp2hm, yunnorm, _ = mk_kp2mh_yunnorm(CFG)

  from math import prod
  def plot_grid(grid, **loader_kwargs):
    loader = mk_loader([CFG.image_paths[0]], cfg=CFG, bs=prod(grid), **loader_kwargs)
    B = next(iter(loader))
    B = batch2cpu(B, z=kp2hm(B))
    for b,ax in zip(B, plot.grid(grid, ([CFG.cropsize]*2))[1]):
      plot.overlay(b.x, b.z, b.m, None, None, CFG.sigma, ax=ax)

  for B in mk_loader(CFG.image_paths, cfg=CFG, bs=1, transforms=mkAugs('demo' if MODE=='demo' else 'test'), shuffle=False):
    b = batch2cpu(B, z=kp2hm(B))[0]
    if MODE=='demo': 
      #ax = plot.overlay(b.x); b.x = norm01(b.x)
      b.fg = 1-load_bgmask(i:=CFG.image_paths[0])
      #ax = plot.overlay(b.x)
      #ax = plot.overlay(b.x, None, None, b.k, b.l, args_points=dict(marker='.', radius=7, colormap={1: 'red', 2: '#7700ff'}, alpha=1))
      #ax = plot.overlay(b.x, b.z, None)
      #ax = plot.overlay(b.x, b.z, b.fg)
      ax = plot.overlay(b.x, b.z, b.m)
      from matplotlib.patheffects import PathPatchEffect, SimpleLineShadow, Normal
      ax.text(b.x.shape[2]/2, b.x.shape[1]*0+50, imgid(i), ha='center', va='center', fontsize=5000, color='#ffffff', 
              path_effects=[SimpleLineShadow(shadow_color="black", linewidth=800, ),Normal()])
    else: 
      ax = plot.overlay(b.x, b.z, b.m, b.k, b.l, CFG.sigma)

  plot_grid((3,3), transforms=mkAugs('val'))
  plot_grid((3,3), transforms=mkAugs('train'))

# %% # Create model 
plt.close('all')

import segmentation_models_pytorch as smp

mk_mk_model_smp = lambda cls, encoder_depth=5, **args: lambda cfg: cls(
  encoder_name=cfg.model_encoder, 
  encoder_weights=None,
  in_channels=1,
  classes=1,
  activation='sigmoid',
  encoder_depth=encoder_depth,
  decoder_channels=[16*2**d for d in range(encoder_depth)],
  **args
).to(device)

mk_model = lambda c: {
  'smp.Unet': mk_mk_model_smp(smp.Unet),
  'smp.Unet:attention': mk_mk_model_smp(smp.Unet, decoder_attention_type='scse'),
  'smp.Unet:deeper': mk_mk_model_smp(smp.Unet, encoder_depth=6),
  'smp.UnetPlusPlus': mk_mk_model_smp(smp.UnetPlusPlus),
  'smp.UnetPlusPlus:attention': mk_mk_model_smp(smp.UnetPlusPlus, decoder_attention_type='scse'),
}[c.model_architecture]

def save_model(model, CFG, _ymax):
  #B = next(iter(mk_loader([CFG.image_paths[0]], cfg=CFG, bs=1, transforms=mkAugs('test'), shuffle=False)))
  model.eval()

  # save a test in/out
  #os.makedirs(cachedir:=os.path.expanduser('~/.cache/cellnet'), exist_ok=True)
  #x = batch2cpu(B)[0].x[None]
  #np.save('./model_export_test_x_1.npy', x)
  #np.save('./model_export_test_y_1.npy', cpu(m(gpu(x, device=device))))

  model.save_pretrained('./model_export')  # specific to master branch of SMP. TODO: make more robust with onnx. But see problem notes in cellnet.yml
  os.remove('./model_export/README.md')

  settings = (CFG.__dict__ | {'ymax':float(_ymax)})
  with open('./model_export/settings.json', 'w') as f:  json.dump(settings, f, indent=2)

# %% # Train 

# def count(y): return yunnorm(y).sum().item()
def accuracy(y,z): 
  ny, nz = y.sum().item(), z.sum().item()
  return 1 - abs(ny - nz) / (nz+1e-9)

def epoch(model, kp2hm, lossf, dl, optim=None):
  l = 0; a = 0; b = 0
  for B in dl:
    x,m = B['image'].to(device), B['masks'][0].to(device)
    z = kp2hm(B).to(device)
    m = 1 # NOTE: quick hack to ignore mask REMOVE REMOVE REMOVE

    y = model(x)
    loss = lossf(y*m, z*m) 
    l += loss.item()
    a += accuracy(y*m, z*m)
    b += 1

    if optim is not None:
      loss.backward()
      optim.step()
      optim.zero_grad()

  return l/b, a/b


results = pd.DataFrame()
if not MODE=='draft': [os.makedirs(_p, exist_ok=True) for _p in ('preds', 'plots')]


# TODO unify log and results
def training_run(cfg, traindl, valdl, kp2hm, model):
  global results  
  p = cfg.__dict__[P]
  ti = cfg.ti; vi = cfg.vi

  optim = torch.optim.Adam(model.parameters(), lr=5e-3)
  sched = torch.optim.lr_scheduler.StepLR(optim, step_size=int(cfg.epochs/cfg.lr_steps)+1, gamma=cfg.lr_gamma)
  losses = dict(
    MSE = torch.nn.MSELoss(),
    BCE = torch.nn.BCELoss(),
    Focal = smp.losses.FocalLoss('binary'),
    MCC = smp.losses.MCCLoss(),
    Dice = smp.losses.DiceLoss('binary', from_logits=False),
    KLD = torch.nn.KLDivLoss(reduction='batchmean'),  
  )
  lossfs = [losses[l] for l in cfg.lossf.split('+')]
  lossf = lambda y,z: sum([lf(y,z) for lf in lossfs])

  log = pd.DataFrame(columns='tl vl ta va lr'.split(' '), index=range(cfg.epochs))
  
  for e in range(cfg.epochs):
    log.loc[e,'lr'] = optim.param_groups[0]['lr']
  
    model.train()
    log.loc[e,'tl'], log.loc[e,'ta'] = epoch(model, kp2hm, lossf, traindl, optim)
    sched.step() 
  
    if valdl is not None: 
      model.eval()
      with torch.no_grad():
        log.loc[e,'vl'], log.loc[e,'va'] = epoch(model, kp2hm, lossf, valdl) 

    if MODE=='draft': plot.train_graph(e, log, info={P: p}, key2text=key2text, clear=True)
  plot.train_graph(cfg.epochs, log, info={P: p}, key2text=key2text, accuracy=False) 

  row = dict(**{P: p}, ti=ti, vi=vi, **log.iloc[-1])

  # compute loss distributions for a few batches 
  # NOTE DUPLICATE COMPUTATIOM is very inefficient since we already do it once batch_size times, which should suffice.. now again n_batches*batch_size times..
  @debug.timeit
  def evaluate_performance():
    tvla = 'tl vl ta va'.split(' ')
    model.eval()
    # edit last row of results
    for k in tvla: row[k] = []  # type: ignore
    for b in range(n_batches := 10): 
      tl, ta = epoch(model, kp2hm, lossf, traindl)
      vl, va = epoch(model, kp2hm, lossf, valdl) if valdl is not None else (float('nan'), float('nan'))
      for k,v in zip(tvla, [tl, vl, ta, va]): row[k].append(v) # type: ignore

    for k in tvla: 
      row[k+'_mean'] = np.array(row[k]).mean() # type: ignore
      row[k+'_std'] = np.array(row[k]).std() # type: ignore
  if MODE != 'draft': evaluate_performance()

  results = pd.DataFrame([row]) if results.empty else pd.concat([results, pd.DataFrame([row])], ignore_index=True)
  
  # NOTE DUPLICATE COMPUTATION partial overlap with evaluate_performance. TODO easiest is to implement a caching decorator?
  @debug.timeit
  def plot_preds():
    # plot and save predictions to disk
    for i in ti+vi:
      B = next(iter(mk_loader([i], bs=1, transforms=mkAugs('test'), shuffle=False, cfg=cfg)))

      @data.wrap_padded
      def infer(x): return cpu(model(x.to(device)))

      model.eval()
      with torch.no_grad(): y = infer(B['image'])
      B = batch2cpu(B, z=kp2hm(B), y=y)[0]

      if (MODE=='release' or (i in vi)):  # plot all validation images (hopefully) once per CFG.param
        ax1 = plot.overlay(B.x, B.y, B.m, B.k, B.l, cfg.sigma) 
        ax2 = plot.diff   (B.y, B.z, B.m, B.k, B.l, cfg.sigma)
        ax3 = None
  
  plot_preds()

  return log

# mode == 2 => aka test augs => no cropping
def get_loader(cfg, ti, vi):
  loader = lambda m, ids: mk_loader(ids, bs = 1 if m==2 else CFG.batch_size, shuffle=False, cfg=cfg, transforms=mkAugs(m))
  return [loader('train', ti), loader('val', vi) if vi is not None and len(vi)>0 else None]
kp2hm, yunnorm, _ymax = mk_kp2mh_yunnorm(CFG)

model:torch.nn.Module = None # type: ignore
_ps = ps if type(ps) is list else [ps]
for p in [_ps[-1]] if MODE=='draft' else _ps:
  cfg = obj(**(CFG.__dict__ | {P: p}))
  if P in ['sigma']: kp2hm, yunnorm, _ymax = mk_kp2mh_yunnorm(cfg)

  for ti, vi in data_splits:
    cfg = obj(**(cfg.__dict__ | dict(ti=ti, vi=vi)))

    traindl, valdl = get_loader(cfg, ti, vi)

    model = mk_model(cfg)(cfg)
    log = training_run(cfg, traindl, valdl, kp2hm, model)
    log.to_csv(f'trainlog_{p}_{ti[0].split("/")[-1]}.csv', index=False) 
  
# %% # save model to disk
save_model(model, CFG, _ymax) # type: ignore

if type(results[P][0]) == str: results[P] = results[P].apply(lambda s: "'"+s+"'")
results.to_csv('results.csv', index=False, sep=';')

debug.print_times()

# %% # independently load and plot results
if MODE != 'release':  # HACK, fix this
  from cellnet import plot
  from cellnet.data import key2text
  from io import StringIO
  import ast, pandas as pd

  ## TODO those nans
  with open('results.csv', 'r') as f:
    nan = "0"
    s = f.read().replace('nan', nan).replace('NaN', nan).strip()
    while ";;"  in s: s = s.replace(";;", ";"+nan+";")
    while ";\n" in s: s = s.replace(";\n", ";"+nan+"\n")
    while "\n;" in s: s = s.replace("\n;", "\n"+nan+";")
    if s[-1] == ";": s = s + nan
    if s[0] == ";": s = nan + s

  cols = pd.read_csv(StringIO(s), sep=';').columns
  R = pd.read_csv(StringIO(s), sep=';', converters={col:ast.literal_eval for col in cols})
  R.rename(columns=dict(vi=key2text['vi']), inplace=True)
  plot.regplot(R, R.columns[0], key2text, sort_by=None)
