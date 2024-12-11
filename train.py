# %% [markdown]
# # CellNet
# TODO adapt runnning scripts for new syntax: "$NAME" "$INDIR" "$INPUT" "$OUTDIR" "$CONFIG" with env vars CELLNET_DRAFT, CELLNET_CROSSVAL


# %% # Config and Imports
import os, json
import numpy as np, pandas as pd, torch

from cellnet.config import *
import cellnet.plot as plot, cellnet.debug as debug, cellnet.data as data
from cellnet.data import *

device = torch.device('cuda:0' if (CUDA:=torch.cuda.is_available()) else 'cpu'); print('device =', device); cfg.device=str(device)

from types import SimpleNamespace as obj
with open('settings.json', 'w') as f:  f.write(json.dumps(cfg.__dict__, indent=2))

_, cfg.xnorm_params = mk_XNorm(cfg, cfg.image_paths)

# %% # Create model 
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

def save_model(model, cfg, _ymax, path='./model_export'):
  model.eval()
  model.save_pretrained(path)
  os.remove(f'{path}/README.md')
  with open(f'{path}/settings.json', 'w') as f:  json.dump(cfg.__dict__ | {'ymax':float(_ymax)}, f, indent=2)
  # save a test in/out
  #os.makedirs(cachedir:=os.path.expanduser('~/.cache/cellnet'), exist_ok=True)
  #B = next(iter(mk_loader([CFG.image_paths[0]], cfg=CFG, bs=1, transforms=mkAugs('test', cfg), shuffle=False)))
  #x = batch2cpu(B)[0].x[None]
  #np.save('./model_export_test_x_1.npy', x)
  #np.save('./model_export_test_y_1.npy', cpu(m(gpu(x, device=device))))  # TODO use data.wrap_padded

# %% # Train 

# NOTE that with empty images the accuracy formula breaks -> TODO fix
def accuracy(y,z): 
  ny, nz = y.sum().item(), z.sum().item()
  return 1 - abs(ny - nz) / (nz+1e-9)

# def count(y): return yunnorm(y).sum().item()

def epoch(model, kp2hm, lossf, dl, optim=None):
  l = 0; a = 0; b = 0
  for B in dl:
    x,m = B['image'].to(device), B['masks'][0].to(device)
    z = kp2hm(B).to(device)

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

def training_run(cfg, traindl, valdl, kp2hm, model, crossval_idx):
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

  log = pd.DataFrame()
  row = {}
  
  for e in range(cfg.epochs):
    row['e']=e
    row['lr'] = optim.param_groups[0]['lr']
  
    model.train()
    row['tl'], row['ta'] = epoch(model, kp2hm, lossf, traindl, optim)
    sched.step() 
  
    if valdl is not None: 
      model.eval()
      with torch.no_grad():
        row['vl'], row['va'] = epoch(model, kp2hm, lossf, valdl) 

    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    if DRAFT: plot.train_graph(e, log, crossval_idx=crossval_idx, info={'config':CONFIG}, key2text=key2text, clear=True)

  return log

def get_loader(cfg, tidx, vidx):
  loader = lambda m, idx: mk_loader([cfg.image_paths[i] for i in idx], bs = 1 if m==2 else cfg.batch_size, shuffle=False, cfg=cfg, transforms=mkAugs(m,cfg))
  return [loader('train', tidx), loader('val', vidx) if vidx is not None and len(vidx)>0 else None]
kp2hm, yunnorm, _ymax = mk_kp2hm_yunnorm(cfg)

for crossval_idx, (tidx, vidx) in enumerate(cfg.crossval_splits_idx):
  cfg = obj(**(cfg.__dict__ | dict(train_idx=tidx, val_idx=vidx)))
  traindl, valdl = get_loader(cfg, tidx, vidx)
  model = mk_model(cfg)(cfg)
  log = training_run(cfg, traindl, valdl, kp2hm, model, crossval_idx)

  save_model(model, cfg, _ymax, f'model_export_{crossval_idx}')
  log.to_csv(f'log_{crossval_idx}.csv', index=False, sep=';')
  if not DRAFT: del model, log

debug.print_times()
