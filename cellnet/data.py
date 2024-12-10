# NOTE on data format: this is all BHWC. Except Keypoints2Heatmap which is HWC. Torch needs BCHW, albumentations makes the conversions

from math import inf
import pathlib, os
from PIL import Image

from scipy.ndimage import gaussian_filter, distance_transform_edt
import numpy as np, torch, torch.utils.data

import albumentations as A
from types import SimpleNamespace as obj


label2int = {'Live Cell':1, 'Cell':1, 'cell':1, 'Dead cell/debris':2, 'Debris':2, 'debris': 2,
             'background': 1, 'Background': 1}

key2text = {'tl': 'Training Loss',     'vl': 'Validation Loss', 
            'ta': 'Training Accuracy', 'va': 'Validation Accuracy', 
            'ti': 'Training Image',    'vi': 'Validation Image',
            'bs': 'Batch Size',        'b' : 'Minibatches',       
            'e' : 'Epoch',             'lr': 'Learning Rate',
            'lossf': 'Loss Function',  'rmbad': 'Proportion of Difficult Labels Removed',
            'fraction': 'Fraction of Data',  'sparsity': 'Artificial Sparsity',  
            'sigma': 'Gaussian Sigma',        'maxdist': 'Max Distance',
            }


no_points = np.ones((1,3), dtype=np.int32)*2  ### NOTE TODO: change dtype according to load_points 
# NOTE: this is a hack to make the dataset work with images that have no points.
# TODO: make later code correctly work with empty point arrays

mapd = lambda d,f: {k:f(v) for k,v in d.items()}
dict2stack = lambda D, ids: np.stack([D[i] for i in (ids if ids!=None else sorted(D.keys()))], axis=0) if len(D)>0 else np.zeros((0,0,0,0))
stack2dict = lambda S, ids: {i:S[idx] for idx,i in enumerate(ids)}
wrapDictAsStack = lambda f, ids: lambda D: stack2dict(f(dict2stack(D, ids)), ids)


def gpu(x, device): return torch.from_numpy(x).float().to(device)
def cpu(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x) if isinstance(x, list) else x

ospath = lambda p: os.path.join(*p.split('/'))

def imgid(path): return pathlib.Path(path).stem

def batched(f):
  def inner(B):
    n_masks = len(B['masks'])
    R= torch.stack([f(
          image=b[0], masks=[b[m] for m in range(1,n_masks+1)], keypoints=b[n_masks+1], class_labels=b[n_masks+2]
        ) for b in zip(B['image'], *B['masks'], B['keypoints'], B['class_labels'])
      ], dim=0)
    return R
  return inner

batch2cpu = lambda B, z=None, y=None: [obj(**{k:cpu(v) for v,k in zip(b, 'xmklzy')}) 
              for b in zip(B['image'], B['masks'][0], B['keypoints'], B['class_labels'], 
              *([] if z is None else [z]), *([] if y is None else [y]))]


def ls(dir, ext='', stem=False): 
  return sorted([((lambda f: pathlib.Path(f).stem) if stem else (lambda x:x))(
    os.path.normpath(os.path.join(dir, f))) for f in os.listdir(dir) if f.endswith(ext)])

def _try_load(f, p, what):
  try: return f(p)
  except Exception as e: raise Exception(f"Could not load {what} for {p}.") from e

def load_image(path): 
  x = _try_load(lambda p: np.array(Image.open(p)), path, "image")
  information_channel = None
  for j in range(x.shape[-1]):
    if len(np.unique(x[...,j]))>1:
      if information_channel is not None and not ((x[...,j] == x[...,information_channel]).all()): 
        print(information_channel, j, np.unique(x[...,j]), np.unique(x[...,information_channel]))
        raise ValueError(f"Image {path} has {x.shape[-1]} channels with information. Currently code only works with grayscale images. QUICK FIX: convert to grayscale. TODO: adjust code to handle RGB images as well.")
      information_channel = j
    
  return x[...,[information_channel or 0]] 

def load_points(path): return _try_load(lambda p: np.load(f'data/cache/points/{imgid(p)}.npy'), path, "points")

def load_bgmask(path): return _try_load(lambda p: np.load(f'data/cache/masks/{imgid(p)}.npy')
                                        [label2int['background']], path, "masks")	# note the [label2int['background']]


class CellnetDataset(torch.utils.data.Dataset):
  def __init__(self, image_paths, sigma, maxdist, sparsity=1.0, fraction=1.0, batch_size=None, 
               transforms=None, override_points=None, not_sparsely_annotated_images=[], **_junk):
    super().__init__()
    if type(image_paths)==str: image_paths = [p for p in os.listdir(image_paths) if p.endswith('.jpg')]
    self.batch_size = batch_size or len(image_paths)
    self.maxdist=maxdist; self.fraction=fraction; self.sparsity=sparsity; self.not_sparsely_annotated_images=not_sparsely_annotated_images
    self.ids=image_paths; self.sigma=sigma; self.transforms = transforms if transforms else lambda **x:x    
    
    self.X = {i: load_image(i) for i in self.ids} # note albumentations=BHWC 但是 torch=BCHW
    if override_points is not None: self.P = override_points
    else: self.P = {i: load_points(imgid(i)) for i in image_paths} 
    # raise an error if not all image_paths have point annotations
    assert set(self.X.keys()) == set(self.P.keys()), f"The following images have no (cached) point annotations: {set(self.X.keys()) - set(self.P.keys())}"

    self._generate_masks(fraction=self.fraction, sparsity=self.sparsity)


  def _generate_masks(self, fraction=1.0, sparsity=1.0):
    assert fraction>0 and sparsity>0, "fraction and sparsity must be positive (0,1]"

    self.M = {}
    for I in self.ids:
      if I in self.not_sparsely_annotated_images: 
        self.M[I] = np.ones(self.X[I].shape[:2])
        print(f"INFO: Because {I} is (eg fully annotated or purposefully empty), foreground mask is set to 1 everywhere.")
      else:
        self.M[I] = mask_sparse(I, self.X[I], self.P[I], self.maxdist, channels=[0,1,2]) # TODO: derive channels from somewhere else

    if (f:=fraction) < 1.0: 
      _s = self.X[self.ids[0]].shape
      x,y = int(_s[0]*f), int(_s[1]*f)
      self.X = mapd(self.X, lambda a: a[:x,:y])
      self.M = mapd(self.M, lambda a: a[:x,:y])

    # filter out all points outside of X
    self.P = {i: np.array([(x,y,l) for x,y,l in self.P[i] 
                           if  0 <= x < self.X[i].shape[1] 
                           and 0 <= y < self.X[i].shape[0]]) 
                 for i in self.ids}
    self.P = {i: no_points if len(P)==0 else P for i,P in self.P.items()}

    if (s:=sparsity) < 1.0:
      self.P = mapd(self.P, lambda a: a[::int(1/s)])
   
  def get(self, n): return getattr(self, n)
  def set(self, n, to): setattr(self, n, to)

  def __len__    (self): return max(self.batch_size, len(self.X))
  def __getitem__(self, i): 
    i = self.ids[i % len(self.X)]
    return self.transforms(image=self.X[i], masks=[self.M[i]], keypoints=self.P[i][:,[0,1]], class_labels=self.P[i][:,2].astype(int))

  def map(self, f, dim='XY'):
    for n in dim: self.set(n, f(self.get(n)))


def mk_loader(image_paths, bs, transforms, cfg, shuffle=True, override_points=None):
  def collate(S):
    return dict(
      image = torch.stack([s['image'] for s in S]),
      masks = [torch.stack([s['masks'][i] for s in S]) for i in range(len(S[0]['masks']))],
      keypoints = [s['keypoints'] for s in S],
      class_labels = [s['class_labels'] for s in S],
    )

  from torch.cuda import device_count as gpu_count; from multiprocessing import cpu_count 
  return torch.utils.data.DataLoader(CellnetDataset(**(cfg.__dict__ | dict(image_paths=image_paths, transforms=transforms, batch_size=bs, override_points=override_points))), 
    batch_size=bs, shuffle=shuffle, collate_fn=collate, pin_memory=True, num_workers=8) # TODO: figure out and fix error and change back #8 if torch.cuda.is_available() else 2)


def mk_XNorm(cfg, norm_using_images=['all']):# -> tuple[Callable[..., Normalize], dict[str, list[Any]]]:
  if norm_using_images == ['all']: norm_using_images = list(set(p for s in cfg.data_splits for tv in s for p in s))
  X = np.stack([load_image(i) for i in norm_using_images], axis=0)
  params = dict(
    mean = list(X.mean(axis=(0,1,2))/255),
    std  = list(X.std (axis=(0,1,2))/255),
  )

  return (lambda **kw: A.Normalize(normalization='standard' if cfg.xnorm_type=='imagenet' else cfg.xnorm_type, **params, **kw)), params


def mk_kp2mh_yunnorm(cfg, norm_using_images=['all'], **overwrite_cfg):
  """Z-score norm improves DNN training according to @lecun2002efficient. BWHC"""
  if norm_using_images == ['all']: norm_using_images = list(set(p for s in cfg.data_splits for tv in s for p in tv))
  ds = CellnetDataset(**(cfg.__dict__ | dict(image_paths=norm_using_images) | overwrite_cfg))

  Y = np.stack([Keypoints2Heatmap(cfg.sigma, ynorm=lambda y:y, labels_to_include=[1])(
                 x.transpose(2,0,1),[m],pl[:,[0,1]],pl[:,2].astype(int)) 
                for x,m,pl in zip(*[v.values() for v in [ds.X, ds.M, ds.P]])], axis=0)
  
  ymax = Y.max() 
  ynorm   = lambda y: y/ymax  # norm to (0,1]
  yunnorm = lambda y: y*ymax
  # Y = ((Y - ymean) / ystd).astype(np.float32)  # unit norm, using dataset wide mean and std

  kp2mh = batched(Keypoints2Heatmap(cfg.sigma, ynorm, labels_to_include=[1]))
  return kp2mh, yunnorm, ymax


# HWC - this function works not with a batch dimension
def Keypoints2Heatmap(sigma, ynorm, labels_to_include=[1]):
  def f(image, masks, keypoints, class_labels):
    hw = image.shape[1:3] 
    if len(keypoints)==0: return torch.zeros(1,*hw).float()  # no keypoints
    Y = onehot(hw, np.array([[(x,y,l) for (x,y,*_),l in zip(keypoints, class_labels)]]))
    Y = Y[0][:,:,[l-1 for l in labels_to_include]]  # the [B][H,W,[Cs]] form is very important
    for c in range(Y.shape[-1]):
      Y[...,c] = gaussian_filter(Y[...,c], sigma=sigma, mode='constant', cval=0)
    return torch.from_numpy(ynorm(Y)).permute(2,0,1).to(torch.float32)  # HWC -> CHW
  return f

# this function expects [B][i][X,Y,L] and returns [B][H,W,C]
def onehot(hw, P, channels=None):
  if channels is None: channels = list(set.union({0,1}, set(np.unique(P[:,:,2].astype(int)))))
  A = np.zeros((len(P),*hw, max(channels))) 
  for b in range(len(P)):
    for x,y,l in P[b]:
      A[b, int(y), int(x), int(l)-1] = 1
  return A  # -> BHWC

def mask_sparse(id, x, p, maxdist, channels): 
  """Returns a mask that is 1 for annotated background or near annotated points, 0 elsewhere."""
  d = onehot(x.shape[0:2], p[None], channels=channels)[0]
  d = d.sum(axis=-1)  # all types of points are treated the same => we dont include the points for negative examples but we train on their image parts! :]
  d = distance_transform_edt(1-d)
  d = (d > maxdist).reshape(d.shape)  # type: ignore
  fg = (1-load_bgmask(id)) > 0
  return 1-(fg & d)[:,:,None].astype(np.float32)
