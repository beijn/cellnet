# %% 
from cellnet.data import *
import cellnet.plot as plot
from cellnet.config import *

def norm01(x):
  x = x.transpose(1,2,0)
  x = x/(x.max((0,1))-x.min((0,1)))
  return (x-x.min((0,1))).transpose(2,0,1)
kp2hm, yunnorm, _ = mk_kp2hm_yunnorm(cfg)

from math import prod
def plot_grid(grid, **loader_kwargs):
  loader = mk_loader([cfg.image_paths[0]], cfg=cfg, bs=prod(grid), **loader_kwargs)
  B = next(iter(loader))
  B = batch2cpu(B, z=kp2hm(B))
  for b,ax in zip(B, plot.grid(grid, ([cfg.cropsize]*2))[1]):
    plot.overlay(b.x, b.z, b.m, None, None, cfg.sigma, ax=ax)

for B in mk_loader(cfg.image_paths, cfg=cfg, bs=1, transforms=mkAugs('test',cfg), shuffle=False):
  b = batch2cpu(B, z=kp2hm(B))[0]
  #ax = plot.overlay(b.x); b.x = norm01(b.x)
  b.fg = 1-load_bgmask(i:=cfg.image_paths[0])
  #ax = plot.overlay(b.x)
  #ax = plot.overlay(b.x, None, None, b.k, b.l, args_points=dict(marker='.', radius=7, colormap={1: 'red', 2: '#7700ff'}, alpha=1))
  #ax = plot.overlay(b.x, b.z, None)
  #ax = plot.overlay(b.x, b.z, b.fg)
  ax = plot.overlay(b.x, b.z, b.m)
  from matplotlib.patheffects import PathPatchEffect, SimpleLineShadow, Normal
  ax.text(b.x.shape[2]/2, b.x.shape[1]*0+50, imgid(i), ha='center', va='center', fontsize=5000, color='#ffffff', 
          path_effects=[SimpleLineShadow(shadow_color="black", linewidth=800, ),Normal()])

plot_grid((3,3), transforms=mkAugs('val',cfg))
plot_grid((3,3), transforms=mkAugs('train',cfg))
