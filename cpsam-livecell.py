# %% [markdown]
# # Cellpose-SAM as a baseline
# modelled after https://github.com/MouseLand/cellpose/blob/main/notebooks/test_Cellpose-SAM.ipynb
# %%
import numpy as np
from cellpose import models, core, io, utils
import matplotlib.pyplot as plt
import cellnet.plot as plotting
from cellnet.evaluate import *
from cellnet.data import masks2centers, imgid, load_points
import pandas as pd
import os, numpy as np

from pycocotools.coco import COCO

io.logger_setup()  # run this to get printing of progress

# Check if colab notebook instance has GPU access
if core.use_gpu() == False:
    raise ImportError("No GPU access, change your runtime")

model = models.CellposeModel(gpu=True)

def get_test_images():
  coco = COCO('livecell/livecell_coco_test.json')
  return ['livecell/images/livecell_test_images/'+coco.loadImgs(img_id)[0]['file_name'] for img_id in coco.getImgIds()]

evaluations = pd.DataFrame(columns=['image', 'predicted_count', 'true_count', 'sMAPE', 'MAE', 'precision', 'recall', 'f1'])

for i,img in enumerate(get_test_images()):
  X = io.imread(img)

  X = X

  P = load_points(img.replace('images', 'points').replace('.jpg', '.json'))
  P = P[P[:,2]==1][:,:2] 
  P = plotting.points_inside_image(P, X.shape)

  masks_pred, flows, styles = model.eval(X, niter=1000)
  Q = masks2centers(masks_pred) 

  smape, mae = sMAPE_MAE([Q], [P])
  p,r,f1 = prec_rec_f1(Q, P, threshold=5)

  evaluations.loc[len(evaluations)] = [img, len(Q), len(P), smape, mae, p, r, f1]

  if i >= 1: continue
  os.system(f'mkdir -p plots')
  ax = plotting.overlay(x=X,k=plotting.points_inside_image(P, X.shape), args_points={'colormap': 'black', 'alpha': 0.35,})
  plotting.points(ax, Q, radius=5, colormap='#0cbcfd', marker='.', lw=3, alpha=0.8)
  plotting.save(ax, f'plots/{imgid(img)}_eval.png')
  plt.close()

evaluations.to_csv('evaluations_cpsam_livecell.csv', index=False)

# %% 
print(evaluations.drop(['image'], axis=1).mean())