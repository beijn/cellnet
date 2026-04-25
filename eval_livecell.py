# %% 
from cellnet.release import init_model, load_image, infer
from cellnet.data import load_points, imgid
import cellnet.plot as plotting
import matplotlib.pyplot as plt
from cellnet.evaluate import *
import pandas as pd

import torch, numpy as np
from skimage.feature import peak_local_max

import os, numpy as np

from pycocotools.coco import COCO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_test_images():
  coco = COCO('livecell/livecell_coco_test.json')
  return ['livecell/images/livecell_test_images/' + coco.loadImgs(img_id)[0]['file_name'] for img_id in coco.getImgIds()]

model = init_model('local')
model.eval()

evaluations = pd.DataFrame(columns=['image', 'predicted_count', 'true_count', 'sMAPE', 'MAE', 'precision', 'recall', 'f1'])

for i,img in enumerate(get_test_images()):
  X,m,s = load_image(img, model.settings)
  X = X[None,...]

  P = load_points(img.replace('images', 'points').replace('.jpg', '.json'))
  P = P[P[:,2]==1][:,:2] 
  P = plotting.points_inside_image(P, X.shape[2:4])

  Y = infer(X, model)

  Q = np.array(peak_local_max(Y[0,0], min_distance=5, threshold_abs=0.1)[:,[1,0]])

  counts = np.sum(Y)*model.settings["ymax"]
  smape, mae = sMAPE_MAE_counts(np.array([counts]), np.array([len(P)]))
  p,r,f1 = prec_rec_f1(Q, P, threshold=5)

  evaluations.loc[len(evaluations)] = [img, counts, len(P), smape, mae, p, r, f1]

  if i >= 1: continue
  os.system(f'mkdir -p plots')
  ax = plotting.overlay(x=X[0],y=Y[0],k=plotting.points_inside_image(P, X[0][0].shape), args_points={'colormap': 'black', 'alpha': 0.35,})
  plotting.points(ax, Q, radius=5, colormap='#0cbcfd', marker='.', lw=3, alpha=0.8)
  plotting.save(ax, f'plots/{imgid(img)}_eval.png')
  plt.close()

evaluations.to_csv('evaluations_livecell.csv', index=False)

print(evaluations.drop(['image'], axis=1).mean())
