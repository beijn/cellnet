# %% [markdown]
# # Cellpose-SAM as a baseline 
# modelled after https://github.com/MouseLand/cellpose/blob/main/notebooks/test_Cellpose-SAM.ipynb
# %% 
import numpy as np
from cellpose import models, core, io, utils
import matplotlib.pyplot as plt

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")

model = models.CellposeModel(gpu=True)

# %% 
# load image from ../data/images/1.jpg
images = [f'{i}.jpg' for i in [1,2,4]]
X = [io.imread(f"../data/images/{f}").transpose(2, 0, 1) for f in images]  #[:,:200,:200]

gt_points = [np.load(f"../data/cache/points/{i}.npy") for i in [1,2,4]]
gt_points = [ps[ps[:,2] == 1][:,:2] for ps in gt_points]

# %% 
masks_pred, flows, styles = model.eval(X, niter=1000) 

# %% 
from cellnet import plot as plotting
import os

os.system('mkdir -p ../plots/cpsam')

for i, (x, y, p) in enumerate(zip(X, masks_pred, gt_points)):
    ax = plotting.image(x)
    if p is not None: 
      plotting.points(ax, p, radius=5, colormap=[1,0,0], marker='o', lw=3)

    for o in utils.outlines_list(y):
        ax.plot(o[:,0], o[:,1], color=[1,1,0.3], lw=200, ls="-")

    plotting.save(ax, f'../plots/cpsam/{i+1}.png'); plt.close()

print("Predicted counts: ")
print({i: len(np.unique(masks_pred[j])) for j,i in enumerate(images)})

# %% 
from cellnet.data import masks2centers
from cellnet.evaluate import sMAPE_MAE, prec_rec_f1

pred = [masks2centers(masks) for masks in masks_pred]

print(f"sMAPE: {sMAPE_MAE(pred, gt_points)[0]:.2f}%")
print(f"MAE: {sMAPE_MAE(pred, gt_points)[1]:.2f}")

for i, y, p in zip(images, masks_pred, gt_points):
  p, r, f1 = prec_rec_f1(masks2centers(y), p, threshold=10)
  print(f"{i}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")
