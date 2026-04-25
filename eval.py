# %% 
from cellnet.release import init_model, load_image
from cellnet.data import load_points, imgid
import cellnet.plot as plotting
import matplotlib.pyplot as plt

import torch, numpy as np
from skimage.feature import peak_local_max
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% 
model = init_model('local')
model.eval()

for img in [f'data/images/{i}.jpg' for i in [1]]: pass

X,m,s = load_image(img, model.settings)
X = X[None,...]#[:,:,:256,:256]

P = load_points(img.replace('images', 'points').replace('.jpg', '.json'))
P = P[P[:,2]==1][:,:2] 
P = plotting.points_inside_image(P, X.shape[2:4])

Y = model(torch.tensor(X).float().to(DEVICE)).detach().cpu().numpy()

Q = np.array(peak_local_max(Y[0,0], min_distance=5, threshold_abs=0.1)[:,[1,0]])

os.system(f'mkdir -p plots')
ax = plotting.overlay(x=X[0],y=Y[0],k=plotting.points_inside_image(P, X[0][0].shape), args_points={'colormap': 'black', 'alpha': 0.35,})
plotting.points(ax, Q, radius=5, colormap='#0cbcfd', marker='.', lw=3, alpha=0.8)
plotting.save(ax, f'plots/{imgid(img)}_eval.png')
plt.close()

# %% 

from cellnet.evaluate import *
print(f"Metrics for cellnet on image {img}:")

counts = np.sum(Y)*model.settings["ymax"]
print(f"Predicted count: {counts:.2f}, Ground truth count: {len(P)}")

smape, mae = sMAPE_MAE_counts(np.array([counts]), np.array([len(P)]))
print(f"sMAPE: {smape:.2f}%, MAE: {mae:.2f}")

p,r,f1 = prec_rec_f1(Q, P, threshold=5)
print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")
