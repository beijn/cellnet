# %%
import os, json, numpy as np
from collections import defaultdict
from cellnet.data import *

from pycocotools.coco import COCO
from pycocotools import mask
from scipy.ndimage import center_of_mass

def convert_coco_project(path, images_subset={}):
  coco = COCO(path)

  for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    if images_subset and img_info['file_name'] not in images_subset: continue
    if not os.path.exists(f'data/cache/points/{imgid(img_info["file_name"])}.npy'): continue
    I = imgid(img_info['file_name'])
    annotation_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(annotation_ids)

    centers = []
    for annotation in annotations:
      mask = coco.annToMask(annotation)
      if mask.sum() == 0: continue
      y,x = center_of_mass(mask.astype(bool))
      centers.append((x, y, annotation['category_id']))
    print(f'INFO: saving {len(centers)} points with labels {np.unique([c[2] for c in centers])} for image {I}.')
    np.save(f'data/cache/points/{I}.npy', np.array(centers))

def main(clear_cache=False):
  [os.makedirs(ospath(f'data/cache/{d}'), exist_ok=True) for d in ('points', 'masks')]
  if clear_cache:
    [os.remove(ospath(f'data/cache/{d}/{file}')) for d in ('points', 'masks') for file in os.listdir(ospath(f'data/cache/{d}'))]

  dataset_experiment = 'livecell/0_train2percent.json'
  images_subset = set(ann["file_name"] for ann in json.load(open(dataset_experiment))['images'])

  for annotation_file in ['livecell/livecell_coco_train.json', 'livecell/livecell_coco_val.json', 'livecell/livecell_coco_test.json']:
    convert_coco_project(os.path.join(annotation_file), images_subset)

  for image_file in images_subset:
    if not os.path.isfile(ospath(f'data/cache/points/{imgid(image_file)}.npy')):
      print(f'WARNING: No point annotations found for image {image_file}. Will autogenerate empty points.')
      np.save(ospath(f'data/cache/points/{imgid(image_file)}.npy'), no_points)
    if not os.path.isfile(ospath(f'data/cache/masks/{imgid(image_file)}.npy')):
      print(f'WARNING: No mask annotations found for image {image_file}. Will autogenerate empty masks.')
      #np.save(ospath(f'data/cache/masks/{imgid(image_file)}.npy'), no_masks)

if __name__=='__main__': main()
