import os, json, numpy as np
from collections import defaultdict

from label_studio_sdk.converter.brush import decode_rle

from cellnet.data import *


def convert_labelstudio_project(path):
  points = defaultdict(list)
  masks = defaultdict(lambda: defaultdict(list))

  project = json.load(open(path))
  for task in project:
    I = imgid(task['file_upload'][9:])
    for annotation in task['annotations']:
      for result in annotation['result']:
        if 'rle' in result['value']:
          l = result['value']['brushlabels'][0].strip()
          mask = decode_rle(result['value']['rle']).reshape(result['original_height'], result['original_width'], -1)

          if not all((mask[...,0] == mask[...,j]).all() for j in range(mask.shape[-1])):
            raise ValueError(f'ERROR: Invalid mask: contains different channels, don\'t know how to unify.\n  Mask {result['id']} in {path} for image {I}.')
          else: mask = mask[...,0]

          if len(np.unique(mask)) > 2:
            print(f"NOTE: Mask contains multiple unqiue values. Will threshold at >0.\n  Mask {result['id']} in {path} for image {I}.")
          mask = mask > 0

          masks[I][l] += [mask]
          
        elif 'x' in result['value'] and 'y' in result['value']:
          x = result['value']['x']/100 * result['original_width']
          y = result['value']['y']/100 * result['original_height']
          l = result['value']['keypointlabels'][0].strip()
          points[I] += [(x,y, label2int[l])]

        else: print(f'WARNING: Unknown annotation type in {path} for image {I}. SKIPPING.\n {result}')

  for I in points:
    p = np.array(points[I])
    print(f'INFO: saving {len(points[I])} points with labels {np.unique(p[:,2])} for image {I}.')
    np.save(f'data/cache/points/{I}.npy', p)

  for I in masks:
    m = np.zeros((1+max(label2int[L] for L in masks[I]), *masks[I][list(masks[I].keys())[0]][0].shape))
    for L in masks[I]:
      if len(masks[I][L]) > 1: print(f'NOTE: Multiple masks for image {I} with label {L}. Will unify with pixelwise OR.')
      m[label2int[L]] = np.sum(np.stack(masks[I][L], axis=-1), axis=-1) > 0
    print(f'INFO: saving masks with labels {np.unique(list(masks[I].keys()))} for image {I}.')
    np.save(f'data/cache/masks/{I}.npy', m)

def main(clear_cache=True):
  [os.makedirs(ospath(f'data/cache/{d}'), exist_ok=True) for d in ('points', 'masks')]
  if clear_cache:
    [os.remove(ospath(f'data/cache/{d}/{file}')) for d in ('points', 'masks') for file in os.listdir(ospath(f'data/cache/{d}'))]

  for annotation_file in os.listdir(dir:=os.path.join('data', 'annotations')):
    if not annotation_file.endswith('.json'): continue
    convert_labelstudio_project(os.path.join(dir, annotation_file))
  
  for image_file in os.listdir(os.path.join('data', 'images')):
    if not os.path.isfile(ospath(f'data/cache/points/{imgid(image_file)}.npy')):
      print(f'WARNING: No point annotations found for image {image_file}. Will autogenerate empty points.')
      np.save(ospath(f'data/cache/points/{imgid(image_file)}.npy'), no_points)
    if not os.path.isfile(ospath(f'data/cache/masks/{imgid(image_file)}.npy')):
      print(f'WARNING: No mask annotations found for image {image_file}. Will autogenerate empty masks.')
    
if __name__=='__main__': main()
