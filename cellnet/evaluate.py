# TODO: use library?
import numpy as np
from scipy.optimize import linear_sum_assignment

def sMAPE_MAE_counts(P, GT):
  return 100 / len(P) * np.sum(np.abs(P - GT) / (np.abs(P) + np.abs(GT))), np.mean(np.abs(P - GT))

def sMAPE_MAE(pred, gt):
  P = np.array([len(ps) for ps in pred])
  GT = np.array([len(ps) for ps in gt])
  return sMAPE_MAE_counts(P, GT)

def prec_rec_f1(pred, gt, threshold):
  if len(pred) == 0 or len(gt) == 0:
      return (1.0 if len(pred) == 0 else 0.0), (1.0 if len(gt) == 0 else 0.0), (1.0 if len(pred) == 0 and len(gt) == 0 else 0.0)
  # pad with dummy points to handle unmatched cases
  pred = np.vstack([pred, np.full((max(0, len(gt) - len(pred)), 2), 1e10)])
  gt = np.vstack([gt, np.full((max(0, len(pred) - len(gt)), 2), 1e10)])


  cost_matrix = np.linalg.norm(pred[:, None] - gt[None, :], axis=-1)
  row_ind, col_ind = linear_sum_assignment(cost_matrix)

  matches = cost_matrix[row_ind, col_ind] <= threshold
  TP = np.sum(matches); FP = len(pred) - TP; FN = len(gt) - TP

  p = TP / (TP + FP) if (TP + FP) > 0 else 1.0
  r = TP / (TP + FN) if (TP + FN) > 0 else 1.0
  f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

  return p, r, f1

# def print_metrics(pred, gt, threshold):
#   p, r, f1 = zip(*[prec_rec_f1(p, g, threshold) for p, g in zip(pred, gt)])

#   smape, mae = sMAPE_MAE(pred, gt)
#   print(f"sMAPE: {smape:.2f}%, MAE: {mae:.2f}, Average Precision: {np.mean(p):.2f}, Average Recall: {np.mean(r):.2f}, Average F1: {np.mean(f1):.2f}")