#!/usr/bin/env python

import os
import cv2
import numpy as np
import os.path as osp
import SimpleITK as sitk


def compute_specificity(SEG, GT):
    TN = np.sum(np.logical_not(np.logical_or(SEG, GT)))
    FP = np.sum(SEG) - np.sum(np.logical_and(SEG, GT))
    spec = TN / (TN + FP)
    return spec


def evaluation_sample(SEG_np, GT_np):
    quality=dict()
    SEG_np = np.uint8(np.where(SEG_np, 1, 0))
    GT_np = np.uint8(np.where(GT_np, 1, 0))
    SEG = sitk.GetImageFromArray(SEG_np)
    GT = sitk.GetImageFromArray(GT_np)

    # Compute the evaluation criteria
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    # Overlap measures
    overlap_measures_filter.Execute(SEG, GT)
    #quality["jaccard"] = overlap_measures_filter.GetJaccardCoefficient()
    quality["dice"] = overlap_measures_filter.GetDiceCoefficient()
    quality["false_negative"] = overlap_measures_filter.GetFalseNegativeError()
    #quality["false_positive"] = overlap_measures_filter.GetFalsePositiveError()
    quality["sensitive"] = 1 - quality["false_negative"]
    quality["specificity"] = compute_specificity(SEG_np, GT_np)

    # Hausdorff distance
    hausdorff_distance_filter.Execute(SEG, GT)
    quality["hausdorff_distance"] = hausdorff_distance_filter.GetHausdorffDistance()

    return quality

#######################################################
#######################################################

data_root = './dataset/'
with open(data_root + 'test_set.txt') as f:
    test_lst = f.readlines()
iids = [x.strip() for x in test_lst]
gt_files = [data_root + 'test_masks/' + x for x in iids]
pred_files = ['./new_output/KEN_main1/' + x for x in iids]
assert len(gt_files) == len(pred_files), 'The number of GT and pred must be equal'

#nthresh = 99
EPSILON = np.finfo(np.float).eps
#thresh = np.linspace(1./(nthresh+1), 1.-1./(nthresh+1), nthresh)

recall = np.zeros((len(gt_files)))
precision = np.zeros((len(gt_files)))
dice = np.zeros((len(gt_files)))
sensitive = np.zeros((len(gt_files)))
specificity = np.zeros((len(gt_files)))
hausdorff_distance = np.zeros((len(gt_files)))

for idx in range(len(gt_files)):
    print(gt_files[idx])
    gt = cv2.imread(gt_files[idx], 0)
    gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
    pred = cv2.imread(pred_files[idx], 0)
    gt = gt == 255
    pred = pred.astype(np.float) / 255

    zeros = 0
    zeros_pred = []
    '''
    for t in range(nthresh):
        bi_pred = pred > thresh[t]
        if np.max(bi_pred) == 0:
            zeros_pred.append(idx)
            zeros = zeros + 1
            continue
    '''
    if np.max(pred) != 0:
        intersection = np.sum(np.logical_and(gt == pred, gt))
        recall[idx] = intersection * 1. / (np.sum(gt) + EPSILON)
        precision[idx] = intersection * 1. / (np.sum(pred) + EPSILON)
        dice[idx] = evaluation_sample(pred, gt).get("dice")
        sensitive[idx] = evaluation_sample(pred, gt).get("sensitive")
        specificity[idx] = evaluation_sample(pred, gt).get("specificity")
        hausdorff_distance[idx] = evaluation_sample(pred, gt).get("hausdorff_distance")

recall = np.mean(recall, axis=0)
precision = np.mean(precision, axis=0)
dice = np.mean(dice, axis=0)
sensitive = np.mean(sensitive, axis=0)
specificity = np.mean(specificity, axis=0)
hausdorff_distance = np.mean(hausdorff_distance, axis=0)
F_beta = (1 + 0.3) * precision * recall / (0.3 * precision + recall + EPSILON)


with open(data_root + 'results.txt', 'a+') as fid:
    fid.write('{} {:10f} {:10f} {:10f} {:10f}\n'.format('FCN', np.max(dice),
    np.max(sensitive), np.max(specificity), np.max(hausdorff_distance)))

print('SEN =', np.max(sensitive))
print('SPC =', np.max(specificity))
print('DSC =', np.max(dice))
print('HD =', np.max(hausdorff_distance))
