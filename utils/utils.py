import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def iou(output, target):
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    predict = (output.detach().cpu().numpy() > 0).astype('int64')  # P
    target = target.detach().cpu().numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all()
    IoU = 1.0 * area_inter / (np.spacing(1) + area_union)

    return IoU


def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s
