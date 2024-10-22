import torch as t
import torch.nn as nn
import numpy as np


class SaliencyLoss(nn.Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, preds, labels, loss_type='cc'):
        losses = []
        if loss_type == 'cc':
            for i in range(labels.shape[0]):  # labels.shape[0] is batch size
                loss = loss_CC(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'kldiv':
            for i in range(labels.shape[0]):
                loss = loss_KLdiv(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'sim':
            for i in range(labels.shape[0]):
                loss = loss_similarity(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'nss':
            for i in range(labels.shape[0]):
                loss = loss_NSS(preds[i], labels[i])
                losses.append(loss)

        return t.stack(losses).mean(dim=0, keepdim=True)


def loss_KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    pred_map = pred_map / t.sum(pred_map)
    gt_map = gt_map / t.sum(gt_map)
    div = t.sum(t.mul(gt_map, t.log(eps + t.div(gt_map, pred_map + eps))))
    return div


def loss_CC(pred_map, gt_map):
    gt_map_ = (gt_map - t.mean(gt_map))
    pred_map_ = (pred_map - t.mean(pred_map))
    cc = t.sum(t.mul(gt_map_, pred_map_)) / t.sqrt(t.sum(t.mul(gt_map_, gt_map_)) * t.sum(t.mul(pred_map_, pred_map_)))
    return cc


def loss_similarity(pred_map, gt_map):
    gt_map = (gt_map - t.min(gt_map)) / (t.max(gt_map) - t.min(gt_map))
    gt_map = gt_map / t.sum(gt_map)

    pred_map = (pred_map - t.min(pred_map)) / (t.max(pred_map) - t.min(pred_map))
    pred_map = pred_map / t.sum(pred_map)

    diff = t.min(gt_map, pred_map)
    score = t.sum(diff)

    return score


def loss_NSS(pred_map, fix_map):
    '''ground truth here is fixation map'''

    pred_map_ = (pred_map - t.mean(pred_map)) / t.std(pred_map)
    mask = fix_map.gt(0)
    score = t.mean(t.masked_select(pred_map_, mask))
    return score


def AUC(img_pred, img_true, t=10, eps=10 ** (-8)):
    # img_true = [n, m]
    # img_pred = [n, m]
    list_metric = [i / t for i in range(1, t)]
    tpr = np.array([[0 for i in range(len(list_metric))]]).astype(float)
    fpr = np.array([[0 for i in range(len(list_metric))]]).astype(float)
    for i in range(len(list_metric)):
        true = np.array(img_true >= list_metric[i]).astype(int)
        pred = np.array(img_pred >= list_metric[i]).astype(int)
        TN = np.array(true + pred == 0).astype(int).sum()
        TP = np.multiply(true, pred).sum()
        FP = np.array(pred - true == 1).astype(int).sum()
        FN = np.array(true - pred == 1).astype(int).sum()
        TN = TN + eps
        TP = TP + eps
        FP = FP + eps
        FN = FN + eps
        tpr[0][i] = TP / (TP + FN)
        fpr[0][i] = FP / (FP + TN)
    roc_auc = np.concatenate((tpr, fpr), axis=0)
    roc_auc = np.array(sorted(roc_auc.T, key=lambda x: x[1])).T
    metric = 0.5 * roc_auc[0][0] * roc_auc[1][0]
    for i in range(1, len(list_metric)):
        metric += 0.5 * (roc_auc[0][i] + roc_auc[0][i - 1]) * (roc_auc[1][i] - roc_auc[1][i - 1])
    metric += 0.5 * (roc_auc[0][len(list_metric) - 1] + 1) * (1 - roc_auc[1][len(list_metric) - 1])
    return metric
