import numpy as np
from pysaliency.roc import general_roc
from pysaliency.numba_utils import auc_for_one_positive
import torch


def _general_auc(positives, negatives):
    if len(positives) == 1:
        return auc_for_one_positive(positives[0], negatives)
    else:
        return general_roc(positives, negatives)[0]


def log_likelihood(log_density, fixation_mask, weights=None):
    # 确保 weights 在 log_density 的设备上
    if weights is None:
        weights = torch.ones(log_density.shape[0], device=log_density.device)

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask

    # 确保 dense_mask 在 log_density 的设备上
    dense_mask = dense_mask.to(log_density.device)

    # 打印日志以调试
    # print(f"log_density shape: {log_density.shape}")
    # print(f"dense_mask shape: {dense_mask.shape}")

    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    # 确保 fixation_count 在 log_density 的设备上
    fixation_count = fixation_count.to(log_density.device)

    ll = torch.mean(
        weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )

    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)


def nss(log_density, fixation_mask, weights=None):
    if weights is None:
        weights = torch.ones(log_density.shape[0], device=log_density.device)

    weights = weights / weights.sum()

    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask

    # print(f"log_density shape: {log_density.shape}")
    # print(f"dense_mask shape: {dense_mask.shape}")

    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    # 确保 log_density 是浮点数
    density = torch.exp(log_density)

    # 计算均值和标准差
    mean, std = torch.std_mean(density, dim=(-1, -2), keepdim=True)

    # 避免标准差为零的情况
    std = torch.clamp(std, min=1e-8)

    # 计算标准化显著性图
    saliency_map = (density - mean) / std

    # 计算 NSS
    nss = torch.mean(
        weights * torch.sum(saliency_map * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )

    return nss


def auc(log_density, fixation_mask, weights=None):
    if weights is None:
        weights = torch.ones(log_density.shape[0], device=log_density.device)

    weights = weights / weights.sum()

    def image_auc(log_density, fixation_mask):
        if isinstance(fixation_mask, torch.sparse.IntTensor):
            dense_mask = fixation_mask.to_dense()
        else:
            dense_mask = fixation_mask

        # 转换为 numpy 数组进行 AUC 计算
        positives = log_density[dense_mask.bool()].detach().cpu().numpy().astype(np.float64)
        negatives = log_density.flatten().detach().cpu().numpy().astype(np.float64)

        # 计算 AUC
        auc_value = _general_auc(positives, negatives)
        return torch.tensor(auc_value, device=log_density.device)

    auc_values = torch.tensor([
        image_auc(log_density[i], fixation_mask[i]) for i in range(log_density.shape[0])
    ], device=log_density.device)

    return torch.mean(weights * auc_values)

