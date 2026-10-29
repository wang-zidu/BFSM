"""Scalar objectives; denominators and 255 scaling follow the fitting contract."""
import torch


def photo_loss(pred_img, gt_img, img_mask):
    rgb_distance = (pred_img.float() - gt_img).square().sum(dim=-1).sqrt()
    numerator = (rgb_distance * img_mask / 255).sum(dim=(1, 2))
    return (numerator / img_mask.sum(dim=(1, 2))).mean()


def lm_loss(pred_lms, gt_lms, weight, img_size=224):
    normalized_error = pred_lms / img_size - gt_lms / img_size
    per_landmark = normalized_error.square().sum(dim=-1)
    return (per_landmark * weight.reshape(1, -1)).sum(dim=1).mean()


def get_l2(tensor):
    return tensor.square().sum()


def reflectance_loss(tex, skin_mask):
    mask = skin_mask[..., None]
    mass = mask.sum()
    center = (tex * mask).sum(dim=1, keepdim=True) / mass
    deviation = (tex - center) * mask / 255.
    return deviation.square().sum() / (len(tex) * mass)
