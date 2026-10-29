"""Independent implementations of the standard SSD decoding equations.

The expressions implement prior-relative center/size and landmark coordinates;
arithmetic order is compatible with the existing pretrained detector.
"""
from typing import List, Tuple, Union

import torch


def decode(loc: torch.Tensor, priors: torch.Tensor,
           variances: Union[List[float], Tuple[float, float]]) -> torch.Tensor:
    """Decode prior-relative offsets to (left, top, right, bottom) boxes."""
    center = priors[:, :2] + (loc[:, :2] * variances[0]) * priors[:, 2:]
    extent = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    lower = center - extent / 2
    upper = extent + lower
    return torch.cat((lower, upper), dim=1)


def decode_landm(pre: torch.Tensor, priors: torch.Tensor,
                 variances: Union[List[float], Tuple[float, float]]) -> torch.Tensor:
    """Decode five prior-relative 2D landmark offsets to image coordinates."""
    offsets = pre.reshape(pre.shape[0], 5, 2)
    coordinates = priors[:, None, :2] + (offsets * variances[0]) * priors[:, None, 2:]
    return coordinates.reshape(pre.shape[0], 10)
