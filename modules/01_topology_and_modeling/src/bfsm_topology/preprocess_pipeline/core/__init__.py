from __future__ import annotations

from scipy.io import loadmat

from ..paths import BFM_MODEL_PATH
from .BFM09Model import BFM09ReconModel


def get_recon_model(model: str = "bfm09", **kwargs):
    if model != "bfm09":
        raise NotImplementedError(model)
    model_dict = loadmat(BFM_MODEL_PATH)
    return BFM09ReconModel(model_dict, **kwargs)
