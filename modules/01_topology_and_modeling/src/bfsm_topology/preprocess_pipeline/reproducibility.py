"""Explicit process-wide controls for repeatable registration comparisons."""
import os
from pathlib import Path
import random
import sys


def configure(device, seed=0):
    import numpy as np
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        if torch.cuda.is_initialized():
            raise RuntimeError("Start a fresh Python process with CUBLAS_WORKSPACE_CONFIG=:4096:8")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.device(device).type == "cuda":
        from bfsm_core.runtime.deterministic_pytorch3d import install
        install()
