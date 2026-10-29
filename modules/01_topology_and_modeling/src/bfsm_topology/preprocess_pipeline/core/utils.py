"""Filesystem and landmark-format helpers for the fitting interface."""
from pathlib import Path
import torch


def mymkdirs(path):
    Path(path).mkdir(exist_ok=True, parents=True)


def get_lm_weights(device):
    emphasis = [10. if 28 <= index < 31 or index >= 48 else 1. for index in range(68)]
    weights = torch.tensor(emphasis, device=device)
    return weights / weights.sum()


def save_obj(path, v, f, c):
    """Write RGB vertices and already one-based triangles, with six decimal places."""
    with Path(path).open("w") as stream:
        for index, vertex in enumerate(v):
            values = (*vertex[:3], *c[index, :3])
            stream.write("v " + " ".join(format(float(value), ".6f") for value in values) + "\n")
        stream.write("\n")
        for triangle in f:
            stream.write("f " + " ".join(str(int(value)) for value in triangle[:3]) + "\n")
