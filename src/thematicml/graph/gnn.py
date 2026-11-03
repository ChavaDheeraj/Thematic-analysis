"""Graph neural network availability.

GNNs are neural networks and are *not* part of the classical pipeline. Training lives in
:mod:`thematicml.gnn`: with PyTorch Geometric when it is installed (``pip install thematicml[gnn]``),
otherwise with the built-in NumPy GraphSAGE, which also runs in the browser edition.
``THEMATICML_GNN_BACKEND=numpy`` or ``pytorch`` chooses one explicitly. GNN results are always
reported next to classical link-prediction heuristics scored on the same held-out edges.
"""
from __future__ import annotations

import os


class OptionalDependencyError(ImportError):
    pass


INSTALL_HINT = ("Install PyTorch Geometric with: pip install thematicml[gnn] (CPU wheels: pip install torch "
                "--index-url https://download.pytorch.org/whl/cpu, then pip install torch_geometric)")


def _torch():
    try:
        import torch
        import torch_geometric
    except ImportError:
        return None
    return torch, torch_geometric


def gnn_status() -> dict:
    wanted = os.environ.get("THEMATICML_GNN_BACKEND", "").lower()
    common = {"kind": "neural network (optional)", "model": "GraphSAGE", "tasks": ["link_prediction", "node_classification"]}
    found = None if wanted == "numpy" else _torch()
    if found:
        torch, pyg = found
        cuda = bool(torch.cuda.is_available())
        return {**common, "available": True, "reason": None, "backend": "pytorch", "torch": torch.__version__,
                "torch_geometric": pyg.__version__, "cuda_available": cuda,
                "default_device": "cpu", "devices": ["cpu", "auto"] + (["cuda"] if cuda else [])}
    if wanted == "pytorch":
        return {**common, "available": False, "backend": "pytorch", "reason": "PyTorch Geometric is not installed. " + INSTALL_HINT}
    return {**common, "available": True, "reason": None, "backend": "numpy", "torch": None, "torch_geometric": None,
            "cuda_available": False, "default_device": "cpu", "devices": ["cpu"],
            "note": "Built-in NumPy GraphSAGE on the CPU. " + INSTALL_HINT + " to train with PyTorch instead."}


def require_gnn() -> None:
    status = gnn_status()
    if not status["available"]:
        raise OptionalDependencyError(status["reason"])
