"""Optional graph neural network module.

GNNs are neural networks and are *not* part of the classical pipeline. This
module reports availability; training lives in :mod:`thematicml.gnn` and
requires ``pip install thematicml[gnn]`` (PyTorch + PyTorch Geometric). GNN
results are always reported next to classical link-prediction heuristics
scored on the same held-out edges.
"""
from __future__ import annotations


class OptionalDependencyError(ImportError):
    pass


def gnn_status() -> dict:
    try:
        import torch
        import torch_geometric
    except ImportError:
        return {
            "available": False,
            "kind": "neural network (optional)",
            "reason": "PyTorch Geometric is not installed. Install with: pip install thematicml[gnn] "
                      "(CPU wheels: pip install torch --index-url https://download.pytorch.org/whl/cpu, then pip install torch_geometric)",
        }
    cuda = bool(torch.cuda.is_available())
    return {"available": True, "kind": "neural network (optional)", "reason": None, "model": "GraphSAGE",
            "tasks": ["link_prediction", "node_classification"], "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__, "cuda_available": cuda,
            "default_device": "cpu", "devices": ["cpu", "auto"] + (["cuda"] if cuda else [])}


def require_gnn() -> None:
    status = gnn_status()
    if not status["available"]:
        raise OptionalDependencyError(status["reason"])
