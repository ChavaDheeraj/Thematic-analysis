"""Graph neural networks (GraphSAGE) for the code co-occurrence graph.

Neural and optional: the classical pipeline never depends on this package. Training uses PyTorch
Geometric when it is installed (``pip install thematicml[gnn]``) and the built-in NumPy implementation
otherwise; :func:`backend` picks one. Neither is imported until an experiment runs.
"""
from thematicml.graph.gnn import OptionalDependencyError, gnn_status, require_gnn

# thresholds are defined here (not in core) so they can be checked without torch
MIN_EDGES = 20
MIN_NODES = 8
# above this share of linked pairs, held-out link prediction is not informative
MAX_DENSITY = 0.5
MIN_PER_CLASS = 3
MIN_LABELLED = 10



def backend(name: str | None = None):
    """The GraphSAGE implementation: ``"pytorch"`` (:mod:`.core`), ``"numpy"`` (:mod:`.numpy_backend`),
    or by default whichever :func:`gnn_status` reports."""
    name = name or gnn_status()["backend"]
    if name == "pytorch":
        require_gnn()
        from thematicml.gnn import core

        return core
    from thematicml.gnn import numpy_backend

    return numpy_backend


__all__ = ["MAX_DENSITY", "MIN_EDGES", "MIN_LABELLED", "MIN_NODES", "MIN_PER_CLASS", "OptionalDependencyError", "backend", "gnn_status", "require_gnn"]
