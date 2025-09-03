"""
LPCA (Local PCA) para estimar bases de espacios tangentes.
- Dado un punto y sus vecinos, proyecta a un plano tangente.
- Devuelve: base ortonormal {e1, e2}, normal n, varianzas locales.
"""
from __future__ import annotations
import numpy as np

def tangent_plane(X: np.ndarray, neighbors_idx: np.ndarray, center_idx: int, d: int = 2):
    """
    Estima el plano tangente en X[center_idx] usando sus vecinos.
    Args:
        X: (n, D) datos en R^D
        neighbors_idx: índices de vecinos del punto center_idx (array 1D)
        center_idx: índice del punto central
        d: dimensión intrínseca local (2 para superficie)
    Returns:
        center: (D,) punto central
        E: (D, d) base ortonormal del tangente (columnas)
        normal: (D,) vector normal aproximado (si D=3 y d=2)
        svals: (d,) valores singulares (escala/curvatura local)
    """
    pass
