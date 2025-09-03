"""
LPCA (Local PCA) para estimar bases de espacios tangentes.
- Dado un punto y sus vecinos, proyecta a un plano tangente.
- Devuelve: base ortonormal {e1, e2}, normal n, varianzas locales.
"""
from __future__ import annotations
import numpy as np

def tangent_plane(X: np.ndarray, neighbors_idx: np.ndarray, center_idx: int, d: int = 2):
    """
    Estima el plano tangente en X[center_idx] usando sus vecinos (LPCA con SVD).

    Args:
        X            : (n, D) datos en R^D (en nuestro caso D=3)
        neighbors_idx: índices de vecinos del punto center_idx (array 1D)
        center_idx   : índice del punto central
        d            : dimensión intrínseca local (2 para superficie)

    Returns:
        center : (D,) punto central
        E      : (D, d) base ortonormal del tangente (columnas)
        normal : (D,) vector normal aproximado (columna ortogonal restante si D=3)
        svals  : (d,) valores singulares principales (escala local)
    """
    center = X[center_idx]              # punto central
    P = X[neighbors_idx] - center       # vecinos centrados (m, D)

    # SVD sobre vecinos centrados: P = U Σ V^T
    # Las columnas de V (en orden por Σ descendente) son direcciones en R^D.
    # Para una superficie 2D, las 2 primeras columnas de V ≈ base tangente;
    # la última (menor singular) ≈ normal.
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    V = Vt.T

    E = V[:, :d]                         # base tangente (D x d)
    normal = V[:, d] if V.shape[1] > d else None  # normal (D,)
    svals = S[:d].copy()

    # Aseguramos orientación determinística del normal (opcional)
    if normal is not None and normal[2] < 0:
        normal = -normal

    return center, E, normal, svals
