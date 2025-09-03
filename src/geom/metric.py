"""
Estimación de la métrica local g_ij a partir de la base tangente.
- En coords locales, g ≈ I si usas base ortonormal; con ruido/curvatura, puede desviarse.
- También provee proyección de vecinos a coords locales.
"""
from __future__ import annotations
import numpy as np

def local_coordinates(X: np.ndarray, center: np.ndarray, E: np.ndarray):
    """
    Proyecta puntos X al sistema de coordenadas local definido por (center, E).
    E: base ortonormal del plano tangente (D x d). Usamos coords = (X - center) @ E.
    Returns:
        coords: (n, d) coordenadas locales
    """
    Xc = X - center  # centrar en el punto
    coords = Xc @ E  # proyectar a la base tangente
    return coords
