"""
Estimación de la métrica local g_ij a partir de la base tangente.
- En coords locales, g ≈ I si usas base ortonormal; con ruido/curvatura, puede desviarse.
- También provee proyección de vecinos a coords locales.
"""
from __future__ import annotations
import numpy as np

def local_coordinates(X: np.ndarray, center: np.ndarray, E: np.ndarray):
    """Proyecta X al sistema de coordenadas local definido por (center, E)."""
    pass

def estimate_metric(coords_local: np.ndarray):
    """
    Estima g_ij con base en vecinos en coords locales (p.ej., covarianza).
    Returns:
        g: (d, d) métrica simétrica positiva aproximada
    """
    pass
