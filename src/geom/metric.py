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


def estimate_metric(coords_local: np.ndarray, weighted: bool = True) -> np.ndarray:
    """
    Estima una métrica local g_ij en el parche (coords_local de dimensión d).
    Idea práctica: usar la (co)varianza local, opcionalmente ponderada por distancia al centro,
    como aproximación a la 2ª estructura fundamental (escala y anisotropías).

    Nota:
    - En una base tangente ortonormal ideal y vecindario simétrico, g ≈ I.
    - Con ruido/curvatura y vecindario finito, g puede desviarse de I.
    - Para hacer comparables las métricas entre puntos, normalizamos la escala:
      g_normalized = (d / trace(S)) * S, donde S es (co)varianza estimada.

    Args:
        coords_local : (m, d) coordenadas de vecinos en el plano tangente
        weighted     : usa pesos gaussianos por distancia radial en el parche

    Returns:
        g : (d, d) métrica simétrica positiva definida normalizada (trace = d)
    """
    m, d = coords_local.shape
    if m < d + 1:
        # muy pocos vecinos para estimar algo estable
        return np.eye(d)

    X = coords_local  # (m, d)

    if weighted:
        # kernel gaussiano radial sobre el parche (h = mediana de distancias)
        r = np.linalg.norm(X, axis=1) + 1e-12
        h = np.median(r)
        if h <= 0:
            w = np.ones(m)
        else:
            w = np.exp(-(r**2) / (2.0 * h**2))
        W = w / (w.sum() + 1e-12)
        # covarianza ponderada: S = sum_i W_i (x_i x_i^T)
        S = (X.T * W) @ X
    else:
        # covarianza no ponderada alrededor del 0 (el centro ya está en 0)
        S = (X.T @ X) / float(m)

    # Normalización de escala: que trace(g) = d, para comparar entre puntos
    tr = np.trace(S)
    if tr <= 1e-12:
        g = np.eye(d)
    else:
        g = (d / tr) * S

    # Simetrizar por estabilidad numérica
    g = 0.5 * (g + g.T)
    return g

