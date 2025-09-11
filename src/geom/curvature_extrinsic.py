"""
Cálculo de curvatura extrínseca en una superficie embebida en R^3
usando ajuste cuadrático local en el parche tangente.

Pasos:
1) Obtener coords locales (u,v) en plano tangente + alturas h respecto a la normal.
2) Ajustar polinomio cuadrático h(u,v).
3) Extraer la Hessiana (segunda forma fundamental aproximada).
4) Calcular curvaturas principales k1,k2, curvatura gaussiana K, media H, escalar R=2K.
"""

import numpy as np

def fit_quadratic_patch(coords2: np.ndarray, heights: np.ndarray, weighted: bool = True):
    """
    Ajusta un polinomio cuadrático h(u,v) = 0.5*a*u^2 + b*u*v + 0.5*c*v^2 + alpha*u + beta*v + gamma
    sobre los vecinos proyectados en el plano tangente.

    Args:
        coords2 : (m,2) coords (u,v) de los vecinos en el plano tangente
        heights : (m,) alturas de los vecinos respecto a la normal
        weighted: usa pesos gaussianos para dar más importancia a los cercanos

    Returns:
        coeffs: [a, b, c, alpha, beta, gamma]
    """
    u = coords2[:,0]; v = coords2[:,1]; h = heights
    m = len(u)

    # diseño: columnas [u^2, u*v, v^2, u, v, 1]
    X = np.column_stack([0.5*u**2, u*v, 0.5*v**2, u, v, np.ones(m)])

    if weighted:
        r = np.sqrt(u**2 + v**2) + 1e-12
        h_band = np.median(r)
        w = np.exp(- (r**2) / (2*h_band**2))
        W = np.diag(w)
        coeffs, *_ = np.linalg.lstsq(W @ X, W @ h, rcond=None)
    else:
        coeffs, *_ = np.linalg.lstsq(X, h, rcond=None)

    return coeffs

def curvature_from_quadratic(coeffs: np.ndarray):
    """
    Extrae curvaturas principales y escalar a partir de los coef. cuadráticos.

    Args:
        coeffs: [a, b, c, alpha, beta, gamma]

    Returns:
        k1, k2: curvaturas principales
        K     : curvatura gaussiana
        H     : curvatura media
        R     : curvatura escalar (2K)
    """
    a, b, c, *_ = coeffs
    II = np.array([[a, b],
                   [b, c]])

    # Autovalores de II ≈ curvaturas principales (en base ortonormal tangente)
    evals, _ = np.linalg.eigh(II)
    k1, k2 = evals
    K = np.linalg.det(II)
    H = 0.5 * np.trace(II)
    R = 2*K

    return k1, k2, K, H, R
