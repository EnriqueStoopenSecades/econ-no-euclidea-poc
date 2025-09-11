"""
Curvaturas locales: tensor de Riemann, Ricci y curvatura escalar.
- A partir de Γ y sus derivadas, construye R^i_{jkl}, contrae a Ricci y a escalar.
"""
from __future__ import annotations
import numpy as np

def riemann_tensor(Gamma: np.ndarray, dGamma: np.ndarray):
    """Construye R^i_{jkl} numéricamente."""
    pass

def ricci_tensor(Riemann: np.ndarray):
    """Contrae R^i_{jil} sobre índices apropiados para obtener Ricci R_jl."""
    pass

def scalar_curvature(Ricci: np.ndarray, g_inv: np.ndarray):
    """R = g^{ij} R_ij (curvatura escalar)."""
    pass
