"""
Cálculo numérico de símbolos de Christoffel Γ^k_{ij} a partir de g_ij y sus derivadas.
- Aproximamos derivadas ∂_i g_jl con diferencias finitas sobre vecinos en coords locales.
"""
from __future__ import annotations
import numpy as np

def christoffel_symbols(g: np.ndarray, dg: np.ndarray):
    """
    Args:
        g: (d, d) métrica
        dg: (d, d, d) derivadas dg_ij / dx^k en coords locales
    Returns:
        Gamma: (d, d, d) símbolos Γ^k_{ij}
    """
    pass
