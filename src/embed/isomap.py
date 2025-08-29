

"""
Embedding con ISOMAP (MDS clásico sobre distancias geodésicas).
"""

import numpy as np
from sklearn.decomposition import PCA

def classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Aplica MDS clásico (a.k.a. Torgerson) sobre la matriz de distancias cuadrada D.
    
    Args:
        D : np.ndarray (n x n), distancias (simétrica, diagonal=0)
        n_components : dimensión objetivo (2 o 3)
    
    Returns:
        Y : np.ndarray (n x n_components), embedding
    """
    # 1) Paso: convertir distancias al "matriz de Gram" (centrada)
    n = D.shape[0]
    H = np.eye(n) - np.ones((n,n)) / n
    D2 = D**2
    B = -0.5 * H @ D2 @ H  # matriz de Gram centrada
    
    # 2) Descomposición espectral
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # 3) Ordenar de mayor a menor
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 4) Tomar los n_components más grandes
    L = np.diag(np.sqrt(np.maximum(eigvals[:n_components], 0)))
    V = eigvecs[:, :n_components]
    Y = V @ L
    return Y
