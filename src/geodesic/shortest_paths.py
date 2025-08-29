"""Cálculo de distancias geodésicas usando caminos más cortos (Dijkstra/Floyd–Warshall)."""



from __future__ import annotations
import numpy as np
import networkx as nx


def compute_geodesic_distances(G: nx.Graph) -> np.ndarray:
    """
    Devuelve la matriz de distancias geodésicas (n x n) a partir del grafo ponderado G.
    Usa Floyd–Warshall (NetworkX) con peso 'weight'.

    Si el grafo no es conectado, las parejas sin camino quedan en +inf.
    """
    # Floyd–Warshall en NetworkX devuelve un np.ndarray (float64)
    D = nx.floyd_warshall_numpy(G, weight="weight")  # (n, n)
    return np.asarray(D, dtype=float)


#Podemos usar un helper para solucionar posibles desconexiones
def handle_disconnections(D: np.ndarray, strategy: str = "big_value") -> np.ndarray:
    """
    Maneja entradas infinitas (pares sin camino) en la matriz de distancias.
    Estrategias:
      - "big_value": reemplaza +inf por (max_finite * 10)
      - "error": lanza ValueError si hay inf
    """
    if not np.isfinite(D).all():
        if strategy == "error":
            raise ValueError("La matriz de geodésicas contiene +inf (grafo desconectado).")
        # big_value
        finite = D[np.isfinite(D)]
        if finite.size == 0:
            raise ValueError("No hay distancias finitas. El grafo está completamente desconectado.")
        big = float(finite.max() * 10.0)
        D = D.copy()
        D[~np.isfinite(D)] = big
        return D
    return D


def is_symmetric(D: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(D, D.T, atol=tol, rtol=0)

def has_zero_diagonal(D: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(np.diag(D), 0.0, atol=tol, rtol=0)

