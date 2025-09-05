import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path

def compute_geodesic_distances_fast(G: nx.Graph, method: str = "D") -> np.ndarray:
    """
    Distancias geodésicas usando scipy.sparse (Dijkstra por defecto).
    method: "D" (Dijkstra) o "J" (Johnson). Con pesos positivos, "D" suele ir muy bien.
    """
    A = nx.to_scipy_sparse_array(G, weight="weight", dtype=float, format="csr")
    D = shortest_path(A, directed=False, method=method)  # N x N, float64 (inf si no hay camino)
    return np.asarray(D, dtype=float)
