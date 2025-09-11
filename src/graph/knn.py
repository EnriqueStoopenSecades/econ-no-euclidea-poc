"""
Construcción de grafos k-NN para el PoC.
"""

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

def build_knn_graph(X: np.ndarray, k: int = 10) -> nx.Graph:
    """
    Construye un grafo k-NN no dirigido a partir de X.

    Args:
        X : np.ndarray con forma (n, d), puntos en R^d
        k : número de vecinos a conectar

    Returns:
        G : networkx.Graph con n nodos y aristas con peso = distancia euclídea
    """
    # TODO: aquí implementaremos la construcción
    
        # 1) Validaciones simples
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    n = X.shape[0]
    if n == 0:
        raise ValueError("X está vacío.")
    if k <= 0:
        raise ValueError("k debe ser > 0.")
    if k >= n:
        raise ValueError(f"k={k} no puede ser >= número de puntos n={n}.")

    # 2) Ajustar un buscador de k vecinos (sumamos 1 para incluir al propio punto)
    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n), #el NN de cada punto K es si mismo, por ello el k + 1
        metric="euclidean",
        algorithm="auto",
    ).fit(X)

    # 3) Para cada punto, obtener (distancias, índices) de sus k vecinos más cercanos
    distances, indices = nbrs.kneighbors(X)

        # 4) Crear grafo vacío y añadir N nodos (0..n-1)
    G = nx.Graph()
    G.add_nodes_from(range(n))


        # 5) Añadir aristas: para cada punto i, conectar con sus k vecinos distintos de sí mismo
    for i in range(n):
        # indices[i, 0] es i mismo (distancia 0). Lo saltamos con [1:]
        neigh_idxs = indices[i, 1:]
        neigh_dsts = distances[i, 1:]

        for j_idx, d in zip(neigh_idxs, neigh_dsts):
            j = int(j_idx)
            if i == j:
                continue

            w = float(d)  # peso = distancia euclídea
            # Evitar duplicar aristas: si ya existe (i,j), conservar el menor peso
            if G.has_edge(i, j):
                if w < G[i][j].get("weight", np.inf):
                    G[i][j]["weight"] = w
            else:
                G.add_edge(i, j, weight=w)

    return G

def graph_summary(G: nx.Graph) -> dict:
    comps = list(nx.connected_components(G))
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 0.0 if n == 0 else (2.0 * m) / n
    return {
        "nodes": n,
        "edges": m,
        "components": len(comps),
        "avg_degree": avg_deg,
        "is_connected": len(comps) == 1,
    }

def largest_component_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    giant = max(nx.connected_components(G), key=len)
    return G.subgraph(giant).copy()