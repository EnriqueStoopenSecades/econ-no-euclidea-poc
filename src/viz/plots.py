"""Funciones de visualización: scatter 3D, embeddings 2D/3D, curvas vs. k."""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_knn_graph_3d(X: np.ndarray, G: nx.Graph, max_edges: int = 200, outpath: str | None = None, title: str = "Grafo k-NN (muestra)"):
    """
    Dibuja puntos 3D y una muestra de aristas del grafo k-NN para no saturar.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:,0], X[:,1], X[:,2], s=15, alpha=0.6)
    # muestra de aristas
    for (i, j) in list(G.edges())[:max_edges]:
        xi, xj = X[i], X[j]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], linewidth=0.5, alpha=0.5)

    for a in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        a([-1.2, 1.2])

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_geodesic_path_3d(X: np.ndarray, G: nx.Graph, src: int, dst: int, outpath: str | None = None, title: str = "Geodésica aprox (camino más corto)"):
    """
    Resalta el camino más corto entre src y dst usando pesos del grafo.
    """
    import networkx as nx

    path = nx.shortest_path(G, source=src, target=dst, weight="weight")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:,0], X[:,1], X[:,2], s=10, alpha=0.45)

    coords = X[path]
    ax.plot(coords[:,0], coords[:,1], coords[:,2], linewidth=2, label=f"path {src}->{dst}")
    ax.scatter([X[src,0]], [X[src,1]], [X[src,2]], s=50, label="origen")
    ax.scatter([X[dst,0]], [X[dst,1]], [X[dst,2]], s=50, label="destino")

    for a in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        a([-1.2, 1.2])

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_radius_hist(r_clean: np.ndarray, r_noisy: np.ndarray, outpath: str | None = None, title: str = "Distribución del radio: clean vs noisy"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.hist(r_clean, bins=20, alpha=0.6, label="clean", density=True)
    plt.hist(r_noisy, bins=20, alpha=0.6, label="noisy", density=True)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("‖x‖"); plt.ylabel("densidad")
    plt.title(title); plt.legend()
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
