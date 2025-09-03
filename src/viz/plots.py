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


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # por si lo necesitas en otros plots

def plot_tangent_frame_3d(X: np.ndarray, center_idx: int, E: np.ndarray, normal: np.ndarray,
                          scale: float = 0.25, outpath: str | None = None,
                          title: str = "Plano tangente y normal (LPCA)"):
    """
    Dibuja los puntos 3D, resalta el punto central y traza:
      - Dos vectores tangentes (columnas de E)
      - El vector normal
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")

    # nube
    ax.scatter(X[:,0], X[:,1], X[:,2], s=10, alpha=0.45)

    # punto central
    c = X[center_idx]
    ax.scatter([c[0]], [c[1]], [c[2]], s=60)

    # vectores tangentes (en rojo y naranja)
    t1 = E[:,0]; ax.plot([c[0], c[0] + scale*t1[0]], [c[1], c[1] + scale*t1[1]], [c[2], c[2] + scale*t1[2]])
    if E.shape[1] > 1:
        t2 = E[:,1]; ax.plot([c[0], c[0] + scale*t2[0]], [c[1], c[1] + scale*t2[1]], [c[2], c[2] + scale*t2[2]])

    # normal (en una tercera dirección)
    if normal is not None:
        n = normal
        ax.plot([c[0], c[0] + scale*n[0]], [c[1], c[1] + scale*n[1]], [c[2], c[2] + scale*n[2]])

    for a in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        a([-1.3, 1.3])

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def plot_local_patch_with_ellipse(coords2: np.ndarray, g: np.ndarray,
                                  outpath: str | None = None,
                                  title: str = "Parche local + elipse (métrica)"):

    # autovalores/vectores de g (métrica normalizada)
    evals, evecs = np.linalg.eigh(g)
    # ordenar desc (para etiquetar mayor/menor)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]

    # construimos el contorno de elipse a 1 desviación (escala visual)
    theta = np.linspace(0, 2*np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (200,2)

    # “raíz” de la métrica (o inversa si prefieres ver iso-contornos); aquí usamos evecs * sqrt(evals)
    A = evecs @ np.diag(np.sqrt(np.maximum(evals, 1e-12)))
    ell = circle @ A.T

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(coords2[:,0], coords2[:,1], s=20, alpha=0.8, label="vecinos")
    plt.plot(ell[:,0], ell[:,1], linewidth=2, label="elipse métrica (trace=2)")

    plt.axhline(0, linewidth=1); plt.axvline(0, linewidth=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("e1"); plt.ylabel("e2")
    plt.legend()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
