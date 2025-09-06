# ECONOMÍA NO EUCLÍDEA — PoC (ISOMAP + Geometría Diferencial)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
![Status](https://img.shields.io/badge/status-PoC%20complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Jupyter](https://img.shields.io/badge/notebooks-Jupyter-orange)

**Objetivo.** Validar un pipeline geométrico para datos con estructura no lineal: grafo k-NN, distancias **geodésicas** sobre el grafo, **ISOMAP/MDS**, estimación de **planos tangentes** (LPCA), **métrica local** y **curvaturas** a partir de parches cuadráticos.  
Incluye benchmarks sintéticos en **esfera** y **toro** y una versión rápida de geodésicas (Dijkstra/CSR) que escala a miles de nodos.

---

## Tabla de contenidos
- [Arquitectura del pipeline](#arquitectura-del-pipeline)
- [Estructura del repo](#estructura-del-repo)
- [Instalación](#instalación)
- [Quickstart](#quickstart)
- [Experimentos incluidos](#experimentos-incluidos)
- [Resultados clave](#resultados-clave)
- [Benchmarks de tiempo](#benchmarks-de-tiempo)
- [Reproducibilidad](#reproducibilidad)
- [Roadmap](#roadmap)
- [Créditos](#créditos)

---

## Arquitectura del pipeline

1. **Datos en ℝ³** → muestras en superficie (esfera/toro) con ruido opcional.  
2. **Grafo k-NN** → no dirigido, pesos = distancia euclídea.  
3. **Geodésicas** → distancias más cortas sobre el grafo (Dijkstra en matriz dispersa CSR).  
4. **ISOMAP/MDS** → embedding 2D/3D desde la matriz de distancias.  
5. **LPCA** → SVD local para base tangente \(E=[e_1,e_2]\) y normal \(n\).  
6. **Coordenadas locales** \((u,v)\) en el plano tangente.  
7. **Métrica local** \(g\) (covarianza ponderada normalizada; \(\mathrm{tr}(g)=2\)).  
8. **Curvaturas** \(k_1,k_2, K, H, \mathcal{R}=2K\) via ajuste cuadrático \(h(u,v)\).

---

## Estructura del repo

├─ notebooks/
│ ├─ PoC.ipynb # pipeline completo en esfera
│ └─ 02_torus_demo.ipynb # pipeline en toro
├─ src/
│ ├─ data/
│ │ └─ synth.py # generadores: esfera, toro, ruido
│ ├─ embed/
│ │ └─ isomap.py # MDS clásico (Torgerson)
│ ├─ geom/
│ │ ├─ lpca.py # plano tangente (LPCA)
│ │ ├─ metric.py # coords locales + métrica g
│ │ └─ curvature_extrinsic.py # ajuste cuadrático y curvaturas
│ ├─ graph/
│ │ ├─ knn.py # grafo k-NN
│ │ └─ shortest_path_FAST.py # geodésicas rápidas (Dijkstra/CSR)
│ └─ viz/
│ └─ plots.py # visualizaciones 2D/3D
├─ artifacts/ # figuras generadas (ignorado)
├─ reports/ # notas y resúmenes
├─ requirements.txt
└─ README.md


---

## Instalación

```bash
# (opcional) conda
conda create -n econ-poc python=3.11 -y
conda activate econ-poc

# dependencias
pip install -r requirements.txt


import numpy as np
from src.data.synth import generate_sphere_points, add_gaussian_noise
from src.graph.knn import build_knn_graph
from src.graph.shortest_path_FAST import compute_geodesic_distances_fast
from src.embed.isomap import classical_mds
from src.geom.lpca import tangent_plane
from src.geom.metric import local_coordinates, estimate_metric
from src.geom.curvature_extrinsic import fit_quadratic_patch, curvature_from_quadratic

# 1) Datos
X  = generate_sphere_points(2000)            # superficie; versión de radio 2: generate_sphere_points2(...)
Xn = add_gaussian_noise(X, std=0.01, seed=42)

# 2) Grafo + geodésicas
G = build_knn_graph(Xn, k=20)
D = compute_geodesic_distances_fast(G, method="D")     # Dijkstra/CSR

# 3) ISOMAP/MDS
Y2 = classical_mds(D, n_components=2)
Y3 = classical_mds(D, n_components=3)

# 4) LPCA + curvaturas locales (en un punto)
center_idx = 42
neighbors_idx = np.array(list(G.neighbors(center_idx)))
center, E, normal, _ = tangent_plane(Xn, neighbors_idx, center_idx, d=2)
Nv = Xn[neighbors_idx]
coords2 = local_coordinates(Nv, center=center, E=E)
heights = (Nv - center) @ normal
coeffs = fit_quadratic_patch(coords2, heights, weighted=True)
k1, k2, K, H, R = curvature_from_quadratic(coeffs)
print("k1,k2,K,H,R:", k1, k2, K, H, R)


## Citacion

@misc{econ-no-euclidea-poc,
  title        = {Economía no euclídea — PoC (ISOMAP + Geometría Diferencial)},
  author       = {Enrique Stoopen Secades},
  year         = {2025},
  howpublished = {\url{https://github.com/EnriqueStoopenSecades/econ-no-euclidea-poc}}
}
