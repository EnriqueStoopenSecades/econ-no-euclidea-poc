# src/eval/curvature.py
import numpy as np
import networkx as nx

from src.geom.lpca import tangent_plane
from src.geom.metric import local_coordinates
from src.geom.curvature_extrinsic import fit_quadratic_patch, curvature_from_quadratic

def _relerr(x, x0):
    return (x - x0) / x0 * 100.0 if x0 != 0 else np.nan

def print_curvature_analysis(coeffs, k1, k2, K, H, R, radius=1.0):
    """
    Imprime un análisis legible del ajuste cuadrático y curvaturas.
    radius: radio esperado (1.0 para esfera unidad).
    """
    a, b, c, alpha, beta, gamma = coeffs

    # Orientación (solo para esfera): H<0 → normal exterior; H>0 → interior
    expected_sign = -1.0 if H < 0 else 1.0
    k_expected  = expected_sign * (1.0 / radius)  # ±1/R
    H_expected  = expected_sign * (1.0 / radius)  # ±1/R
    K_expected  = 1.0 / (radius**2)               # 1/R^2
    R_expected  = 2 * K_expected                  # 2K

    err_k1 = _relerr(k1, k_expected)
    err_k2 = _relerr(k2, k_expected)
    err_K  = _relerr(K,  K_expected)
    err_H  = _relerr(H,  H_expected)
    err_R  = _relerr(R,  R_expected)

    ratio = max(abs(k1), abs(k2)) / max(min(abs(k1), abs(k2)), 1e-12)
    diff  = abs(k1 - k2)
    mean_ac = 0.5 * (abs(a) + abs(c))
    b_rel = abs(b) / (mean_ac + 1e-12)
    lin_norm = float(np.hypot(alpha, beta))

    orient_txt = "normal exterior (espera k1,k2≈-1, H≈-1)" if expected_sign < 0 else \
                 "normal interior (espera k1,k2≈+1, H≈+1)"

    print("=== Análisis del ajuste cuadrático (parche local) ===")
    print(f"Coeficientes [a, b, c, α, β, γ]: [{a:.6f}, {b:.6f}, {c:.6f}, {alpha:.6f}, {beta:.6f}, {gamma:.6f}]")
    print(f"Orientación inferida por H: {orient_txt}\n")

    print(f"Curvaturas principales: k1={k1:.6f}  (err {err_k1:+.2f}%) | "
          f"k2={k2:.6f}  (err {err_k2:+.2f}%)")
    print(f"  • Desbalance |k1-k2| = {diff:.6f}   |   razón anisotropía (max/min) = {ratio:.3f}")
    print(f"Curvatura gaussiana  K = {K:.6f}   (vs {K_expected:.1f}, err {err_K:+.2f}%)")
    print(f"Curvatura media      H = {H:.6f}   (vs {H_expected:+.1f}, err {err_H:+.2f}%)")
    print(f"Curvatura escalar    R = {R:.6f}   (vs {R_expected:.1f}, err {err_R:+.2f}%)\n")

    print("Diagnóstico de términos del polinomio:")
    print(f"  • a≈{a:.4f}, c≈{c:.4f}; b≈{b:.4e} (cruzado pequeño deseable; relativo={(b_rel):.2e})")
    print(f"  • Pendientes: α={alpha:.4e}, β={beta:.4e}  (‖(α,β)‖={lin_norm:.4e})")
    print(f"  • Offset γ={gamma:.4e}\n")

def analyze_center(center_idx, X, G: nx.Graph, weighted=True, radius=1.0, echo=True):
    """
    Pipeline compacto para un punto:
      1) vecinos kNN desde G
      2) LPCA → (center, E, normal)
      3) coords locales + alturas
      4) ajuste cuadrático
      5) curvaturas (k1,k2,K,H,R)
    Devuelve (coeffs, k1, k2, K, H, R). Si echo=True, imprime análisis.
    """
    neighbors_idx = np.array(list(G.neighbors(center_idx)))
    center, E, normal, _ = tangent_plane(X, neighbors_idx, center_idx, d=2)

    Nv = X[neighbors_idx]                      # vecinos en 3D
    coords2 = local_coordinates(Nv, center=center, E=E)   # (m,2)
    heights = (Nv - center) @ normal                       # (m,)

    coeffs = fit_quadratic_patch(coords2, heights, weighted=weighted)
    k1, k2, K, H, R = curvature_from_quadratic(coeffs)

    if echo:
        print_curvature_analysis(coeffs, k1, k2, K, H, R, radius=radius)

    return coeffs, k1, k2, K, H, R
