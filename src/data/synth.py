"""
Generación de datos sintéticos para el PoC.
Aquí vamos a empezar con una sola función: generar puntos en la esfera.
"""

import numpy as np


def generate_sphere_points(n: int) -> np.ndarray:
    """
    Genera n puntos en la esfera unitaria en R^3 (sin ruido).
    """
    # Paso 1: muestreamos coordenadas aleatorias
    u = np.random.uniform(-1.0, 1.0, size=n)      # eje z
    phi = np.random.uniform(0.0, 2.0 * np.pi, n)  # ángulo

    # Paso 2: convertimos a (x,y,z)
    r_xy = np.sqrt(1 - u**2)   # radio en plano xy
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    z = u

    # Paso 3: juntamos todo
    X = np.stack([x, y, z], axis=1)
    return X

def add_gaussian_noise(X: np.ndarray, std: float = 0.01, seed: int = 42) -> np.ndarray:


    """
    Agrega ruido gaussiano N(0, std^2) a cada coordenada de X en R^3.
    El resultado ya no está exactamente en la esfera, simulando mediciones reales.
    """
    np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=std, size=X.shape)
    X_noisy = X + noise
    return X_noisy

import numpy as np

def generate_torus_points(n: int, R: float = 2.0, r: float = 0.6, uniform: bool = True, seed: int | None = None) -> np.ndarray:
    """
    Genera n puntos en la superficie de un toro T(R,r) en R^3.
    Si uniform=True, usa aceptación-rechazo para muestreo ~ uniforme en área.
    """
    rng = np.random.default_rng(seed)

    # 1) muestreo de u
    u = rng.uniform(0.0, 2*np.pi, size=n)

    # 2) muestreo de v
    if uniform:
        v = []
        M = R + r  # cota superior de (R + r cos v)
        while len(v) < n:
            vcand = rng.uniform(0.0, 2*np.pi, size=n)
            y = rng.uniform(0.0, 1.0, size=n)
            accept = (R + r*np.cos(vcand)) / M
            keep = vcand[y < accept]
            v.extend(keep.tolist())
        v = np.array(v[:n])
    else:
        v = rng.uniform(0.0, 2*np.pi, size=n)  # (no uniforme; útil para pruebas rápidas)

    # 3) parametrización
    x = (R + r*np.cos(v)) * np.cos(u)
    y = (R + r*np.cos(v)) * np.sin(u)
    z =  r * np.sin(v)
    return np.column_stack([x, y, z])

import numpy as np

def generate_sphere_points2(n: int) -> np.ndarray:
    """
    Genera n puntos en la esfera de radio 2 en R^3 (sin ruido),
    con muestreo uniforme sobre la superficie.
    """
    # 1) Muestreo en la esfera unitaria
    u = np.random.uniform(-1.0, 1.0, size=n)           # z
    phi = np.random.uniform(0.0, 2.0*np.pi, size=n)    # ángulo en xy

    r_xy = np.sqrt(1.0 - u**2)
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    z = u

    X_unit = np.stack([x, y, z], axis=1)               # (n,3)
    return 2.0 * X_unit                                # escalar a radio 2
