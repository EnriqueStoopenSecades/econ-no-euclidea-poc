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
