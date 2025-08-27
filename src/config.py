"""
Archivo de configuración global para el Proof of Concept (PoC).

Aquí definimos parámetros clave como número de puntos,
semilla aleatoria, ruido, valores de k en el grafo y dimensiones objetivo.
"""
"Recordar que estaremos simulando datos en la superficie de una esfera S^2 C R^3"

# Semilla para reproducibilidad, siempre quiero que los numeros aleatorios sean los mismos
SEED = 42  #Guide to the Galaxy Easter Egg LOL

# Número de puntos a generar en la esfera
N_POINTS = 800

# Nivel de ruido (desviación estándar en R^3)
NOISE_STD = 0.01 #Considerando errores naturales como medicion, instrumentacion...etc

# Valores de k para grafo k-NN
#Luego vemos con cual trabajamos
K_LIST = [6, 8, 10, 12, 15]

# Dimensiones objetivo para embeddings
TARGET_DIMS = [2, 3]
