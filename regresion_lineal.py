import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('LAB01.csv')  # Ajusta la ruta según la ubicación del archivo en tu carpeta

# Exploración y limpieza de datos
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Convertir 'horsepower' a numérico (es una columna de tipo objeto en tu dataset)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# Eliminar filas con valores nulos
data.dropna(inplace=True)

# Seleccionar características y variable objetivo
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']].values
y = data['mpg'].values

# Agregar una columna de unos a X para el término de intercepción
X = np.hstack([np.ones((X.shape[0], 1)), X])

m = y.size  # Número de ejemplos de entrenamiento

def computeCost(X, y, theta):
    m = y.size
    predictions = np.dot(X, theta)
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(np.square(errors))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = []
    
    for i in range(num_iters):
        predictions = np.dot(X, theta)
        errors = predictions - y
        theta -= (alpha / m) * np.dot(X.T, errors)
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history

# Inicializar theta
theta = np.zeros(X.shape[1])

# Configuraciones para el descenso por el gradiente
iterations = 1500
alpha = 0.001  # Reducido para evitar overflow

# Ejecutar descenso por gradiente
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print(f'Theta encontrada por descenso gradiente: {theta}')

# Graficar la convergencia del costo
plt.figure()
plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel('Número de iteraciones')
plt.ylabel('Costo J')
plt.title('Convergencia del costo')

# Predecir valores (ajusta las características según tu dataset)
predict1 = np.dot([1, 4, 200, 100, 2500, 15, 1], theta)  # Ejemplo de predicción con valores ficticios
print(f'Para características dadas, se predice un valor de {predict1:.2f}')

plt.show()
