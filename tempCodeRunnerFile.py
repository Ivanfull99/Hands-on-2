import numpy as np #para realizar operaciones numericas y matematicas completas

# Clase para la Regresión Lineal Simple
class SimpleLinearRegression:
    def __init__(self):
        # Dataset hardcoded con los datos de (Advertising y Sales)
        self.X = np.array([23, 26, 30, 34, 43, 48, 52, 57, 58]) # Advertising (valriable independiente)
        self.Y = np.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]) # Sales (variable dependiente)
        
        #se iniciailizan los coeficientes beta_0 y beta_1
        self.beta_0 = 0
        self.beta_1 = 0

    # Método para calcular los coeficientes de la regresión beta_1 (pendiente) y beta_0 (intersección)
    def fit(self):
        #se obtiene el numero de elementos en x
        n = len(self.X)  

        # Calcular las sumas necesarias según las fórmulas
        sum_x = np.sum(self.X)             # Σx
        sum_y = np.sum(self.Y)             # Σy
        sum_xy = np.sum(self.X * self.Y)   # Σ(x*y)
        sum_x2 = np.sum(self.X ** 2)       # Σ(x^2)

        # Fórmula para calcular beta_1 (pendiente)
        self.beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        # Fórmula para calcular beta_0 (intersección)
        self.beta_0 = (sum_y - self.beta_1 * sum_x) / n

    # Método para predecir nuevos valores
    def predict(self, x_new):
        return self.beta_0 + self.beta_1 * x_new

    # Método para imprimir la ecuación de la regresión
    def equation(self):
        return f"ŷ = {self.beta_0:.2f} + {self.beta_1:.2f}x"

# Función principal que se ejecuta desde la terminal
def main():
    # Crear un objeto del modelo
    slr = SimpleLinearRegression()
    
    # Ajustar el modelo
    slr.fit()
    
    # Imprimir la ecuación de la regresión
    print(slr.equation())
    
    # Solicitar al usuario que ingrese un valor de Advertising para predecir
    try:
        x_new = float(input("Ingrese un valor de Advertising para predecir las ventas: "))
        
        # Predecir el valor de Sales
        prediccion = slr.predict(x_new)
        print(f"Predicción de Sales para Advertising = {x_new}: {prediccion:.2f}")
    
    except ValueError:
        print("Por favor, ingrese un número válido.")

# Ejecutar la función main al ejecutar el script
if __name__ == "__main__":
    main()
