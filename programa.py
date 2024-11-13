import numpy as np #para realizar operaciones numericas y matematicas complejas

# Clase para la Regresión Lineal Simple
class SimpleLinearRegression:
    def __init__(self):
        # Dataset hardcoded con los datos de (Advertising y Sales)
        self.X = np.array([1,2,3,4,5,6,7,8,9]) # Advertising (valriable independiente)
        self.Y = np.array([2,4,6,8,10,12,14,16,18]) # Sales (variable dependiente)
        
        #se iniciailizan los coeficientes beta_0(intersección) y beta_1(pendiente) en cero
        self.beta_0 = 0
        self.beta_1 = 0

    # Método para calcular los coeficientes beta_1 (pendiente) y beta_0 (intersección)
    def fit(self):
        #se obtiene el numero de elementos en x(numero de datos)
        n = len(self.X)  

        # Se calcula las sumas necesarias según las fórmulas
        sum_x = np.sum(self.X)             # Σx: suma de los valores de X
        sum_y = np.sum(self.Y)             # Σy: suma de los valores de Y
        sum_xy = np.sum(self.X * self.Y)   # Σ(x*y): suma del producto de cada par de valores (x, y)
        sum_x2 = np.sum(self.X ** 2)       # Σ(x^2): suma de los cuadrados de los valores de X

        # Fórmula para calcular beta_1 (pendiente)
        self.beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        # Fórmula para calcular beta_0 (intersección)
        self.beta_0 = (sum_y - self.beta_1 * sum_x) / n

    # Método para predecir nuevos valores de Y dado un nuevo valor de X
    def predict(self, x_new):
        # Utiliza la ecuación de la recta (ŷ = beta_0 + beta_1 * x_new) para calcular la predicción
        return self.beta_0 + self.beta_1 * x_new

    # Método para imprimir la ecuación de la regresión
    def equation(self):
        # Devuelve la ecuación de la recta de regresión
        return f"ŷ = {self.beta_0:.2f} + {self.beta_1:.2f}x"

def main():
    # Crear un objeto del modelo
    slr = SimpleLinearRegression()
    
    # Ajusta el modelo a los datos de X e Y usando el método fit
    slr.fit()
    
    # Imprimir la ecuación de la regresión
    print(slr.equation())
    
    # Solicitar al usuario que ingrese un valor de Advertising para predecir
    try:
        x_new = float(input("Ingrese un valor de Advertising para predecir las ventas: "))
        
        # Calcula la predicción de Sales usando el valor ingresado de Advertising
        prediccion = slr.predict(x_new)
        print(f"Predicción de Sales para Advertising = {x_new}: {prediccion:.2f}")
    
    except ValueError:
        # Maneja errores en caso de que el usuario no ingrese un número válido
        print("Por favor, ingrese un número válido.")

# Ejecutar la función main al ejecutar el script
if __name__ == "__main__":
    main()
