import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class DataTransformer:
    def __init__(self):
        self.encoders = {
            "impactoCrecimientoEmpresa": OrdinalEncoder(categories=[["Ninguno", "Bajo", "Medio", "Alto"]]),
            "impactoUtilidadEmpresa": OrdinalEncoder(categories=[["Nada útil", "Poco útil", "Medio", "Alto", "Muy útil"]]),
            "impactoVentasEmpresa": OrdinalEncoder(categories=[["Nulo", "Bajo", "Medio", "Alto", "Muy alto"]]),
            "numEmpleados": OrdinalEncoder(categories=[["1 o 2 empleados", "3 o 4 empleados", "5 o 6 empleados", "7 o 8 empleados", "9 empleados o más"]]),
            "tiempoUsoPlatDigital": OrdinalEncoder(categories=[["Menos de 1 año", "1 a 2 años", "3 a 4 años", "5 a 6 años", "7 o más años"]]),
            "VentasMensualesPlatDigitales": OrdinalEncoder(categories=[[
                "Ninguna, no recibo pagos por medios digitales", 
                "Rara vez (menos del 20% de los ingresos por ventas son digitales)", 
                "Pocas (entre 20% y 40% del total de ventas son digitales)", 
                "Aproximadamente la mitad (entre el 41% y 60%)", 
                "La mayoría (entre el 61% y 80%)", 
                "Todas o casi todas (Más del 80% del total de ventas son digitales)"
            ]]),
            "PlataformaDigital": OrdinalEncoder(categories=[["No", "Si"]]),
        }
    
    def transform(self, registro):
        transformed_data = {}
        for key, encoder in self.encoders.items():
            if key in registro:
                value = [[registro[key]]]
                transformed_value = encoder.fit_transform(value)[0][0]
                transformed_data[key] = transformed_value
            else:
                raise ValueError(f"Falta el valor para la clave: {key}")
        
        return transformed_data
