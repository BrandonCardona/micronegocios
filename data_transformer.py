import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

class DataTransformer:
    def __init__(self):
        # Cargar el modelo desde el archivo pickle
        with open("models/pickle_modelsvm.pkl", 'rb') as file:
            var_pkl = pickle.load(file)
        
        # Obtener el df_reducido del pickle
        self.df_reducido = var_pkl['df_reducido']

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

        # Ajustar los codificadores con los datos de ejemplo para que puedan hacer la transformación
        for key, encoder in self.encoders.items():
            encoder.fit([[category] for category in encoder.categories[0]])

        # Inicializar el MinMaxScaler
        self.scaler = MinMaxScaler()

    def transform_and_scale(self, registro):
        """
        Este método recibe un registro del usuario con datos categóricos,
        lo transforma a numérico, lo agrega a df_reducido, y aplica escalado.
        """
        # Transformar los datos del usuario
        transformed_data = {}
        for key, encoder in self.encoders.items():
            if key in registro:
                value = [[registro[key]]]
                transformed_value = encoder.transform(value)[0][0]
                transformed_data[key] = transformed_value
            else:
                raise ValueError(f"Falta el valor para la clave: {key}")

        # Convertir los datos transformados a un DataFrame
        df_transformed = pd.DataFrame([transformed_data])

        # Asegurarse de que las columnas del df_reducido y el registro transformado coincidan
        # Renombrar las columnas para que coincidan
        df_transformed = df_transformed.rename(columns={
            "impactoCrecimientoEmpresa": "impactoCrecimientoEmpresa_encoded",
            "impactoUtilidadEmpresa": "impactoUtilidadEmpresa_encoded",
            "PlataformaDigital": "PlatDig_Nequi"
        })

        # Concatenar el registro transformado con el df_reducido para mantener las mismas columnas
        df_full = pd.concat([self.df_reducido, df_transformed], ignore_index=True)

        # Aplicar MinMaxScaler solo a los datos
        scaled_array = self.scaler.fit_transform(df_full)

        # Convertir el array escalado de vuelta a DataFrame
        scaled_df = pd.DataFrame(scaled_array, columns=df_full.columns)

        # Separar el registro del usuario escalado (última fila del DataFrame)
        registro_usuario_escalado = scaled_df.iloc[-1:]

        # Retornar el DataFrame escalado completo y el registro del usuario escalado
        return scaled_df, registro_usuario_escalado
