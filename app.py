import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  # Importa el módulo completo de matplotlib para cambiar el backend

# Configurar el backend de matplotlib
matplotlib.use("Agg")

# Título de la app
st.title("Aplicación de prueba en Streamlit")

# Texto de bienvenida
st.write("Bienvenido a la aplicación de prueba desplegada en Streamlit.")

# Mostrar un número aleatorio
random_number = np.random.randint(1, 100)
st.write(f"Número aleatorio generado: {random_number}")

# Generar y mostrar un gráfico básico
data = pd.DataFrame({
    'x': range(10),
    'y': np.random.rand(10)
})

st.write("Aquí tienes un gráfico de muestra:")

# Crear una figura de matplotlib
fig, ax = plt.subplots()
ax.plot(data['x'], data['y'])

# Mostrar la figura en Streamlit
st.pyplot(fig)
