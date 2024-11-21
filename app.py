import streamlit as st
import pickle
import pandas as pd
from matplotlib import pyplot as plt

# Ruta al archivo .pkl
pkl_filename = "models/pickle_modelsvm.pkl"

with open(pkl_filename, 'rb') as file:
    var_pkl = pickle.load(file)

def asignar_variables_pkl():
    feature_importances_sorted = var_pkl['feature_importances_sorted']
    X_value_copy = var_pkl['X_value_copy']
    X_reduced = var_pkl['X_reduced']
    return feature_importances_sorted, X_value_copy, X_reduced

# Configuración de la interfaz
st.set_page_config(page_title="Interfaz de Métodos", layout="wide")

# Estado inicial del navbar
if "navbar_selection" not in st.session_state:
    st.session_state["navbar_selection"] = "Preprocesamiento"

# Función para cambiar la selección del navbar
def set_navbar(selection):
    st.session_state["navbar_selection"] = selection

# Navbar
st.sidebar.title("Navegación")
if st.sidebar.button("Preprocesamiento"):
    set_navbar("Preprocesamiento")
if st.sidebar.button("Métodos"):
    set_navbar("Métodos")
if st.sidebar.button("Predicciones"):
    set_navbar("Predicciones")

# Menú desplegable de la izquierda para "Preprocesamiento"
if st.session_state["navbar_selection"] == "Preprocesamiento":
    st.sidebar.title("Opciones de Preprocesamiento")

    # Agregar los dos botones en el menú desplegable de la izquierda (barra lateral)
    if st.sidebar.button("Variables más importantes"):
        st.write("Este botón mostrará las variables más importantes del modelo.")

        feature_importances_sorted, X_value_copy, X_reduced = asignar_variables_pkl()

        st.write("feature_importances_sorted cargado:")
        st.write(feature_importances_sorted)

        st.write("X_value cargado:")
        st.write(X_value_copy)

        st.write("X_reduced cargada:")
        st.write(X_reduced)

        st.subheader("Gráfico de Importancia de Variables")
        
        for_plot = pd.DataFrame({'x_axis': X_value_copy.columns, 'y_axis': feature_importances_sorted}).sort_values(by='y_axis', ascending=True)
        plt.figure(figsize=(10, 6)) 
        for_plot['y_axis'].plot.barh()

        plt.title("Importancia de las Variables")
        plt.xlabel("Importancia")
        plt.ylabel("Variables")

        st.pyplot(plt)

        
    if st.sidebar.button("Matriz de correlación"):
        st.write("Este botón mostrará la matriz de correlación de las variables.")

elif st.session_state["navbar_selection"] == "Métodos":
    st.title("Menú de Métodos")
    # Inicializar estados en session_state
    if "metodo_confirmado_principal" not in st.session_state:
        st.session_state["metodo_confirmado_principal"] = None
    if "metodo_confirmado_secundario" not in st.session_state:
        st.session_state["metodo_confirmado_secundario"] = None
    if "calcular" not in st.session_state:
        st.session_state["calcular"] = False

    # Función para manejar el botón calcular
    def calcular():
        st.session_state["metodo_confirmado_principal"] = st.session_state["metodo_principal"]
        st.session_state["metodo_confirmado_secundario"] = st.session_state["metodo_secundario"]
        st.session_state["calcular"] = True

    # Contenedor izquierdo
    with st.sidebar:
        st.title("Opciones")
        
        metodo_principal = st.selectbox(
            "¿Qué método desea usar?",
            ["Seleccione una opción", "Métodos supervisados", "Métodos no supervisados"],
            key="metodo_principal"
        )
        metodos_disponibles = []
        if metodo_principal == "Métodos supervisados":
            metodos_disponibles = ["SVM (Super Vector Machines)", "Naive Bayes", "KNN", "Random Forest"]
        elif metodo_principal == "Métodos no supervisados":
            metodos_disponibles = ["K-Means", "Clustering jerárquico", "DB-Scan", "GMM (Gaussian Mixture Clustering)"]
        metodo_secundario = st.selectbox(
            "Métodos disponibles",
            metodos_disponibles if metodo_principal != "Seleccione una opción" else [], 
            key="metodo_secundario"
        )
        boton_habilitado = metodo_principal != "Seleccione una opción" and metodo_secundario != ""
        st.button("Calcular", on_click=calcular, disabled=not boton_habilitado)

    # Contenedor derecho
    st.title("Métricas")
    if st.session_state["calcular"]:
        metodo_confirmado_principal = st.session_state["metodo_confirmado_principal"]
        metodo_confirmado_secundario = st.session_state["metodo_confirmado_secundario"]
        if metodo_confirmado_principal == "Métodos supervisados":
            if metodo_confirmado_secundario == "Random Forest":
                st.subheader("Métricas para Random Forest")
                st.write("- MSE (Error Cuadrático Medio)")
                st.write("- R2 Score")

            elif metodo_confirmado_secundario in ["SVM (Super Vector Machines)", "Naive Bayes", "KNN"]:
                st.subheader(f"Métricas para {metodo_confirmado_secundario}")
                st.write("- Accuracy")
                st.write("- Recall")
                st.write("- F1 Score")
                st.write("- Matriz de confusión")
        elif metodo_confirmado_principal == "Métodos no supervisados":
            st.subheader(f"Métricas para {metodo_confirmado_secundario}")
            st.write("- Pureza")
            st.write("- Silueta")
            st.write("- Accuracy")
    else:
        st.write("Seleccione un método y haga clic en calcular para ver las métricas.")

elif st.session_state["navbar_selection"] == "Predicciones":
    st.title("Menú de Predicciones")
    st.write("Aquí se mostrarán las opciones relacionadas con predicciones.")
