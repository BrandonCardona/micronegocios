import streamlit as st
import pickle

# Ruta al archivo .pkl
pkl_filename = "models/varios_variables.pkl"  # Cambia esta ruta por la ubicación de tu archivo .pkl

# Función para cargar las variables del archivo pkl
def cargar_variables_pkl():
    with open(pkl_filename, 'rb') as file:
        variables = pickle.load(file)
    df_cargado = variables['dataframe']
    modelo_cargado = variables['modelo']
    lista_cargada = variables['lista']
    return df_cargado, modelo_cargado, lista_cargada

# Configuración de la interfaz
st.set_page_config(page_title="Interfaz de Métodos", layout="wide")

# Navbar simulado con columnas
st.markdown(
    """
    <style>
        .navbar {
            display: flex;
            justify-content: center;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
        }
        .navbar button {
            margin: 0 5px;
            font-size: 16px;
            font-weight: bold;
            color: #333;
            background-color: #f1f1f1;
            border: 1px solid #ccc;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        .navbar button:hover {
            background-color: #e0e0e0;
        }
        .navbar button.active {
            background-color: #007bff;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Estado inicial del navbar
if "navbar_selection" not in st.session_state:
    st.session_state["navbar_selection"] = "Preprocesamiento"

# Función para manejar la selección del navbar
def set_navbar(selection):
    st.session_state["navbar_selection"] = selection

# Renderizar el navbar
navbar_html = f"""
<div class="navbar">
    <button class="{'active' if st.session_state['navbar_selection'] == 'Preprocesamiento' else ''}" onclick="window.location.href='?selection=Preprocesamiento'">Preprocesamiento</button>
    <button class="{'active' if st.session_state['navbar_selection'] == 'Metodos' else ''}" onclick="window.location.href='?selection=Metodos'">Metodos</button>
    <button class"```
={'active' if st.session_state['navbar_selection'] == 'Predicciones' else ''}" onclick="window.location.href='?selection=Predicciones'">Predicciones</button>
</div>
"""
st.markdown(navbar_html, unsafe_allow_html=True)

# Actualizar la selección del navbar según los botones
if st.session_state["navbar_selection"] == "Preprocesamiento":
    st.title("Menú de Preprocesamiento")
    st.write("Aquí se mostrarán las opciones relacionadas con el preprocesamiento de datos.")

elif st.session_state["navbar_selection"] == "Metodos":
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
        uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
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
                
                # Leer el archivo .pkl con las variables preguardadas
                df_cargado, modelo_cargado, lista_cargada = cargar_variables_pkl()
                    
                # Mostrar las variables cargadas
                st.write("DataFrame cargado:")
                st.write(df_cargado)

                st.write("Modelo cargado:")
                st.write(modelo_cargado)

                st.write("Lista cargada:")
                st.write(lista_cargada)
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
