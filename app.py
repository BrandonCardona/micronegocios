import streamlit as st
import pickle


pkl_filename = "models/pickle_modelsvm.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

st.write(modelo)

# Configuración de la interfaz
st.set_page_config(page_title="Interfaz de Métodos", layout="wide")

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
    
    # Campo para cargar un archivo CSV
    uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
    
    # Lista desplegable para elegir el tipo de método
    metodo_principal = st.selectbox(
        "¿Qué método desea usar?",
        ["Seleccione una opción", "Métodos supervisados", "Métodos no supervisados"],
        key="metodo_principal"
    )
    
    # Lista desplegable dependiente del método principal
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
    
    # Verificar si el botón debe estar habilitado
    boton_habilitado = metodo_principal != "Seleccione una opción" and metodo_secundario != ""

    # Botón calcular (habilitado/deshabilitado)
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
