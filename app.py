import streamlit as st

# Configuración de la interfaz
st.set_page_config(page_title="Interfaz de Métodos", layout="wide")

# Inicializar el estado de 'calcular' si no existe
if "calcular" not in st.session_state:
    st.session_state["calcular"] = False

# Función para resetear el estado de 'calcular' al cambiar selecciones
def reset_calculo():
    st.session_state["calcular"] = False

# Contenedor izquierdo
with st.sidebar:
    st.title("Opciones")
    
    # Campo para cargar un archivo CSV
    uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
    
    # Lista desplegable para elegir el tipo de método
    metodo_principal = st.selectbox(
        "¿Qué método desea usar?",
        ["Seleccione una opción", "Métodos supervisados", "Métodos no supervisados"],
        key="metodo_principal",
        on_change=reset_calculo,  # Resetea el cálculo si cambia esta opción
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
        key="metodo_secundario",
        on_change=reset_calculo,  # Resetea el cálculo si cambia esta opción
    )
    
    # Verificar si ambas opciones han sido seleccionadas
    boton_habilitado = metodo_principal != "Seleccione una opción" and metodo_secundario != ""
    
    # Botón calcular (habilitado/deshabilitado)
    if st.button("Calcular", disabled=not boton_habilitado):
        st.session_state["calcular"] = True

# Contenedor derecho
st.title("Métricas")
if st.session_state["calcular"]:
    if metodo_principal == "Métodos supervisados":
        if metodo_secundario == "Random Forest":
            st.subheader("Métricas para Random Forest")
            st.write("- MSE (Error Cuadrático Medio)")
            st.write("- R2 Score")
        elif metodo_secundario in ["SVM (Super Vector Machines)", "Naive Bayes", "KNN"]:
            st.subheader(f"Métricas para {metodo_secundario}")
            st.write("- Accuracy")
            st.write("- Recall")
            st.write("- F1 Score")
            st.write("- Matriz de confusión")
    elif metodo_principal == "Métodos no supervisados":
        st.subheader(f"Métricas para {metodo_secundario}")
        st.write("- Pureza")
        st.write("- Silueta")
        st.write("- Accuracy")
else:
    st.write("Seleccione un método y haga clic en calcular para ver las métricas.")
