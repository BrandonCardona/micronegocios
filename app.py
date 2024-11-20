import streamlit as st

# Configuración de la interfaz
st.set_page_config(page_title="Interfaz de Métodos", layout="wide")

# Contenedor izquierdo
with st.sidebar:
    st.title("Opciones")
    
    # Campo para cargar un archivo CSV
    uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
    
    # Lista desplegable para elegir el tipo de método
    metodo_principal = st.selectbox(
        "¿Qué método desea usar?",
        ["Seleccione una opción", "Métodos supervisados", "Métodos no supervisados"]
    )
    
    # Lista desplegable dependiente del método principal
    metodos_disponibles = []
    if metodo_principal == "Métodos supervisados":
        metodos_disponibles = ["Seleccione una opción", "SVM (Super Vector Machines)", "Naive Bayes", "KNN", "Random Forest"]
    elif metodo_principal == "Métodos no supervisados":
        metodos_disponibles = ["Seleccione una opción", "K-Means", "Clustering jerárquico", "DB-Scan", "GMM (Gaussian Mixture Clustering)"]
    
    metodo_secundario = st.selectbox("Métodos disponibles", metodos_disponibles)
    
    # Botón calcular
    if st.button("Calcular"):
        st.session_state["calcular"] = True

# Contenedor derecho
st.title("Métricas")
if "calcular" in st.session_state and st.session_state["calcular"]:
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
