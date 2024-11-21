import streamlit as st
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Ruta al archivo .pkl
pkl_filename = "models/pickle_modelsvm.pkl"

with open(pkl_filename, 'rb') as file:
    var_pkl = pickle.load(file)

feature_importances_sorted = var_pkl['feature_importances_sorted']
X_value_copy = var_pkl['X_value_copy']
X_reduced = var_pkl['X_reduced']
class_labels = var_pkl['class_labels']

svm_results = var_pkl['svm_results']
accuracy = svm_results['accuracy']
recall = svm_results['recall']
f1_score = svm_results['f1_score']
confusion_matrix = svm_results['confusion_matrix']

nb_results = var_pkl['nb_results']
accuracy_nb = nb_results['accuracy']
recall_nb = nb_results['recall']
f1_score_nb = nb_results['f1_score']
confusion_matrix_nb = nb_results['confusion_matrix']

knn_results = var_pkl['knn_results']
accuracy_knn = knn_results['accuracy']
recall_knn = knn_results['recall']
f1_score_knn = knn_results['f1_score']
confusion_matrix_knn = knn_results['confusion_matrix']

RF_results = var_pkl['RF_results']
r2_RF = RF_results['r2_RF']
mse_RF = RF_results['mse_RF']
y_pr_RF = RF_results['y_pr']
y_te_RF = RF_results['y_te']

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

    if st.sidebar.button("Variables más importantes"):
        st.subheader("VARIABLES MAS IMPORTANTES EMPLEANDO RANDOM FOREST")
        
        for_plot = pd.DataFrame({'x_axis': X_value_copy.columns, 'y_axis': feature_importances_sorted}).sort_values(by='y_axis', ascending=True)
        plt.figure(figsize=(10, 6)) 
        for_plot['y_axis'].plot.barh()

        plt.title("Importancia de las Variables")
        plt.xlabel("Importancia")
        plt.ylabel("Variables")
        st.pyplot(plt)
    
    if st.sidebar.button("Histogramas"):
        st.subheader("HISTOGRAMAS")
        num_columns = 2
        num_histograms = len(X_value_copy.columns)
        num_rows = (num_histograms + 1) // num_columns  
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, 6 * num_rows))
        axes = axes.flatten()
        for i, column in enumerate(X_value_copy.columns):
            X_value_copy[column].hist(ax=axes[i], bins=20)
            axes[i].set_title(column, fontsize=10)
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Frecuencia')
        for j in range(num_histograms, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(plt)

    if st.sidebar.button("Matriz de correlación"):
        st.subheader("MATRIZ DE CORRELACIÓN")
        plt.figure(figsize=(20, 20))  
        sns.heatmap(X_reduced.corr(), annot=True, cmap='Spectral', linewidths=0.1)
        st.pyplot(plt)

elif st.session_state["navbar_selection"] == "Métodos":

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

    if st.session_state["calcular"]:
        metodo_confirmado_principal = st.session_state["metodo_confirmado_principal"]
        metodo_confirmado_secundario = st.session_state["metodo_confirmado_secundario"]
        if metodo_confirmado_principal == "Métodos supervisados":
            if metodo_confirmado_secundario == "Random Forest":
                st.subheader("Métricas para Random Forest")
                st.write(f"**- MSE (Error Cuadrático Medio)**: {mse_RF:.8f}")
                st.write(f"**- R2 Score**: {r2_RF:.8f}")

                residuos = y_pr_RF - y_te_RF
                st.subheader("Distribución de los Residuos (Errores)")
                plt.figure(figsize=(8,6))
                sns.histplot(residuos, kde=True, color='green')
                plt.title('Distribución de los Residuos (Errores)')
                plt.xlabel('Residuo (Predicción - Valor Real)')
                plt.ylabel('Frecuencia')
                st.pyplot(plt)

            elif  metodo_confirmado_secundario == "SVM (Super Vector Machines)":
                st.title("Resultados del Modelo Super Vector Machines")
                st.write(f"**Accuracy**: {accuracy:.8f}", font_size=24)
                st.write(f"**Recall**: {recall:.8f}", font_size=24)
                st.write(f"**F1-score**: {f1_score:.8f}", font_size=24)

                st.write("### Matriz de Confusión")
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                            xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix - SVM")
                st.pyplot(plt)

            elif  metodo_confirmado_secundario == "Naive Bayes":
                st.title("Resultados del Modelo Naive Bayes")
                st.write(f"**Accuracy**: {accuracy_nb:.8f}")
                st.write(f"**Recall**: {recall_nb:.8f}")
                st.write(f"**F1-score**: {f1_score_nb:.8f}")

                st.write("### Matriz de Confusión")
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix_nb, annot=True, fmt="d", cmap="Blues",
                            xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix - NB")
                st.pyplot(plt)

            elif  metodo_confirmado_secundario == "KNN":
                st.title("Resultados del Modelo K-Nearest Neighbors")
                st.write(f"**Accuracy**: {accuracy_knn:.8f}")
                st.write(f"**Recall**: {recall_knn:.8f}")
                st.write(f"**F1-score**: {f1_score_knn:.8f}")

                st.write("### Matriz de Confusión")
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix_knn, annot=True, fmt="d", cmap="Blues",
                            xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix - KNN")
                st.pyplot(plt)  

        elif metodo_confirmado_principal == "Métodos no supervisados":
            if metodo_confirmado_secundario == "K-Means":
                st.subheader("Métricas para K-Means")
                st.write("- Pureza")
                st.write("- Silueta")
                st.write("- Accuracy")

            elif metodo_confirmado_secundario == "Clustering jerárquico":
                st.subheader("Métricas para Clustering jerárquico")
                st.write("- Pureza")
                st.write("- Silueta")
                st.write("- Accuracy")

            elif metodo_confirmado_secundario == "DB-Scan":
                st.subheader("Métricas para DB-Scan")
                st.write("- Pureza")
                st.write("- Silueta")
                st.write("- Accuracy")

            elif metodo_confirmado_secundario == "GMM (Gaussian Mixture Clustering)":
                st.subheader("Métricas para GMM (Gaussian Mixture Clustering)")
                st.write("- Pureza")
                st.write("- Silueta")
                st.write("- Accuracy")

    else:
        st.write("Seleccione un método y haga clic en calcular para ver las métricas.")

elif st.session_state["navbar_selection"] == "Predicciones":
    st.title("Menú de Predicciones")
    st.write("Aquí se mostrarán las opciones relacionadas con predicciones.")
