import streamlit as st
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
import scipy.cluster.hierarchy as shc
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# Ruta al archivo .pkl
pkl_filename = "models/pickle_modelsvm.pkl"

with open(pkl_filename, 'rb') as file:
    var_pkl = pickle.load(file)

feature_importances_sorted = var_pkl['feature_importances_sorted']
X_value_copy = var_pkl['X_value_copy']
X_reduced = var_pkl['X_reduced']
centroids = var_pkl['centroids']
labels = var_pkl['labels']
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

distortions = var_pkl['distortions']
model_Kmeans = var_pkl['model_Kmeans']
X_train_reduced_Kmeans = var_pkl['X_train_reduced_Kmeans']

y_km = var_pkl['y_km']
n_clusters = var_pkl['n_clusters']
silhouette_vals = var_pkl['silhouette_vals']
kmeans_1 = var_pkl['kmeans_1']
cluster_df = var_pkl['cluster_df']
score_kemans_g = var_pkl['score_kemans_g']

Kmeans_results = var_pkl['Kmeans_results']
score_kemans_s = Kmeans_results['score_kemans_s']
score_kemans_c = Kmeans_results['score_kemans_c']
score_kemans_d = Kmeans_results['score_kemans_d']

modelHC = var_pkl['modelHC']
X_train_scaled_HC = var_pkl['X_train_scaled_HC']
HC_results = var_pkl['HC_results']
score_AGclustering_s = HC_results['score_AGclustering_s']
score_AGclustering_c = HC_results['score_AGclustering_c']
score_AGclustering_d = HC_results['score_AGclustering_d']

eps_dbScan = var_pkl['eps_dbScan']
cluster_df_dbScan = var_pkl['cluster_df_dbScan']


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
            metodos_disponibles = ["K-Means", "Clustering jerárquico", "DB-Scan"]
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
                st.title("Métricas para Random Forest")
                st.markdown(f"<h2 style='font-size: 24px;'>- MSE (Error Cuadrático Medio): {mse_RF:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>- R2 Score: {r2_RF:.8f}</h2>", unsafe_allow_html=True)

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
                st.markdown(f"<h2 style='font-size: 24px;'>Accuracy: {accuracy:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Recall: {recall:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>F1-score: {f1_score:.8f}</h2>", unsafe_allow_html=True)

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
                st.markdown(f"<h2 style='font-size: 24px;'>Accuracy: {accuracy_nb:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Recall: {recall_nb:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>F1-score: {f1_score_nb:.8f}</h2>", unsafe_allow_html=True)

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
                st.markdown(f"<h2 style='font-size: 24px;'>Accuracy: {accuracy_knn:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Recall: {recall_knn:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>F1-score: {f1_score_knn:.8f}</h2>", unsafe_allow_html=True)

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

                # PRIMER GRAFICO DE CODO
                plt.figure(figsize=(8, 6))
                plt.plot(range(1, 7), distortions, marker='o')
                plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
                plt.xlabel('Número de Clusters')
                plt.ylabel('Distorsión')
                plt.tight_layout()
                st.subheader("Método del Codo (Elbow Method) para KMeans")
                st.pyplot(plt)

                # SEGUNDO GRAFICO DE CODO
                visualizer = KElbowVisualizer(model_Kmeans, k=(1, 10), timings=True)
                visualizer.fit(X_train_reduced_Kmeans)
                st.pyplot(visualizer.fig)
                
                # TERCER GRAFICO DE SILUETA
                plt.figure(figsize=(10, 7))
                y_ax_lower, y_ax_upper = 0, 0
                yticks = []
                for i, c in enumerate(np.unique(y_km)):
                    c_silhouette_vals = silhouette_vals[y_km == c]
                    c_silhouette_vals.sort()
                    y_ax_upper += len(c_silhouette_vals)
                    color = cm.jet(float(i) / n_clusters)
                    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                            edgecolor='none', color=color)

                    yticks.append((y_ax_lower + y_ax_upper) / 2.)
                    y_ax_lower += len(c_silhouette_vals)

                silhouette_avg = np.mean(silhouette_vals)
                plt.axvline(silhouette_avg, color="red", linestyle="--")
                plt.yticks(yticks, np.unique(y_km) + 1)
                plt.ylabel('Cluster')
                plt.xlabel('Coeficiente de Silueta')
                plt.title('Gráfico de Silueta para KMeans')
                plt.tight_layout()
                st.subheader("Gráfico de Silueta para KMeans")
                st.pyplot(plt)

                # GRAFICO DE CENTROIDES
                fig, ax = plt.subplots(figsize=(10, 7))  # Crear una figura y un eje
                ax.scatter(
                    X_reduced['impactoCrecimientoEmpresa_encoded'], 
                    X_reduced['impactoVentasEmpresa'], 
                    c=labels, 
                    cmap='viridis', 
                    s=100, 
                    alpha=0.7, 
                    label='Puntos de datos'
                )
                ax.scatter(
                    centroids[:, 0], 
                    centroids[:, 1], 
                    s=300, 
                    marker='X', 
                    c='red', 
                    label='Centroides', 
                    edgecolor='black'
                )
                ax.legend(loc='upper right')
                ax.set_title('Gráfica de Puntos de Datos con Centroides')
                st.subheader("Gráfico de Dispersión de los centroides")
                st.pyplot(fig)

                plt.clf()

                # MAPA DE DISTANCIA ENTRE CLUSTERS
                visualizer = InterclusterDistance(kmeans_1)
                visualizer.fit(cluster_df)
                st.subheader("Mapa de Distancia entre Clusters (Intercluster Distance)")
                st.pyplot(visualizer.fig)

                st.markdown(f"<h2 style='font-size: 24px;'>Gap Statistic Score: {score_kemans_g:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Silhouette Score: {score_kemans_s:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Calinski Harabasz Score: {score_kemans_c:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Davies Bouldin Score: {score_kemans_d:.8f}</h2>", unsafe_allow_html=True)

            elif metodo_confirmado_secundario == "Clustering jerárquico":

                st.subheader("Métricas para Clustering jerárquico")
                visualizer = KElbowVisualizer(modelHC, k=(2, 30), timings=True)
                visualizer.fit(cluster_df)
                st.subheader("Método del Codo para Clustering Jerárquico (AgglomerativeClustering)")
                st.pyplot(visualizer.fig) 
                
                plt.figure(figsize=(10, 7))
                plt.title("Dendrograma")
                dend = shc.dendrogram(shc.linkage(X_train_scaled_HC, method='ward'))
                st.subheader("Dendrograma para Clustering Jerárquico (AgglomerativeClustering)")
                st.pyplot(plt)
                
                
                st.markdown(f"<h2 style='font-size: 24px;'>Silhouette Score: {score_AGclustering_s:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Calinski Harabasz Score: {score_AGclustering_c:.8f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size: 24px;'>Davies Bouldin Score: {score_AGclustering_d:.8f}</h2>", unsafe_allow_html=True)

            elif metodo_confirmado_secundario == "DB-Scan":
                st.subheader("Métricas para DB-Scan")

                nearest_neighbors = NearestNeighbors(n_neighbors=11)
                neighbors = nearest_neighbors.fit(cluster_df_dbScan)
                distances, indices = neighbors.kneighbors(cluster_df_dbScan)
                distances = np.sort(distances[:, 10], axis=0)  # Tomar las distancias del k-ésimo vecino más cercano

                # Usar KneeLocator para encontrar el "codo" en la curva
                i = np.arange(len(distances))
                knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

                # Crear la gráfica del codo
                fig = plt.figure(figsize=(5, 5))
                knee.plot_knee()
                plt.xlabel("Puntos")
                plt.ylabel("Distancia")

                # Mostrar el gráfico en Streamlit
                st.subheader("Método del Codo para DBSCAN")
                st.pyplot(fig)

                # Mostrar el valor de la distancia en el "codo"
                st.write(f"Distancia en el codo: {distances[knee.knee]}")

                st.write("- Pureza")
                st.write("- Silueta")
                st.write("- Accuracy")

    else:
        st.write("Seleccione un método y haga clic en calcular para ver las métricas.")

elif st.session_state["navbar_selection"] == "Predicciones":
    st.title("Menú de Predicciones")
    st.write("Aquí se mostrarán las opciones relacionadas con predicciones.")
