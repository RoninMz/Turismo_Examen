import pandas as pd  # Librería para la manipulación de datos
import numpy as np  # Librería para cálculos numéricos
from sklearn.cluster import KMeans  # Algoritmo KMeans para clustering
from sklearn.preprocessing import StandardScaler  # Para normalizar los datos
from scipy.spatial.distance import cdist  # Para calcular la distancia entre puntos y centros de clúster
from scipy.stats import f_oneway  # Para realizar la prueba ANOVA
import seaborn as sns  # Para visualización de datos
import matplotlib.pyplot as plt  # Para generar gráficos

# Cargar el archivo de Excel que contiene los datos
file_path = r'C:\Universidad\8ciclo\Inteligencia Artificial\codigo\Tarea\Turismo.xlsx'
data = pd.read_excel(file_path)  # Lee los datos desde el archivo de Excel

# Configuración para que pandas muestre todas las columnas sin truncar (ver todo sin cortarse)
pd.set_option('display.max_columns', None)

# Seleccionar las variables que contienen la palabra 'Importancia' (filtrar solo las columnas necesarias para el análisis)
variables_importancia = data.filter(like='Importancia')

# Normalizar los datos para que todas las variables tengan la misma escala
scaler = StandardScaler()  # Inicializar el normalizador
data_scaled = scaler.fit_transform(variables_importancia)  # Ajustar y transformar los datos

# Aplicar el algoritmo K-Means con 3 clústeres (agrupar en 3 grupos)
n_clusters = 3  # Definir el número de clústeres
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)  # Inicializar KMeans
kmeans.fit(data_scaled)  # Ajustar el modelo KMeans a los datos normalizados

# ---- Parte 1: Historial de Iteraciones ----
# Inicializar una lista para almacenar los cambios en los centros de clúster en cada iteración
centros_por_iteracion = []

# Hacer el fit de KMeans y capturar los centros en cada iteración
for i in range(1, kmeans.max_iter + 1):  # Recorrer el número máximo de iteraciones
    kmeans_iter = KMeans(n_clusters=n_clusters, init=kmeans.cluster_centers_ if i > 1 else 'k-means++', max_iter=i, n_init=1, random_state=42)
    # En la primera iteración, usa 'k-means++', después, usa los centros calculados previamente
    kmeans_iter.fit(data_scaled)  # Ajustar el modelo

    # Guardar los centros de los clústeres en cada iteración
    centros_por_iteracion.append(kmeans_iter.cluster_centers_.copy())
    
    # Verificar si el algoritmo ha convergido antes de alcanzar el número máximo de iteraciones
    if kmeans_iter.n_iter_ < i:  # Si converge antes de alcanzar el máximo, se detiene el ciclo
        break

# Convertir los centros de cada iteración en un DataFrame para su análisis
centros_iteracion_df = pd.DataFrame(
    np.vstack(centros_por_iteracion),  # Convertir la lista de centros en un arreglo 2D
    columns=[f'Variable {i+1}' for i in range(variables_importancia.shape[1])]  # Asignar nombres de columna como 'Variable 1', 'Variable 2', etc.
)

# Crear un índice para marcar las iteraciones y centros
index = []
for iter_num in range(len(centros_por_iteracion)):  # Recorrer cada iteración
    for centro in range(n_clusters):  # Recorrer cada centro (cluster)
        index.append(f"Iter {iter_num+1} Centro {centro+1}")  # Crear un índice con el número de iteración y centro

# Asignar el índice creado a la tabla
centros_iteracion_df.index = index

# Mostrar la tabla con el historial de iteraciones de los centros de clústeres
print("Historial de iteraciones de los centros de clústeres:")
print(centros_iteracion_df)

# ---- Parte 2: Tabla de Pertenencia a Clúster ----
# Asignar el número de clúster a cada fila del DataFrame original
data['Clúster'] = kmeans.labels_  # Asignar a cada fila el clúster correspondiente (0, 1, 2)

# Calcular la distancia de cada punto al centro de su clúster asignado
distancias = cdist(data_scaled, kmeans.cluster_centers_, 'euclidean')  # Calcular la distancia euclidiana de cada punto a los centros

# Crear una columna que contiene la distancia mínima al centro del clúster al que está asignado
data['Distancia'] = [distancias[i, cluster] for i, cluster in enumerate(kmeans.labels_)]  # Asignar la distancia mínima de cada punto a su centro

# Agregar una columna de número de caso (simplemente el índice del DataFrame más 1 para empezar en 1)
data['Número del caso'] = data.index + 1  # Numerar cada caso

# Seleccionar las columnas que queremos mostrar en la tabla de pertenencia a clúster
tabla_clusper = data[['Número del caso', 'Clúster', 'Distancia']]  # Seleccionar columnas para la tabla

# Mostrar la tabla de pertenencia a clústeres (primeras 15 filas)
print("\nTabla de pertenencia a clústeres:")
print(tabla_clusper.head(15))  # Mostrar las primeras 15 filas

# Visualizar la tabla de pertenencia usando un gráfico de barras para la distancia al centro del clúster
plt.figure(figsize=(10, 6))  # Tamaño de la figura
sns.barplot(x='Número del caso', y='Distancia', hue='Clúster', data=tabla_clusper)  # Gráfico de barras con la distancia a los centros
plt.title('Distancia al centro del clúster por número de caso')
plt.show()

# ---- Parte 3: Centros de Clústeres Finales ----
# Crear un DataFrame con los centros finales de los clústeres
centros_finales_df = pd.DataFrame(kmeans.cluster_centers_, columns=variables_importancia.columns)  # Crear DataFrame con los centros de los clústeres

# Desnormalizar los centros para que los valores vuelvan a su escala original
centros_finales_desnormalizados = scaler.inverse_transform(centros_finales_df)  # Desnormalizar los valores

# Crear un DataFrame con los valores desnormalizados
centros_finales_desnormalizados_df = pd.DataFrame(centros_finales_desnormalizados, columns=variables_importancia.columns)  # Crear DataFrame con los centros originales

# Renombrar las columnas para que representen las variables correctamente
centros_finales_desnormalizados_df.columns = ['Importancia que da al entorno', 
                                              'Importancia que da a la gastronomía',
                                              'Importancia que da al costo del viaje',
                                              'Importancia que da a la vida nocturna',
                                              'Importancia que da al alojamiento',
                                              'Importancia que da al arte y cultura']

# Agregar una columna para indicar a qué clúster corresponde cada centro
centros_finales_desnormalizados_df.index = ['Clúster 1', 'Clúster 2', 'Clúster 3']  # Asignar nombres de clústeres

# Mostrar la tabla de centros de clústeres finales
print("\nCentros de clústeres finales:")
print(centros_finales_desnormalizados_df)

# ---- Parte 4: Cuadro de Identificación Corregido ----
# Crear un DataFrame vacío para el cuadro de identificación
cuadro_identificacion = pd.DataFrame('', index=centros_finales_desnormalizados_df.columns, columns=['Grupo 1', 'Grupo 2', 'Grupo 3'])  # Crear DataFrame vacío

# Llenar el cuadro de identificación con "X" donde cada variable es más representativa para cada clúster
for variable in centros_finales_desnormalizados_df.columns:
    # Encontrar el clúster donde la variable tiene el mayor valor
    max_clust = centros_finales_desnormalizados_df.loc[:, variable].idxmax()  # Encontrar el clúster con el valor más alto
    
    # Marcar la columna correspondiente al grupo/clúster
    cuadro_identificacion.loc[variable, max_clust.replace('Clúster', 'Grupo')] = 'X'  # Asignar "X" en el clúster que tiene el valor más alto

# Mostrar el cuadro de identificación corregido
print("\nCuadro de Identificación:")
print(cuadro_identificacion)

# ---- Parte Opcional: Visualización del Cuadro de Identificación ----
plt.figure(figsize=(6, 3))  # Tamaño de la figura
sns.heatmap(cuadro_identificacion.isna(), cmap="Greys", linewidths=0.5, linecolor='black', cbar=False, annot=cuadro_identificacion, fmt='s')  # Crear heatmap
plt.title('Cuadro de Identificación de Grupos')
plt.show()

# ---- Parte 5: Tabla ANOVA ----
# Crear una lista para almacenar los resultados de ANOVA (análisis de varianza)
anova_resultados = {
    'Variable': [],  # Nombre de la variable analizada
    'Media Cuadrática (Clúster)': [],  # Media cuadrática entre los clústeres
    'gl (Clúster)': [],  # Grados de libertad para el clúster
    'Media Cuadrática (Error)': [],  # Media cuadrática del error
    'gl (Error)': [],  # Grados de libertad para el error
    'F': [],  # Valor F de la prueba ANOVA
    'Significancia (Sig.)': []  # Valor p para evaluar la significancia estadística
}

# Calcular ANOVA para cada variable (se compara la variabilidad entre los clústeres frente a la variabilidad interna)
for variable in variables_importancia.columns:
    anova_resultados['Variable'].append(variable)  # Añadir la variable actual a la lista de resultados
    
    # Agrupar los datos por clúster (extraer las observaciones de cada clúster para esa variable)
    grupos = [variables_importancia[variable][data['Clúster'] == i] for i in range(n_clusters)]
    
    # Realizar la prueba ANOVA utilizando f_oneway (compara las medias de los diferentes grupos)
    f_valor, p_valor = f_oneway(*grupos)
    
    # Calcular Media Cuadrática (Clúster y Error)
    ms_clust = np.var([np.mean(grupo) for grupo in grupos])  # Media cuadrática entre los clústeres
    ms_error = np.var(np.concatenate(grupos)) / (len(variables_importancia) - n_clusters)  # Media cuadrática del error
    
    # Añadir los resultados de la ANOVA a las listas correspondientes
    anova_resultados['Media Cuadrática (Clúster)'].append(ms_clust)
    anova_resultados['gl (Clúster)'].append(n_clusters - 1)  # Los grados de libertad del clúster son n_clusters - 1
    anova_resultados['Media Cuadrática (Error)'].append(ms_error)
    anova_resultados['gl (Error)'].append(len(variables_importancia) - n_clusters)  # Los grados de libertad del error
    anova_resultados['F'].append(f_valor)  # Valor F
    anova_resultados['Significancia (Sig.)'].append(p_valor)  # Valor p (significancia)

# Crear un DataFrame para mostrar la tabla ANOVA
anova_df = pd.DataFrame(anova_resultados)

# Mostrar la tabla ANOVA con todos los resultados
print("\nTabla ANOVA:")
print(anova_df)

# ---- Visualización de ANOVA (Opcional) ----
# Crear un heatmap (mapa de calor) para visualizar los resultados ANOVA por variable
plt.figure(figsize=(12, 6))  # Definir el tamaño de la figura
sns.heatmap(anova_df.set_index('Variable'), annot=True, cmap="coolwarm", linewidths=0.5)  # Crear heatmap con anotaciones
plt.title('Resultados ANOVA por Variable')  # Título del gráfico
plt.show()  # Mostrar el gráfico
