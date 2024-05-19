#!/usr/bin/env python
# coding: utf-8

# In[261]:


#Cargar Librerias que necesitamos
import pandas as pd #estrucutra de datos 
import numpy as np #calculo metrico y analisis de datos
import seaborn as sns #viasulizar datos estadisticos
import matplotlib.pyplot as plt #graficas
import matplotlib.pyplot as plot
import plotly.express as px

#Librerias de aprendizaje no supervisado
from sklearn.cluster import AgglomerativeClustering  
from sklearn.cluster import KMeans
#realiza clustering jerárquico aglomerativo
from scipy.cluster.hierarchy import dendrogram, linkage   #visualizar y calcular la estructura jerárquica de los clusters
import matplotlib.pyplot as plt   #muestra los resultados :)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score  #culcular métricas comunes para evaluar la calidad de los clusters, como el coeficiente de silueta, el índice de Davies-Bouldin y el índice de Calinski-Harabasz.


# 𝟭.𝙍𝙚𝙖𝙡𝙞𝙯𝙖𝙧 𝙪𝙣 𝙖𝙣𝙖́𝙡𝙞𝙨𝙞𝙨 𝙚𝙭𝙥𝙡𝙤𝙧𝙖𝙩𝙤𝙧𝙞𝙤 𝙙𝙚 𝙡𝙤𝙨 𝙙𝙖𝙩𝙤𝙨 𝙥𝙖𝙧𝙖 𝙞𝙙𝙚𝙣𝙩𝙞𝙛𝙞𝙘𝙖𝙧 
# 𝙧𝙚𝙡𝙖𝙘𝙞𝙤𝙣𝙚𝙨 𝙚𝙣𝙩𝙧𝙚 𝙫𝙖𝙧𝙞𝙖𝙗𝙡𝙚𝙨, 𝙫𝙖𝙡𝙤𝙧𝙚𝙨 𝙖𝙩𝙞́𝙥𝙞𝙘𝙤𝙨, 𝙩𝙚𝙣𝙙𝙚𝙣𝙘𝙞𝙖𝙨, 𝙚𝙩𝙘. 

# In[262]:


datos = pd.read_csv('C:/Users/saman/Downloads/Algoritmos de Aprendizaje no Supervisado/Clientes_CC.csv') #Se caraga la data y se guarda en datos


# In[263]:


datos.head(30) #visualizacion y descripcion  de los datos


# In[264]:


plt.figure(figsize=(6, 3))
sns.boxplot(x=datos['Edad'])
plt.title('Identificación de datos faltantes y atípicos en la variable edad', fontsize=10)
plt.show()


# In[265]:


plt.figure(figsize=(6, 3))
sns.boxplot(x=datos['Ingresos_anuales(k$)'])
plt.title('Identificación de datos faltantes y atípicos en la variable Ingresos anuales', fontsize=10)
plt.show()


# In[266]:


plt.figure(figsize=(6, 3))
sns.boxplot(x=datos['Puntuacion_de_gasto (1-100)'])
plt.title('Identificación de datos faltantes y atípicos en la variable Puntuacion de gasto', fontsize=10)
plt.show()


# In[267]:


datos.describe() #trae diversas medidas estadisticas 


# In[268]:


X = datos['Edad']  #accede a la variable datos del dataframe
Y = datos['Ingresos_anuales(k$)'] #accede a la variable Ingresos Anuales del dataframe
plot.plot(X, Y, "b*") #Crea grafica de dispersion e indica que usa punto azules para marcar los datos
plot.xlabel("Edad(X)") #establece la etiqueta del eje xxx jeje
plot.ylabel("Ingresos anuales(y)") #establece la etiqueta del eje y , en este caso es ingresos anuales
plot.grid() #hace la cuadricula 
plot.show() #muestra nuestras super configuraciones ;)


# In[269]:


X = datos['Edad']  
Y = datos['Genero'] 
plot.plot(X, Y, "b*") 
plot.xlabel("Edad(X)") 
plot.ylabel("Genero(y)") 
plot.grid() 
plot.show() 


# 𝟮. 𝙋𝙧𝙚𝙥𝙧𝙤𝙘𝙚𝙨𝙖𝙧 𝙡𝙤𝙨 𝙙𝙖𝙩𝙤𝙨 𝙡𝙞𝙢𝙥𝙞𝙖́𝙣𝙙𝙤𝙡𝙤𝙨, 𝙩𝙧𝙖𝙩𝙖𝙣𝙙𝙤 𝙫𝙖𝙡𝙤𝙧𝙚𝙨 𝙛𝙖𝙡𝙩𝙖𝙣𝙩𝙚𝙨 𝙮 𝙩𝙧𝙖𝙣𝙨𝙛𝙤𝙧𝙢𝙖́𝙣𝙙𝙤𝙡𝙤𝙨 𝙨𝙚𝙜𝙪́𝙣 𝙨𝙚𝙖 𝙣𝙚𝙘𝙚𝙨𝙖𝙧𝙞𝙤. 

# In[270]:


print("Cantidad de datos faltantes en la categoría ClienteID:", datos['ClienteID'].isin([0]).sum())
print("Cantidad de datos faltantes en la categoría Genero:", datos['Genero'].isin([0]).sum())
print("Cantidad de datos faltantes en la categoría Edad:", datos['Edad'].isin([0]).sum())
print("Cantidad de datos faltantes en la categoría Ingresos_anuales(k$):", datos['Ingresos_anuales(k$)'].isin([0]).sum())
print("Cantidad de datos faltantes en la categoría Puntuacion_de_gasto (1-100):", datos['Puntuacion_de_gasto (1-100)'].isin([0]).sum())



# In[271]:


datos.isna().sum()  #mujer prevenida vale x2 confirmacion de datos nulos, de forma simple 


# In[272]:


datos.info() #Revision de tipo de datos


# In[273]:


print(datos['Genero'].info()) #Fprma de hacerlo de manera individual


# 𝟯. 𝙎𝙚𝙡𝙚𝙘𝙘𝙞𝙤𝙣𝙖𝙧 𝙡𝙖𝙨 𝙘𝙖𝙧𝙖𝙘𝙩𝙚𝙧𝙞́𝙨𝙩𝙞𝙘𝙖𝙨 𝙢𝙖́𝙨 𝙧𝙚𝙡𝙚𝙫𝙖𝙣𝙩𝙚𝙨 𝙥𝙖𝙧𝙖 𝙚𝙣𝙩𝙧𝙚𝙣𝙖𝙧 𝙚𝙡 
# 𝙢𝙤𝙙𝙚𝙡𝙤 𝙪𝙩𝙞𝙡𝙞𝙯𝙖𝙣𝙙𝙤 𝙨𝙚𝙡𝙚𝙘𝙘𝙞𝙤́𝙣 𝙙𝙚 𝙘𝙖𝙧𝙖𝙘𝙩𝙚𝙧𝙞́𝙨𝙩𝙞𝙘𝙖�
# •	Ingresos Anuales 
# •	puntuación de gasto  
# En relación con la ed, esto para evaluar la relacion de credito, con respecto al salario y la edad de los compradores d
# �

# In[274]:


#Filtrar las columnas de datoos 
datos = datos[["Edad", "Ingresos_anuales(k$)", "Puntuacion_de_gasto (1-100)"]]
datos.head(5) #Revision de las variables


# In[275]:


#Grafica antes del agrupamiento 

plt.scatter(datos['Ingresos_anuales(k$)'], datos['Puntuacion_de_gasto (1-100)']) #cada punto representara un cliente, donde el eje x es ingresos anuales y el eje y y putnacion de gasto 
#Python es sensible con los espacios, tenerlos en cuenta en las variables ejeje
plt.xlabel('Ingresos_anuales(k$)') #etiqueta de nuestro eje x
plt.ylabel('Puntuacion_de_gasto(1-100)') #etiqueta de nuestro eje y
plt.title('Grupos de clientes en relacion de gasto a sus ingresos')  #titutlo de la grafica
plt.show() #que nos muestre todo jeje


# 𝟰. 𝙀𝙣𝙩𝙧𝙚𝙣𝙖𝙧 𝙚𝙡 𝙢𝙤𝙙𝙚𝙡𝙤 𝙘𝙤𝙣𝙛𝙞𝙜𝙪𝙧𝙖𝙣𝙙𝙤 𝙡𝙤𝙨 𝙙𝙞𝙛𝙚𝙧𝙚𝙣𝙩𝙚𝙨 
# 𝙝𝙞𝙥𝙚𝙧𝙥𝙖𝙧𝙖́𝙢𝙚𝙩𝙧𝙤𝙨.

# In[276]:


#grafica para identificar el valor de k (k=nunmero de agrupamiento de los datos)
#utiliza buena memoria jeje abrir cmd y poner set OMP_NUM_THREADS=1
#esta grafica nos ayuda a saber en cuanto dividir el algoritmo, sabemos que se puede dividir ya cuando la linea empieza a ponerse derecha, en este caso segun mi super analisis es 5 
#ademas tambien se tiene en cuenta las sepraciones entre punto y punto, asi que si es 5 ;)
Nc = range(1, 8) #crea el rango y lo gurada en Nc
kmeans = [KMeans(n_clusters=i) for i in Nc] #crea la lista la cual se llama kmeans con difrentes cantidades de clusters En la primera iteración del bucle for, se crea un objeto KMeans con 1 cluster, en la segunda repeticion se crea otro objeto con 2 clusters, y así sucesivamente hasta 7 clusters.
print(kmeans) #imprime kmeans
score = [kmeans[i].fit(datos).score(datos) for i in range(len(kmeans))] 
print(score)
plt.plot(Nc,score,marker='o') #Se utiliza matplotlib para crear un gráfico de línea. En el eje x se colocan los números de clusters (Nc), y en el eje y se colocan los puntajes de ajuste (score). Se utiliza un marcador circular ('o') en cada punto del gráfico.
plt.xlabel('Numero de clusters')
plt.ylabel('Score')
plt.title('Garfica de codo')
plt.show()


# In[277]:


#agrupamiento de 5 en K means 

modelo = KMeans(n_clusters=5, random_state=0) #creo la variable modelo con 5 clusters
modelo.fit(datos) #que entrene jeje


# In[278]:


modelo.labels_ #accede ala etiquetes ac¿siganadas al modelo kmeans
datos['Numero_Grupo'] = modelo.labels_  #columna adicional que muestra el grupo asiganado , recordar que arranca desde el 0
print(datos) #imprime los datos ;)


# 𝟱. 𝙀𝙫𝙖𝙡𝙪𝙖𝙧 𝙚𝙡 𝙙𝙚𝙨𝙚𝙢𝙥𝙚𝙣̃𝙤 𝙙𝙚𝙡 𝙢𝙤𝙙𝙚𝙡𝙤 𝙘𝙤𝙣 𝙢𝙚́𝙩𝙧𝙞𝙘𝙖𝙨 𝙘𝙤𝙢𝙤 𝘾𝙤𝙚𝙛𝙞𝙘𝙞𝙚𝙣𝙩𝙚 𝙙𝙚 𝙎𝙞𝙡𝙝𝙤𝙪𝙚𝙩𝙩𝙚, 𝙄́𝙣𝙙𝙞𝙘𝙚 𝙙𝙚 𝘾𝙖𝙡𝙞𝙣𝙨𝙠𝙞-𝙃𝙖𝙧𝙖𝙗𝙖𝙨𝙯, 𝙚𝙩𝙘.

# In[279]:


#Se evelaura el desempeño del modelo a tarves de : Coeficiente de Silhouette, Índice de Calinski-Harabasz,Bouldin score
#prepracion de la metrica de la columna de grupo
observaciones = len(datos)
X = datos.drop('Numero_Grupo', axis = 1)
clusters = datos['Numero_Grupo'] 

#calculo de metricas
sil_score = silhouette_score(X, clusters)
calinski_score = calinski_harabasz_score(X, clusters)
davies_score = davies_bouldin_score(X, clusters)

#tabla para visualizacion de datos 
table_data = [
    ["Numero de Observaciones:", observaciones],
    ["Coeficiente silhoutte: ", sil_score],
    ['Indice Calinski Harbasz: ', calinski_score],
    ['Indice Davies Bouldin: ', davies_score]
 ]

#Mostrar tabla
from tabulate import tabulate
print (tabulate(table_data, headers=["Metric", "value"], tablefmt='pretty'))



# 𝟲. 𝙍𝙚𝙖𝙡𝙞𝙯𝙖𝙧 𝙡𝙖𝙨 𝙙𝙞𝙛𝙚𝙧𝙚𝙣𝙩𝙚𝙨 𝙜𝙧𝙖́𝙛𝙞𝙘𝙖𝙨 𝙦𝙪𝙚 𝙥𝙚𝙧𝙢𝙞𝙩𝙖𝙣 𝙫𝙞𝙨𝙪𝙖𝙡𝙞𝙯𝙖𝙧 𝙡𝙤𝙨
# 𝙧𝙚𝙨𝙪𝙡𝙩𝙖𝙙𝙤𝙨 𝙙𝙚𝙡 𝙢𝙤𝙙𝙚𝙡𝙤

# In[280]:


plt.scatter(datos['Ingresos_anuales(k$)'], datos['Puntuacion_de_gasto (1-100)'],c=datos['Numero_Grupo'], cmap='viridis') #cada punto representara un cliente, donde el eje x es ingresos anuales y el eje y y putnacion de gasto 
#Python es sensible con los espacios, tenerlos en cuenta en las variables ejeje
plt.xlabel('Ingresos_anuales(k$)') #etiqueta de nuestro eje x
plt.ylabel('Puntuacion_de_gasto(1-100)') #etiqueta de nuestro eje y
plt.title('Grupos de clientes en relacion de gasto a sus ingresos')  #titutlo de la grafica
plt.show()


# In[285]:


#modelo 3d que relacione los datos agrupada, relacionando ahora tamvien la edad
Grafica_3d = px.scatter_3d(datos, 
                           x='Ingresos_anuales(k$)', 
                           y='Puntuacion_de_gasto (1-100)', 
                           z='Edad', 
                           color='Numero_Grupo', 
                           color_continuous_scale=px.colors.qualitative.Set1)
Grafica_3d.update_layout(showlegend=False)


# 𝟳. 𝙄𝙣𝙩𝙚𝙧𝙥𝙧𝙚𝙩𝙖𝙧, 𝙖𝙣𝙖𝙡𝙞𝙯𝙖𝙧 𝙮 𝙙𝙤𝙘𝙪𝙢𝙚𝙣𝙩𝙖𝙧 𝙡𝙤𝙨 𝙧𝙚𝙨𝙪𝙡𝙩𝙖𝙙𝙤𝙨 𝙤𝙗𝙩𝙚𝙣𝙞𝙙𝙤𝙨.
# 

# •	𝐄𝐧 𝐞𝐥 𝐚𝐧𝐚́𝐥𝐢𝐬𝐢𝐬 𝐬𝐞 𝐨𝐛𝐭𝐮𝐯𝐨 𝐮𝐧 𝐂𝐨𝐞𝐟𝐢𝐜𝐢𝐞𝐧𝐭𝐞 𝐬𝐢𝐥𝐡𝐨𝐮𝐭𝐭𝐞:  𝟎.𝟒𝟒𝟒, 𝐪𝐮𝐞 𝐬𝐢 𝐛𝐢𝐞𝐧 𝐩𝐨𝐝𝐫𝐢́𝐚 𝐞𝐬𝐭𝐚𝐫 𝐦𝐚𝐬 𝐜𝐞𝐫𝐜𝐚 𝐚 𝐮𝐧𝐨, 𝐧𝐨  𝐞𝐬 𝐮𝐧 𝐦𝐚𝐥 𝐧𝐮𝐦𝐞𝐫𝐨 𝐝𝐞 𝐜𝐨𝐞𝐟𝐢𝐜𝐢𝐞𝐧𝐭𝐞 𝐲 𝐧𝐨𝐬 𝐢𝐧𝐝𝐢𝐜𝐚 𝐪𝐮𝐞 𝐥𝐚𝐬 𝐦𝐮𝐞𝐬𝐭𝐫𝐚𝐬 𝐞𝐬𝐭𝐚́𝐧 𝐫𝐞𝐥𝐚𝐭𝐢𝐯𝐚𝐦𝐞𝐧𝐭𝐞 𝐛𝐢𝐞𝐧 𝐚𝐠𝐫𝐮𝐩𝐚𝐝𝐚𝐬 𝐲 𝐞𝐬𝐭𝐚́𝐧 𝐬𝐞𝐩𝐚𝐫𝐚𝐝𝐚𝐬 𝐞𝐧𝐭𝐫𝐞 𝐬𝐢́. 𝐑𝐞𝐜𝐨𝐫𝐝𝐞𝐦𝐨𝐬 𝐪𝐮𝐞 𝐞𝐬𝐭𝐞 𝐜𝐨𝐞𝐟𝐢𝐜𝐢𝐞𝐧𝐭𝐞 𝐦𝐢𝐝𝐞 𝐥𝐚 𝐜𝐚𝐥𝐢𝐝𝐚𝐝 𝐝𝐞 𝐧𝐮𝐞𝐬𝐭𝐫𝐨 𝐜𝐥𝐮𝐬𝐭𝐞𝐫𝐢𝐧𝐠 (𝐬𝐞𝐩𝐚𝐫𝐚𝐜𝐢𝐨́𝐧 𝐝𝐞 𝐧𝐮𝐞𝐬𝐭𝐫𝐨𝐬 𝐝𝐚𝐭𝐨𝐬)
# 
# 
# •	𝐒𝐞 𝐨𝐛𝐭𝐮𝐯𝐨 𝐮𝐧 𝐈́𝐧𝐝𝐢𝐜𝐞 𝐂𝐚𝐥𝐢𝐧𝐬𝐤𝐢 𝐇𝐚𝐫𝐛𝐚𝐬𝐳: 𝟏𝟓𝟏.𝟎𝟒𝟑, 𝐮𝐬𝐮𝐚𝐥𝐦𝐞𝐧𝐭𝐞 𝐞𝐧𝐭𝐫𝐞 𝐦𝐚𝐬 𝐚𝐥𝐭𝐨 𝐦𝐞𝐣𝐨𝐫 𝐟𝐮𝐞 𝐥𝐚 𝐚𝐠𝐫𝐮𝐩𝐚𝐜𝐢𝐨́𝐧, 𝐦𝐮𝐞𝐬𝐭𝐫𝐚 𝐪𝐮𝐞 𝐥𝐨𝐬 𝐠𝐫𝐮𝐩𝐨𝐬 𝐞𝐬𝐭𝐚́𝐧 𝐛𝐢𝐞𝐧 𝐝𝐞𝐟𝐢𝐧𝐢𝐝𝐨𝐬 𝐲 𝐪𝐮𝐞 𝐥𝐚 𝐦𝐮𝐞𝐬𝐭𝐫𝐚𝐬 𝐬𝐨𝐧 𝐜𝐞𝐫𝐜𝐚𝐧𝐚𝐬 𝐞𝐧𝐭𝐫𝐞 𝐬𝐢́.
# 
# 
# •	𝐒𝐞 𝐨𝐛𝐭𝐮𝐯𝐨 𝐈𝐧𝐝𝐢𝐜𝐞 𝐃𝐚𝐯𝐢𝐞𝐬 𝐁𝐨𝐮𝐥𝐝𝐢𝐧:   𝟎.𝟖𝟐𝟏, 𝐞𝐬 𝐮𝐧 𝐜𝐨𝐟𝐢𝐜𝐢𝐞𝐧𝐭𝐞 𝐛𝐚𝐣𝐨 𝐲 𝐩𝐨𝐫 𝐥𝐨 𝐭𝐚𝐧𝐭𝐨 𝐢𝐧𝐝𝐢𝐜𝐚 𝐪𝐮𝐞 𝐞𝐬 𝐦𝐮𝐲 𝐛𝐮𝐞𝐧𝐨 𝐞𝐧 𝐥𝐚 𝐜𝐚𝐥𝐢𝐝𝐚𝐝 𝐝𝐞 𝐥𝐚 𝐬𝐞𝐩𝐚𝐫𝐚𝐜𝐢𝐨́𝐧 𝐝𝐞 𝐠𝐫𝐮𝐩𝐨𝐬 𝐲 𝐛𝐚𝐬𝐭𝐚𝐧𝐭𝐞 𝐜𝐨𝐦𝐩𝐚𝐜𝐭𝐨𝐬.
# 
# 
# •	𝐄𝐧 𝐠𝐞𝐧𝐞𝐫𝐚𝐥 𝐞𝐧 𝐥𝐨𝐬 𝐭𝐫𝐞𝐬 𝐢́𝐧𝐝𝐢𝐜𝐞𝐬 𝐬𝐞 𝐨𝐛𝐭𝐮𝐯𝐢𝐞𝐫𝐨𝐧 𝐛𝐮𝐞𝐧𝐨𝐬 𝐫𝐞𝐬𝐮𝐥𝐭𝐚𝐝𝐨𝐬, 𝐩𝐨𝐫 𝐞𝐧𝐝𝐞, 𝐧𝐨𝐬 𝐝𝐚𝐦𝐨𝐬 𝐜𝐮𝐞𝐧𝐭𝐚 𝐝𝐞 𝐪𝐮𝐞 𝐥𝐚 𝐬𝐞𝐩𝐚𝐫𝐚𝐜𝐢𝐨́𝐧 𝐲 𝐞𝐥 𝐚𝐠𝐫𝐮𝐩𝐚𝐦𝐢𝐞𝐧𝐭𝐨 𝐟𝐮𝐞 𝐞𝐱𝐢𝐭𝐨𝐬𝐨. 𝐌𝐨𝐬𝐭𝐫𝐚́𝐧𝐝𝐨𝐧𝐨𝐬 𝐥𝐨𝐬 𝐠𝐫𝐮𝐩𝐨𝐬 𝐪𝐮𝐞 𝐭𝐢𝐞𝐧𝐞 𝐦𝐚𝐲𝐨𝐫 𝐞𝐱𝐩𝐞𝐜𝐭𝐚𝐭𝐢𝐯𝐚 𝐞𝐧 𝐚𝐝𝐪𝐮𝐢𝐫𝐢𝐫 𝐜𝐫𝐞́𝐝𝐢𝐭𝐨 𝐞𝐧 𝐞𝐥 𝐦𝐚𝐥𝐥, 𝐜𝐨𝐧 𝐫𝐞𝐥𝐚𝐜𝐢𝐨́𝐧 𝐚 𝐞𝐝𝐚𝐝, 𝐬𝐚𝐥𝐚𝐫𝐢𝐨 𝐲 𝐜𝐫𝐞́𝐝𝐢𝐭𝐨. 
# 

# In[ ]:




