#Carga de datos
import pandas as pd
import numpy as np
from Server_PD import download_dataframe_minio
from bertopic import BERTopic 
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import re

#Graficas
import seaborn as sns
import matplotlib.pyplot as plt
import isodate

from itertools import combinations
from functools import reduce
from collections import Counter
import ast


#transformaciones

def iso_a_minutos(iso_duration):
    """"
    Funcion que convierte la duracion a minutos
    """
    try:
        duracion = isodate.parse_duration(iso_duration)
        return duracion.total_seconds() / 60
    except:
        return 0

def division_edad(df): 
    """
    Divide un dataframe en dos dataframes, uno de niños y otro de adultos

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe con una columna llamada "Rango_edad"

    Returns
    -------
    df_Adult: Pandas.dataframe
        Dataframe de niños
    df_Kids: Pandas.dataframe
        Dataframe de adultos

    """
    rangos_kids = ['0-4', '5-8', '9-12']
    df_Kids = df[df["Rango_edad"].isin(rangos_kids)].copy()
    df_Adult = df[df["Rango_edad"] == "Adult"].copy()

    return df_Adult, df_Kids

def division_generos(df): 
    """
    Divide un dataframe en un diccionario de dataframes,
    uno por cada género de YouTube.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe con una columna llamada 'generos'.

    Returns
    -------
    dict_generos : dict
        Diccionario donde:
        - key: nombre del género
        - value: dataframe con los vídeos de ese género
    """
    youtube_categories = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers"
}
    dict_generos = {}

    for genero in youtube_categories.values():
        df_genero = df[df["Generos"] == genero].copy()
        dict_generos[genero] = df_genero
    
    return dict_generos


def limpieza_final(dato):
    """"
    Funcion que limpia texto, encorchetado y en comillado
    """
    limpio = str(dato).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    
    partes = limpio.split(',')
    
    resultado = []
    for p in partes:
        nombre = re.sub(r'\(.*\)', '', p).strip() 
        if nombre and nombre.lower() != 'none':
            resultado.append(nombre.capitalize())
            
    return list(set(resultado)) 


#Densidad de habla
def calcular_wpm(row):

    """"
    Función que devuelve el numero de palabras por minuto de un vídeo. (Numero total de palabras /
    duración total del vídeo)
    """
    
    # Convertimos a string y  limpiamos nulos
    sub_texto = str(row['Subtitulos'])
    if sub_texto == "None" or sub_texto == "nan" or row['Duracion'] == 0:
        return 0
    
    numero_palabras = len(sub_texto.split())
    return numero_palabras / row['Duracion']

def gráfico_frecuencias_generos(diccionario_generos):
    """
    Muestra un gráficos de las frecuencias de cada género de los datos

    Parameters
    ----------
    diccionario_generos: dicc
        Diccionario de dataframes divididos por géneros

    Returns
    -------
    Imagen: .png
        Imagen del nuevo gráfico
    -------
    """
    #Frecuencias
    generos_nombres = list(diccionario_generos.keys())
    frecuencias = [len(df) for df in diccionario_generos.values()]

    #Ordenamos
    datos_ordenados = pd.Series(frecuencias, index=generos_nombres).sort_values(ascending=False)

    #Grafica
    plt.figure(figsize=(10, 8))
    sns.barplot(x=datos_ordenados.values, y=datos_ordenados.index, palette='viridis')
    #Ticks
    max_frecuencia = datos_ordenados.max()
    tus_ticks = [0, 250, 500]

    paso_automatico = 200 if max_frecuencia < 2000 else 500
    ticks_automaticos = list(range(0, int(max_frecuencia) + paso_automatico, paso_automatico))

    ticks_finales = sorted(list(set(tus_ticks + ticks_automaticos)))
    plt.xticks(ticks_finales)


    plt.title("Distribución de Vídeos en los 15 Géneros Principales")
    plt.xlabel("Cantidad de Vídeos")
    plt.ylabel("Género")
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    plt.show()

#Duracion
def graficar_histograma_duracion(df, titulo, ax_obj, color):
    """"
    Funcion que genera un histograma comparativo de la duración de los vídeos
    """

    sns.histplot(df['Duracion'], bins=np.arange(0,105,5), kde=True, color=color, ax=ax_obj, edgecolor='white', alpha=0.7)
    
    ax_obj.set_title(titulo)
    ax_obj.set_xlabel('Minutos')
    ax_obj.set_ylabel('Cantidad de Vídeos')
    
    ax_obj.set_xlim(0, 100)

    ticks = np.arange(0, 100, 5)
    ax_obj.set_xticks(ticks)
    ax_obj.set_xticklabels([f"{int(t)}" for t in ticks])


#Géneros
#Ahora funciona con una columna seleccionable
#Entre Generos y Subgeneros
def graficar_top_generos(df, titulo_grafica, ax_obj, paleta, columna):
    #  Limpiamos los datos
    generos = df[columna].astype(str).str.split(', ').explode()
    generos = generos[generos != 'None']
    
    # top 10
    top_generos = generos.value_counts().head(10)
    
    # Grafica
    if not top_generos.empty:
        sns.barplot(x=top_generos.values, y=top_generos.index, palette=paleta, ax=ax_obj)
        ax_obj.set_title(titulo_grafica)
        ax_obj.set_xlabel('Frecuencia')
        ax_obj.set_ylabel(columna)
    else:
        ax_obj.set_title(f"{titulo_grafica} (Sin Datos)")

#Ahora funciona con una columna seleccionable
#Entre Generos y Subgeneros
def graficar_generos_ausentes_kids(df_adult, df_Kids, ax_obj, columna):
    # Lista de generos (Adultos)
    gen_adult = df_adult[columna].astype(str).str.split(', ').explode()
    gen_adult = set(gen_adult[gen_adult != 'None'].unique())
    
    # Lista de generos (Kids) 
    gen_kids = df_Kids[columna].astype(str).str.split(', ').explode()
    gen_kids = set(gen_kids[gen_kids != 'None'].unique())
    
    # Géneros que están en Adultos pero no en Kids
    ausentes = list(gen_adult - gen_kids)
    
    # Gráfica
    if ausentes:
        conteo_adult = df_adult[columna].astype(str).str.split(', ').explode()
        datos_grafica = conteo_adult[conteo_adult.isin(ausentes)].value_counts()
        
        sns.barplot(x=datos_grafica.values, y=datos_grafica.index, palette='rocket', ax=ax_obj)
        ax_obj.set_title(columna + ' Exclusivos de Adultos')
        ax_obj.set_xlabel('Frecuencia en Dataset Adulto')
        ax_obj.set_ylabel(columna)
    else:
        ax_obj.set_title("No hay géneros con presencia cero en Kids")

def analizar_bertopic(df, columna="descripcion", idioma="english"): #Funciona con todo tipo de textos
    """
    Analiza textos de un DataFrame usando BERTopic.

    Retorna
    -------
    topic_model : modelo BERTopic entrenado
    df_resultado : DataFrame con el tema asignado a cada fila
    Topic: Temática asignada al texto ---> -1 implica outliers
    Información extra --->
    Probabilidad: Como de seguro está con la asignación de topicos 
    temas_info : información resumida de los temas
    -------
    """

    # eliminar nulos
    textos = df[columna].dropna().astype(str).tolist()

    # crear modelo
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    topic_model = BERTopic(
    embedding_model=embedding_model,
    language=idioma
    )

    # entrenar modelo
    topics, probs = topic_model.fit_transform(textos)

    # dataframe con resultados
    df_resultado = df.loc[df[columna].notna()].copy()
    df_resultado["Topic"] = topics
    df_resultado["Probabilidad"] = probs

    # resumen de temas
    temas_info = topic_model.get_topic_info()

    return topic_model, df_resultado, temas_info

def analizar_bertopic_dict(diccionario_dfs, columna="Titulo", idioma="english"):
    """
    Unifica varios DataFrames y analiza sus textos de un DataFrame usando BERTopic.

    Args
    -------
        diccionario_dfs (dict): Diccionario donde las llaves son categorías y los valores son DataFrames de Pandas.
        columna (str): Nombre de la columna que contiene el texto a analizar.
        idioma (str): Idioma principal de los textos para el modelo. Por defecto "english".

    Retorna
    -------
    topic_model : modelo BERTopic entrenado
    df_resultado : DataFrame con el tema asignado a cada fila
    Topic: Temática asignada al texto ---> -1 implica outliers
    Información extra --->
    Probabilidad: Como de seguro está con la asignación de topicos 
    temas_info : información resumida de los temas
    -------

    """
  
    # unificamos los dfs del diccionario
    df_unificado = pd.concat([df.assign(Genero_Origen=k) for k, df in diccionario_dfs.items()])
    
    # limpieza
    df_clean = df_unificado.dropna(subset=[columna]).copy()
    df_clean[columna] = df_clean[columna].astype(str).str.strip()
    df_clean = df_clean[df_clean[columna] != ""].reset_index(drop=True)

    textos = df_clean[columna].tolist()

    # crear modelo
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    topic_model = BERTopic(embedding_model=embedding_model, language=idioma)

    # entrenar modelo
    topics, probs = topic_model.fit_transform(textos)

    # dataframe con resultados
    df_clean["Topic"] = topics
    df_clean["Probabilidad"] = probs

    # resumen de temas
    temas_info = topic_model.get_topic_info()
    
    return topic_model, df_clean, temas_info

def frecuencia_tags(df, columna="Tags"):
    """
    Localiza las frecuencias de los tags.

    Retorna
    -------
    
    """
    # separar tags
    tags = (
        df[columna]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .str.lower()

    )

    # contar frecuencia
    freq = Counter(tags)

    # convertir a dataframe
    freq_df = pd.DataFrame(freq.items(), columns=["Tag", "Frecuencia"])
    freq_df = freq_df.sort_values("Frecuencia", ascending=False)

    return freq_df

def tfidf_tags(df, columna="Tags"): #Bastante util
    """
    Obten los tags más importantes de un dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe con el que vamos a trabajar.
    
    columna: string
        Columnas por la cual se va a trabajar

    Returns
    -------
    tfidf_df: pandas.Dataframe
        Retorno un dataframe con columnas: Tags, Importancia.
    """
    # reemplazar comas por espacios
    tags = (
        df[columna]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .str.lower()
    )

    # cada tag será un "documento"
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b"
    )

    X = vectorizer.fit_transform(tags)

    palabras = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1

    tfidf_df = pd.DataFrame({
        "Tag": palabras,
        "Score": scores
    }).sort_values("Score", ascending=False)

    return tfidf_df

def ngramas_tags(df, columna="Tags", n=2): #No muy util
    """
    Obten los tags más importantes de un dataframe, en pares o tríos

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe con el que vamos a trabajar.
    
    columna: string
        Columnas por la cual se va a trabajar
    
    n: int
        Número de combinaciones de tags más frecuentes

    Returns
    -------
    ngram_df: pandas.Dataframe
        Retorno un dataframe con columnas: Tags, Frecuencias.
    """
    textos = df[columna].fillna("").str.replace(",", " ")

    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(textos)

    ngrams = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1

    ngram_df = pd.DataFrame({
        "Ngram": ngrams,
        "Frecuencia": counts
    }).sort_values("Frecuencia", ascending=False)

    return ngram_df

def frecuencia_titulos(df, columna="Titulo"): #Caca
    """
    Calcula la frecuencia de palabras en los títulos.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe con los datos.
    
    columna : str
        Columna que contiene los títulos.

    Returns
    -------
    freq_df : pandas.DataFrame
        Dataframe con columnas: Palabra, Frecuencia.
    """

    # separar palabras
    palabras = (
        df[columna]
        .dropna()
        .str.lower()
        .str.split()
        .explode()
        .str.strip()
    )

    # contar frecuencia
    freq = Counter(palabras)

    # convertir a dataframe
    freq_df = pd.DataFrame(freq.items(), columns=["Palabra", "Frecuencia"])
    freq_df = freq_df.sort_values("Frecuencia", ascending=False)

    return freq_df

def tfidf_ngrams_titles(df, columna="Titulo", ngram_range=(1,3), min_df=2):
    """
    Obten los títulos más importantes de un dataframe, en pares y tríos

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe con el que vamos a trabajar.
    
    columna: string
        Columnas por la cual se va a trabajar

    ngram_range: vector
        Rango de valores para combinaciones de palabras (desde 1 a 3 de base)
    
    n: int
        Mínimo de frecuencia para ser considerado

    Returns
    -------
    ngram_df: pandas.Dataframe
        Retorno un dataframe con columnas: Tags, Frecuencias.
    """
    textos = df[columna].fillna("").str.lower()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df
    )

    X = vectorizer.fit_transform(textos)

    terms = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1

    tfidf_df = pd.DataFrame({
        columna: terms,
        "Score": scores
    }).sort_values("Score", ascending=False)

    return tfidf_df

def channel_embeddings_clustering(df, columna="Titulo_canal", n_clusters=5): #Embedding + Clustering
    """
    Divide en clusters ("Categorías") los títulos

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe con el que vamos a trabajar.
    
    columna: string
        Columnas por la cual se va a trabajar

    n_clusters: int
        Numero de clusters con los que vamos a trabajar

    Returns
    -------
    result_df: pandas.Dataframe
        Retorno un dataframe con columnas: Texto, Cluster.
    kmenas: Modelo
        Retorna el modelo sin entrenar
    """
    # limpiar datos
    textos = df[columna].dropna().tolist()

    # modelo de embeddings
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # generar embeddings
    embeddings = model.encode(textos, show_progress_bar=True)

    # clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # dataframe resultado
    result_df = pd.DataFrame({
        columna: textos,
        "Cluster": clusters
    })

    return result_df, kmeans

def show_clusters(result_df): 
    """
    Muestra todos los clusters generados
    """
    clusters = result_df["Cluster"].unique()
    

    for c in clusters:
        print(f"\nCluster {c}")
        print(result_df[result_df["Cluster"] == c]["ChannelTitle"].head(10))

def common_terms (df_1, df_2, columna): 
    """
    Agrupa los terminos comunes de df1 y df2 y da una diferencia de score si la hay

    Parameters
    ----------
    df_1 : pandas.DataFrame
        Dataframe 1

    df_2 : pandas.DataFrame
        Dataframe 2
    
    columna: String
        Nombre de la columna original 

    Returns
    -------
    common_terms: pandas.Dataframe
        Retorno un dataframe con columnas: columna, Score_kids, Score_adults, Diff
    """ 

    common_terms = (
    df_1
    .merge(df_2, on= columna, suffixes=("_kids", "_adults"))
)
    if "Score_kids" in common_terms: 
        common_terms["Diff"] = common_terms["Score_kids"] - common_terms["Score_adults"]
    
    return common_terms

def exclusive_terms (df_1, df_2, columna): 
    """
    Separa los terminos exclusivos de df1 y df2

    Parameters
    ----------
    df_1 : pandas.DataFrame
        Dataframe 1

    df_2 : pandas.DataFrame
        Dataframe 2
    
    columna: String
        Nombre de la columna original 

    Returns
    -------
    result_df_1: pandas.Dataframe
        Retorno un dataframe con columnas: columna, Score
    result_df_2: pandas.Dataframe
        Retorno un dataframe con columnas: columna, Score
    """ 
    result_df_1 = df_1[~df_1[columna].isin(df_2[columna])]
    result_df_2 = df_2[~df_2[columna].isin(df_1[columna])]

    return result_df_1, result_df_2


#A PARTIR DE AQUI HAY QUE AÑADIR DESCRIPCIONES

def graficar_bertopic (df_1, df_2, nombre1, nombre2, columna):
    """
    Compara la distribución de tópicos entre dos dfs (adults vs kids).
    
    Parameters:
    -----------
    df_1 : pandas.DataFrame
        Dataframe 1

    df_2 : pandas.DataFrame
        Dataframe 2

    nombre 1 : Nombre del grupo 1 

    nombre 2 : Nombre del grupo  2

    columna : String
    """
    # contar topics
    top_1 = df_1["Topic"].value_counts()
    top_2=  df_2["Topic"].value_counts()

    # eliminar outliers
    top_1 = top_1[top_1.index != -1]
    top_2 = top_2[top_2.index != -1]

    # convertir a dataframe
    df_1 = top_1.reset_index()
    df_1.columns = ["Topic", "Count"]
    df_1["Grupo"] = nombre1

    df_2 = top_2.reset_index()
    df_2.columns = ["Topic", "Count"]
    df_2["Grupo"] = nombre2

    # unir
    df_plot = pd.concat([df_1, df_2])

    # quedarnos con los topics más comunes
    top_topics = (
        df_plot.groupby("Topic")["Count"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    df_plot = df_plot[df_plot["Topic"].isin(top_topics)]

    # gráfico
    sns.barplot(data=df_plot, x="Topic", y="Count", hue="Grupo")

    plt.title("Comparación de Topics -" + columna + "(" + nombre1 + "vs" + nombre2 + ")") #se podria añadir si es titulo, descripicion etc
    plt.ylabel("Número de videos")
    plt.xlabel("Topic")    

#COMPROBAR
def common_terms_dictionary(diccionario_dfs, columna_id):
    """
    Agrupa los terminos comunes de los dfs del diciconario y da una diferencia de score si la hay

    Parameters
    ----------
    diccionario_dfs : dicionario con los nombres de los generos como claves y pandas.DataFrame como valores
    
    columna_id: String
        Nombre de la columna original 

    Returns
    -------
    common_terms: pandas.Dataframe
        Retorno un dataframe con columnas: columna, Score_genero1, Score_genero2, ..., Diff
    """ 

    generos = list(diccionario_dfs.keys())
    listado_dfs = []
    
    for genero in generos:
        df_temp = diccionario_dfs[genero].copy()
        
        # 1. Detectamos cuál es la columna de valores (que no sea la de ID como 'Tag' o 'ngram')
        # Buscamos 'Score' o 'Frecuencia'
        col_valor = [c for c in df_temp.columns if c != columna_id][0] 
        
        # 2. Renombramos dinámicamente: 'Frecuencia' -> 'Frecuencia_Education'
        df_temp = df_temp[[columna_id, col_valor]]
        df_temp = df_temp.rename(columns={col_valor: f"{col_valor}_{genero}"})
        
        listado_dfs.append(df_temp)

    # 3. Merge sucesivo
    df_comun = reduce(lambda left, right: pd.merge(left, right, on=columna_id), listado_dfs)

    # 4. Cálculos finales basados en las columnas que se han creado
    # Buscamos todas las que empiecen por Score_ o Frecuencia_
    metric_cols = [c for c in df_comun.columns if c != columna_id]
    
    df_comun['Score_Std'] = df_comun[metric_cols].std(axis=1)
    df_comun['Score_Total'] = df_comun[metric_cols].sum(axis=1)

    return df_comun.sort_values(by='Score_Total', ascending=False)

def comparativa_terminos(generos_a_comparar, df):
    """
    Genera un gráfico de barras comparativo de los términos más relevantes 
    entre varios géneros utilizando sus puntuaciones de importancia.
    
    Args:
        generos_a_comparar (list): Lista de nombres de las columnas a comparar.
        df : pandas.DataFrame (contiene los términos y sus scores por género).
        
    Returns:
        None: Despliega un gráfico de Seaborn.
    """
    lista_procesada = []

    #Creamos un nuevo df con los 4 generos
    for gen in generos_a_comparar:
        df_aux = df[gen].copy()
        df_aux.columns = ['Termino', 'Score'] 
        lista_procesada.append(df_aux.assign(Genero=gen))

    df_comparativa = pd.concat(lista_procesada)

    #Seleccionamos los 15 terminos mas importantes a nivel global
    top_15_global = df_comparativa.groupby('Termino')['Score'].sum().nlargest(15).index
    df_final_plot = df_comparativa[df_comparativa['Termino'].isin(top_15_global)]

    # Grafica
    sns.barplot(data=df_final_plot, x='Score', y='Termino', hue='Genero', palette='muted')

    plt.title('Comparativa de Términos TF-IDF')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.legend(title='Género', bbox_to_anchor=(1.05, 1), loc='upper left')


def graficar_bertopic_multiple(df_bertopic, lista_generos, columna_analizada):
    """
    Compara la distribución de tópicos entre varios géneros.
    
    Parameters:
    -----------
    df_bertopic : DataFrame resultante de analizar_bertopic_dict

    lista_generos : Lista de strings con los nombres de los géneros a comparar

    columna_analizada : String
    """

    # Filtramos los géneros que queremos analizar
    df_filtrado = df_bertopic[df_bertopic['Genero_Origen'].isin(lista_generos)].copy()

    # Agrupamos por género y tópico y contamos
    df_counts = (
        df_filtrado[df_filtrado["Topic"] != -1]
        .groupby(["Genero_Origen", "Topic"])
        .size()
        .reset_index(name="Count")
    )

    # Identificamos los 20 topics más frecuentes
    top_topics = (
        df_counts.groupby("Topic")["Count"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    df_plot = df_counts[df_counts["Topic"].isin(top_topics)]

    # Gráfico
    plt.figure(figsize=(15, 7))
    sns.barplot(data=df_plot, x="Topic", y="Count", hue="Genero_Origen", palette="hsv")

    nombres_vs = " vs ".join(lista_generos)
    plt.title(f"Comparación de Topics - {columna_analizada} ({nombres_vs})")
    plt.ylabel("Número de vídeos")
    plt.xlabel("ID del Topic (Tema)")
    
    plt.legend(title="Género", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
