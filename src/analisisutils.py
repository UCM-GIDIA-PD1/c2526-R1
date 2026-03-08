#Carga de datos
import pandas as pd
import json
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

#Desfasada --> Eliminable 
def graficar_palabras_comunes(df, columna, titulo_grafica, ax_obj, color):
    
    """"
    Funcion que genera un barplot de las top 10 palabras más comunes del título por cada grupo. 
    """

    keywords = obtener_keywords_en(df[columna], top_n=10)
    
    if not keywords:
        ax_obj.set_title(f"{titulo_grafica} (Sin datos)")
        return 

    # Desempaquetamos los resultados
    palabras_lista, conteos = zip(*keywords)
    
    # Dibujar grafica
    sns.barplot(x=list(conteos), y=list(palabras_lista), ax=ax_obj, color=color)
    ax_obj.set_title(titulo_grafica)
    ax_obj.set_xlabel('Frecuencia')
    ax_obj.set_ylabel('Palabra')


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


#Duracion
def graficar_histograma_duracion(df, titulo, ax_obj, color):
    """"
    Funcion que genera un histograma comparativo de la duración de los vídeos
    """
    limite_sugerido = df['Duracion'].quantile(0.98)
    limite_final = int(np.ceil(limite_sugerido / 5) * 5)
    limite_final = max(min(limite_final, 90), 15)

    sns.histplot(df['Duracion'], bins=limite_final, kde=True, color=color, ax=ax_obj, edgecolor='white', alpha=0.7)
    
    ax_obj.set_title(titulo)
    ax_obj.set_xlabel('Minutos')
    ax_obj.set_ylabel('Cantidad de Vídeos')
    
    ax_obj.set_xlim(0, limite_final)

    ticks = np.arange(0, limite_final + 1, 5)
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
    df_resultado["topic"] = topics
    df_resultado["probabilidad"] = probs

    # resumen de temas
    temas_info = topic_model.get_topic_info()

    return topic_model, df_resultado, temas_info

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
    freq_df = pd.DataFrame(freq.items(), columns=["tag", "frecuencia"])
    freq_df = freq_df.sort_values("frecuencia", ascending=False)

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
    Obten los tags más importantes de un dataframe

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

def frecuencia_titulos(df, columna="Titulo"):
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
    Obten los títulso más importantes de un dataframe, en pares y tríos

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
        "Titulo": terms,
        "Score": scores
    }).sort_values("Score", ascending=False)

    return tfidf_df

def channel_embeddings_clustering(df, columna="Titulo_canal", n_clusters=5): #Embedding + Clustering

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
        "ChannelTitle": textos,
        "Cluster": clusters
    })

    return result_df, kmeans

def show_clusters(result_df): 

    clusters = result_df["Cluster"].unique()

    for c in clusters:
        print(f"\nCluster {c}")
        print(result_df[result_df["Cluster"] == c]["ChannelTitle"].head(10))