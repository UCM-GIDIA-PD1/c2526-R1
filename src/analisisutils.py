#Carga de datos
import pandas as pd
import json
import numpy as np
from Server_PD import download_dataframe_minio

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
    
    limpio = str(dato).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    
    partes = limpio.split(',')
    
    resultado = []
    for p in partes:
        nombre = re.sub(r'\(.*\)', '', p).strip() 
        if nombre and nombre.lower() != 'none':
            resultado.append(nombre.capitalize())
            
    return list(set(resultado)) 


#keywords

# Lista extendida de stopwords en inglés
stopwords_en = {
    # Tus originales
    'the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 
    'with', 'as', 'i', 'his', 'they', 'be', 'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 
    'hot', 'word', 'but', 'what', 'some', 'we', 'can', 'out', 'other', 'were', 'all', 'there', 
    'when', 'up', 'use', 'your', 'how', 'said', 'an', 'each', 'she', 'which', 'do', 'their', 
    'if', 'will', 'about', 'many', 'then', 'them', 'these', 'so', 'her', 'would', 'make', 
    'like', 'him', 'into', 'time', 'has', 'look', 'two', 'more', 'write', 'go', 'see', 
    'number', 'no', 'way', 'could', 'people', 'my', 'than', 'first', 'water', 'been', 
    'called', 'who', 'am', 'its', 'now', 'find', 'get', 'none', 'full', 'nbsp',
    'because', 'because', 'so', 'yet', 'unless', 'while', 'although', 'since', 'also',
    'just', 'very', 'really', 'even', 'still', 'maybe', 'actually', 'well', 'too', 'only',
    'never', 'always', 'sometimes', 'ever', 'again', 'back', 'here', 'there', 'where',
    'know', 'think', 'want', 'take', 'tell', 'come', 'give', 'mean', 'need', 'should',
    'right', 'let', 'may', 'must', 'keep', 'put', 'seem', 'look', 'much', 'many',
    'me', 'us', 'our', 'mine', 'ours', 'yourself', 'something', 'anything', 'everything',
    'another', 'every', 'own', 'same', 'such', 'very', 'next', 'right', 'okay', 'why', 'here'
    'not', 'yeah', 'kind', 'going', 'today', 'those', 'good', 'thank'
}


def obtener_keywords_en(columna, top_n=20):
    """"
    Funcion que devuelve el top n de palabras más comunes de un columna (título o subtitulos)
    """
    todas_las_palabras = []
    
    for texto in columna:
        texto_limpio = str(texto).lower()
        # Solo extraemos letras
        palabras = re.findall(r'[a-z]+', texto_limpio)
        
        # Filtrar por longitud y stopwords
        palabras_filtradas = [p for p in palabras if p not in stopwords_en and len(p) > 3]
        todas_las_palabras.extend(palabras_filtradas)
    
    return Counter(todas_las_palabras).most_common(top_n)



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

def graficar_top_generos(df, titulo_grafica, ax_obj, paleta):
    #  Limpiamos los datos
    generos = df['Subgeneros'].astype(str).str.split(', ').explode()
    generos = generos[generos != 'None']
    
    # top 10
    top_generos = generos.value_counts().head(10)
    
    # Grafica
    if not top_generos.empty:
        sns.barplot(x=top_generos.values, y=top_generos.index, palette=paleta, ax=ax_obj)
        ax_obj.set_title(titulo_grafica)
        ax_obj.set_xlabel('Frecuencia')
        ax_obj.set_ylabel('Género')
    else:
        ax_obj.set_title(f"{titulo_grafica} (Sin Datos)")


def graficar_generos_ausentes_kids(df_adult, df_Kids, ax_obj):
    # Lista de generos (Adultos)
    gen_adult = df_adult['Subgeneros'].astype(str).str.split(', ').explode()
    gen_adult = set(gen_adult[gen_adult != 'None'].unique())
    
    # Lista de generos (Kids) 
    gen_kids = df_Kids['Subgeneros'].astype(str).str.split(', ').explode()
    gen_kids = set(gen_kids[gen_kids != 'None'].unique())
    
    # Géneros que están en Adultos pero no en Kids
    ausentes = list(gen_adult - gen_kids)
    
    # Gráfica
    if ausentes:
        conteo_adult = df_adult['Subgeneros'].astype(str).str.split(', ').explode()
        datos_grafica = conteo_adult[conteo_adult.isin(ausentes)].value_counts()
        
        sns.barplot(x=datos_grafica.values, y=datos_grafica.index, palette='rocket', ax=ax_obj)
        ax_obj.set_title('Géneros Exclusivos de Adultos')
        ax_obj.set_xlabel('Frecuencia en Dataset Adulto')
        ax_obj.set_ylabel('Género')
    else:
        ax_obj.set_title("No hay géneros con presencia cero en Kids")