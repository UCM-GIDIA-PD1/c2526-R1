import isodate
import json
from comun.Server_PD import download_dataframe_minio
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def iso_a_minutos(iso_duration):
    """"
    Funcion que convierte la duracion a minutos
    """
    try:
        duracion = isodate.parse_duration(iso_duration)
        return duracion.total_seconds() / 60
    except:
        return 0
    
def download_model_dfs():
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
        df_train = download_dataframe_minio("pd1", "grupo1/modelos/train_no_filters", claves, "parquet")
        df_validation = download_dataframe_minio("pd1", "grupo1/modelos/validation_no_filters", claves, "parquet")
        df_test = download_dataframe_minio("pd1", "grupo1/modelos/test_no_filters", claves, "parquet")
        df_train['Duracion'] = df_train['Duracion'].apply(iso_a_minutos)
        df_validation['Duracion'] = df_validation['Duracion'].apply(iso_a_minutos)
        df_test['Duracion'] = df_test['Duracion'].apply(iso_a_minutos)
        return df_train, df_validation, df_test

preprocess_bag_of_words = ColumnTransformer(
    transformers=[
        ("Titulo", CountVectorizer(max_features=2000, ngram_range=(1,2)), "Titulo"),
        ("Descripcion", CountVectorizer(max_features=4000, ngram_range=(1,2)), "Descripcion"),
        ("Tags", CountVectorizer(max_features=2000, ngram_range=(1,2)), "Tags"),
        ("Subtitulos", CountVectorizer(max_features=5000, ngram_range=(1,2)), "Subtitulos"),
        ("Rango_edad", OneHotEncoder(), ["Rango_edad"]),
        ("Duracion", StandardScaler(), ["Duracion"])
    ]
    )

preprocess_tfidf = ColumnTransformer(
    transformers=[
        ("Titulo", TfidfVectorizer(max_features=2000, ngram_range=(1,2)), "Titulo"),
        ("Descripcion", TfidfVectorizer(max_features=4000, ngram_range=(1,2)), "Descripcion"),
        ("Tags", TfidfVectorizer(max_features=2000, ngram_range=(1,2)), "Tags"),
        ("Subtitulos", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), "Subtitulos"),
        ("Rango_edad", OneHotEncoder(), ["Rango_edad"]),
        ("Duracion", StandardScaler(), ["Duracion"])
    ]
    )

