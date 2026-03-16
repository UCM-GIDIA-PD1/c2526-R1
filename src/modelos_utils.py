import isodate
import json
from comun.Server_PD import download_dataframe_minio
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
import numpy as np

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

#https://www.kaggle.com/code/siddhvr/introduction-to-word-embeddings-with-word2vec
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.model = model
        self.dim = model.vector_size

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        return np.array([self.vectorize_text(text) for text in X])

    def vectorize_text(self, text):
        words = simple_preprocess(text)
        vectors = [self.model.wv[word] for word in words if word in self.model.wv]

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.dim)
        
def build_preprocess_word2vec(model):
    
    preprocess_word2vec = ColumnTransformer(
        transformers=[
            ("Titulo", Word2VecVectorizer(model), "Titulo"),
            ("Descripcion", Word2VecVectorizer(model), "Descripcion"),
            ("Tags", Word2VecVectorizer(model), "Tags"),
            ("Subtitulos", Word2VecVectorizer(model), "Subtitulos"),
            ("Rango_edad", OneHotEncoder(), ["Rango_edad"]),
            ("Duracion", StandardScaler(), ["Duracion"])
        ]
    )

    return preprocess_word2vec