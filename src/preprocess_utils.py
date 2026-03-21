import isodate
import json
from comun.Server_PD import download_dataframe_minio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from itertools import product
import numpy as np
import pandas as pd

def iso_a_minutos(iso_duration):
    """"
    Funcion que convierte la duracion a minutos
    """
    try:
        duracion = isodate.parse_duration(iso_duration)
        return duracion.total_seconds() / 60
    except:
        return 0
    
def build_score(score, y_val, preds, average = "weighted"): 
    if score == "Accuracy":
        return accuracy_score(y_val, preds)
    elif score == "Precision": 
        return precision_score(y_val, preds, average = average)
    elif score == "Recall": 
        return recall_score(y_val, preds, average = average)
    else:
        return f1_score(y_val, preds, average = average)

def unzip_params(params): #Comprobar que ocurre cuando usas solo una clave
    """
    Dado un conjunto de parametros, hace todas las combinaciones posibles entre todos ellos

    Parameters
    ----------
    params: diccionarios con listas o valores unicos
        Ejemplo: {k: [3,4,5], metric: ["cosine", "normal"]} 

    Returns
    -------
    unzip_params_: lista con diccionarios con todas las mezclas
        Ejemplo: [{k: 3, metric: cosine}, k:3, {metric: normal}...]
    """ 
    keys = params.keys()
    values = params.values()
    unzip_params_ = [dict(zip(keys, v)) for v in product(*values)]
    return unzip_params_


def download_model_dfs(): #Deprecated
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
        df_train = download_dataframe_minio("pd1", "grupo1/modelos/train_no_filters", claves, "parquet")
        df_validation = download_dataframe_minio("pd1", "grupo1/modelos/validation_no_filters", claves, "parquet")
        df_test = download_dataframe_minio("pd1", "grupo1/modelos/test_no_filters", claves, "parquet")
        df_train['Duracion'] = df_train['Duracion'].apply(iso_a_minutos)
        df_validation['Duracion'] = df_validation['Duracion'].apply(iso_a_minutos)
        df_test['Duracion'] = df_test['Duracion'].apply(iso_a_minutos)
        return df_train, df_validation, df_test
    
def download_model_dfs_filtered(): #Deprecated
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
        df_train_filtered = download_dataframe_minio("pd1", "grupo1/modelos/train_filtered", claves, "parquet")
        df_validation_filtered = download_dataframe_minio("pd1", "grupo1/modelos/validation_filtered", claves, "parquet")
        df_test_filtered = download_dataframe_minio("pd1", "grupo1/modelos/test_filtered", claves, "parquet")

        df_train_filtered['Duracion'] = df_train_filtered['Duracion'].apply(iso_a_minutos)
        df_validation_filtered['Duracion'] = df_validation_filtered['Duracion'].apply(iso_a_minutos)
        df_test_filtered['Duracion'] = df_test_filtered['Duracion'].apply(iso_a_minutos)
        return df_train_filtered, df_validation_filtered, df_test_filtered

def download_and_divide(to_predict): #Deprecated
    df_train, df_validation, df_test = download_model_dfs()
    df_train = pd.concat([df_train, df_validation])
    X_train = df_train.drop(columns=[to_predict])
    y_train = df_train[to_predict]
    
    X_test = df_test.drop(columns=[to_predict])
    y_test = df_test[to_predict]
    return X_train, y_train, X_test, y_test

types_of_prepro = {"Titulo": "text", "Descripcion": "text", "Tags": "text", "Subtitulos": "text", 
                   "Rango_edad": OneHotEncoder(), "Generos": OneHotEncoder(), 
                   "Duracion": StandardScaler(), "Made for kids": "passthrough"} #passthrough marca que no se hacen transformaciones

def build_preprocess_bow(columns):
    transformers = []
    for col in columns:
        if types_of_prepro[col] == "text":
            transformers.append((col, CountVectorizer(max_features=5000, ngram_range=(1,2)), col))
        else:
            transformers.append((col, types_of_prepro[col], [col]))
    return ColumnTransformer(transformers = transformers)
 
def build_preprocess_tfidf(columns):
    transformers = []
    for col in columns:
        if types_of_prepro[col] == "text":
            transformers.append((col, TfidfVectorizer(max_features=5000, ngram_range=(1,2)), col))
        elif types_of_prepro[col] == "skip": 
             transformers.append((col, "passthrough", [col])) #No hace transformaciones para made for kids porque ya es booleana
        else:
            transformers.append((col, types_of_prepro[col], [col]))
    return ColumnTransformer(transformers = transformers)

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
        
def build_preprocess_word2vec(X_tr, columns):
    #preprocesamiento para word2vec
    all_text = ( #asumiendo que estas cuatro columnas van a estar
            X_tr["Titulo"].astype(str) + " " +
            X_tr["Descripcion"].astype(str) + " " +
            X_tr["Tags"].astype(str) + " " +
            X_tr["Subtitulos"].astype(str)
        )

    sentences = [simple_preprocess(text) for text in all_text]

    model = Word2Vec(
            sentences=sentences,
            vector_size=300,
            window=5,
            min_count=2,
            workers=4,
            seed = 42
        )
    
    transformers = []
    for col in columns:
        if types_of_prepro[col] == "text":
            transformers.append((col, Word2VecVectorizer(model), col))
        else:
            transformers.append((col, types_of_prepro[col], [col]))
    return ColumnTransformer(transformers = transformers)

def build_preprocess(type, columns, X_tr):
    if (type == "Bag of words"):
        return build_preprocess_bow(columns)
    elif (type == "TF-IDF"):
        return build_preprocess_tfidf(columns)
    elif (type == "Word2Vec"):
        return build_preprocess_word2vec(X_tr, columns)
    else:
        raise Exception("Preprocess type not valid. Valid types are Bag of words, TF-IDF and Word2Vec.")
    
