import isodate
import json
from tqdm import tqdm
from comun.Server_PD import download_dataframe_minio
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics import accuracy_score, classification_report
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
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
    
def download_and_divide(to_predict):
    df_train, df_validation, df_test = download_model_dfs()
    df_train = pd.concat([df_train, df_validation])
    X_train = df_train.drop(columns=[to_predict])
    y_train = df_train[to_predict]
    
    X_test = df_test.drop(columns=[to_predict])
    y_test = df_test[to_predict]
    return X_train, y_train, X_test, y_test

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
        
def build_preprocess_word2vec(X_tr):
    #preprocesamiento para word2vec
    all_text = (
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
            workers=4
        )
    
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

def build_preprocess(type, X_tr = None):
    if (type == "Bag of words"):
        return preprocess_bag_of_words
    elif (type == "TF-IDF"):
        return preprocess_tfidf
    elif (type == "Word2Vec"):
        # if X_tr == None:
        #     raise Exception("To build Word2Vec preprocesser, please pass X_train")
        # else:
        return build_preprocess_word2vec(X_tr)
    else:
        raise Exception("Preprocess type not valid. Valid types are Bag of words, TF-IDF and Word2Vec.")
    
def run_cross_validation(X_train, y_train, preprocess_type, parameter_name, parameter_vals, modelo, n_splits=5):
    best_param = None
    best_acc = 0
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) #n splits 5

    scores_dict = {k: [] for k in parameter_vals}
    for train_idx, val_idx in tqdm(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        preprocess = build_preprocess(preprocess_type, X_tr)
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=300))
        ])

        X_tr_trans = pipe.fit_transform(X_tr, y_tr)
        X_val_trans = pipe.transform(X_val)

        for param_val in parameter_vals: #un array
            model = modelo(param_val)#parameter_name = param_val) #KNeighborsClassifier(n_neighbors=k, metric="cosine")
            model.fit(X_tr_trans, y_tr)

            preds = model.predict(X_val_trans)
            acc = accuracy_score(y_val, preds)

            scores_dict[param_val].append(acc)

    mean_acc_scores = {k: np.mean(v) for k, v in scores_dict.items()}
    for k in mean_acc_scores.keys():
        print(f"{parameter_name}={k} -> CV accuracy: {mean_acc_scores[k]:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_param = k

    print(f"\nMejor {parameter_name} encontrado: {best_param}")
    print(f"CrossVal accuracy: {best_acc:.4f}")

    return best_acc, best_param
    
def run_best_model(preprocess_type, X_train, y_train, X_test, y_test, modelo, param_name, param_value, metric_val):
    preprocess = build_preprocess(preprocess_type, X_train)
    best_model = Pipeline([
    ("preprocess", preprocess),
    ("svd", TruncatedSVD(n_components=300)),
    ("model", modelo(param_value, metric=metric_val))
    ])

    best_model.fit(X_train, y_train)

    # Evaluación final con test
    pred_test = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    print("Accuracy:", accuracy_score(y_test, pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    return best_model