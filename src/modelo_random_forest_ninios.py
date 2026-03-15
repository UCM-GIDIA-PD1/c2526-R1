from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import wandb


def entramiento_modelo_decission_tree_ninios(): #Sin parametrizar 
    wandb.init(
        project="clasificacion_generos_knn",
        name="knn_generos_v0",
        config={
            "tfidf_titulo_max_features": 2000,
            "tfidf_descripcion_max_features": 4000,
            "tfidf_tags_max_features": 2000,
            "tfidf_subtitulos_max_features": 5000,
            "ngram_range": (1,2),
            
            "cv_folds": 5
        }
    )

    config = wandb.config

    preprocess = ColumnTransformer(
        transformers=[
            ("Titulo", TfidfVectorizer(max_features=config.tfidf_titulo_max_features, ngram_range=config.ngram_range), "Titulo"),
            ("Descripcion", TfidfVectorizer(max_features=config.tfidf_descripcion_max_features, ngram_range=config.ngram_range), "Descripcion"),
            ("Tags", TfidfVectorizer(max_features=config.tfidf_tags_max_features, ngram_range=config.ngram_range), "Tags"),
            ("Subtitulos", TfidfVectorizer(max_features=config.tfidf_subtitulos_max_features, ngram_range=config.ngram_range), "Subtitulos"),
            ("Rango_edad", OneHotEncoder(), ["Rango_edad"]),
            ("Duracion", StandardScaler(), ["Duracion"])
        ]
    )
    best_k = None
    best_acc = 0
    best_model = None
