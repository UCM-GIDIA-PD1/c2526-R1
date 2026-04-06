from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from preprocess_utils import build_preprocess, unzip_params, build_score
from filter_and_divide_data import get_data_models_train_test
from collections import defaultdict
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


import numpy as np
import pandas as pd
import wandb
def evalute_forest():
    X_train, X_test, y_train, y_test = get_data_models_train_test(filtrado = 2, to_predict="Made for kids")
    le = LabelEncoder()
    y_train_encoded = pd.Series(le.fit_transform(y_train))
    y_test_encoded = pd.Series(le.transform(y_test))

    #Entrenamos el mejor modelo
    best_model = run_forest(3000, (1,2), 150, "Word2Vec", 
                              ["Titulo","Descripcion","Tags","Subtitulos","Subgeneros","Duracion","Titulo_canal"], 
                              X_train, y_train_encoded, X_test, y_test_encoded, RandomForestClassifier,
                              {"criterion": "gini", "n_estimators": 30, "max_depth":30, "max_features":"sqrt"},
                                "F1", "weighted", le)
    
    print("\n--- PERMUTATION IMPORTANCE ---")

    result = permutation_importance(
        best_model,
        X_test,
        y_test_encoded,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # IMPORTANTE: usar columnas reales
    perm_imp = pd.Series(
        result.importances_mean,
        index=["Titulo","Descripcion","Tags","Subtitulos","Subgeneros","Duracion","Titulo_canal"]
    ).sort_values(ascending=False)

    print(perm_imp)

    # Plot
    plt.figure()
    perm_imp.sort_values().plot(kind='barh')
    plt.title("Permutation Importance (features originales)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()


    print("\n--- FEATURE IMPORTANCE (componentes SVD) ---")

    rf_model = best_model.named_steps["model"]
    importances = rf_model.feature_importances_

    feat_imp = pd.Series(importances).sort_values(ascending=False)

    print(feat_imp.head(10))

    plt.figure()
    feat_imp.head(20).sort_values().plot(kind='barh')
    plt.title("Importancia (componentes SVD)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()
def run_forest(max_features, ngram, svd, preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, paramset, score, average, le):
    preprocess = build_preprocess(preprocess_type, columns, X_train, max_features, ngram, svd)
    best_model = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=svd, random_state=42)),
        ("model", modelo(**paramset))
    ])

    best_model.fit(X_train, y_train)

    raw_preds = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    best_score_test = build_score(score, y_test, raw_preds, average)
    print("Score:", best_score_test)
    class_names = le.classes_.tolist()
    y_test_text = le.inverse_transform(y_test)
    pred_test_text = le.inverse_transform(raw_preds)

    return best_model
def run_knn(max_features, ngram, svd, preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, paramset, score, average, le):
    preprocess = build_preprocess(preprocess_type, columns, X_train, max_features, ngram, svd)
    best_model = Pipeline([
        ("preprocess", preprocess),
        #("svd", TruncatedSVD(n_components=svd, random_state=42)),
        ("model", modelo(**paramset))
    ])

    best_model.fit(X_train, y_train)

    raw_preds = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    best_score_test = build_score(score, y_test, raw_preds, average)
    print("Score:", best_score_test)
    class_names = le.classes_.tolist()
    y_test_text = le.inverse_transform(y_test)
    pred_test_text = le.inverse_transform(raw_preds)

    return best_model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data_models_train_test(filtrado = 2, to_predict="Generos")
    le = LabelEncoder()
    y_train_encoded = pd.Series(le.fit_transform(y_train))
    y_test_encoded = pd.Series(le.transform(y_test))

    #Entrenamos el mejor modelo
    best_model = run_knn(5000, (1,3), 150, "Word2Vec", 
                              ["Titulo","Descripcion","Tags","Subtitulos","Duracion","Titulo_canal", "Made for kids"], 
                              X_train, y_train_encoded, X_test, y_test_encoded, KNeighborsClassifier,
                              {"metric": "cosine", "n_jobs":-1, "n_neighbors": 6, "weights": "distance"},
                                "F1", "weighted", le)
    
    baseline = best_model.score(X_test, y_test_encoded)

    importancias = {}

    for col in X_test.columns:
        X_test_perm = X_test.copy()
        X_test_perm[col] = np.random.permutation(X_test_perm[col].to_numpy())
        
        score = best_model.score(X_test_perm, y_test_encoded)
        importancias[col] = baseline - score

    print(sorted(importancias.items(), key=lambda x: x[1], reverse=True))