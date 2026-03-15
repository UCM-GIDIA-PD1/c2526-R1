from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
from modelos_utils import download_model_dfs, preprocess_tfidf 

def entramiento_modelo_knn_generos(): 

    preprocess = preprocess_tfidf
    
    best_k = None
    best_acc = 0
    best_model = None

    df_train, df_validation, df_test = download_model_dfs()
    #print(df_train.columns)
    df_train = pd.concat([df_train, df_validation])
    X_train = df_train.drop(columns=["Generos"])
    y_train = df_train["Generos"]
    #X_validation = df_validation.drop(columns=["Generos"])
    #y_validation = df_validation["Generos"]
    X_test = df_test.drop(columns=["Generos"])
    y_test = df_test["Generos"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    k_values = [3,4,5,6,7,8,9,10,11,12,13,14,15]
    scores_dict = {k: [] for k in k_values}

    for train_idx, val_idx in tqdm(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit preprocess + SVD once per fold
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=300))
        ])

        print("Starting to transform")
        X_tr_trans = pipe.fit_transform(X_tr, y_tr)
        X_val_trans = pipe.transform(X_val)
        print("Finished transforming")

        # Loop only over KNN
        for k in tqdm(k_values, leave=False):
            model = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            model.fit(X_tr_trans, y_tr)

            preds = model.predict(X_val_trans)
            acc = accuracy_score(y_val, preds)

            scores_dict[k].append(acc)


    mean_acc_scores = {k: np.mean(v) for k, v in scores_dict.items()}
    for k in mean_acc_scores.keys():
        print(f"K={k} -> CV accuracy: {mean_acc_scores[k]:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_k = k

        
    best_model = Pipeline([
    ("preprocess", preprocess),
    ("svd", TruncatedSVD(n_components=300)),
    ("model", KNeighborsClassifier(n_neighbors=best_k, metric="cosine"))
    ])

    best_model.fit(X_train, y_train)
    print(f"\nMejor K encontrado: {best_k}")
    print(f"Validation accuracy: {best_acc:.4f}")

    # Evaluación final con test
    pred_test = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    print("Accuracy:", accuracy_score(y_test, pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    return best_model

if __name__ == '__main__':
    entramiento_modelo_knn_generos()
