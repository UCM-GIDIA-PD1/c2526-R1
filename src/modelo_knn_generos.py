from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
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

    for k in [3,4,5,6,7,8,9,10,11,12,13,14,15]:

        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=300)),
            ("model", KNeighborsClassifier(n_neighbors=k, metric="cosine"))
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        acc = scores.mean()

        print(f"K={k} -> CV accuracy: {acc:.4f}")

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
