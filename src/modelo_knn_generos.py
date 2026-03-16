from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import wandb
import joblib
import pandas as pd
from modelos_utils import download_model_dfs, preprocess_tfidf

def entramiento_modelo_knn_generos(): #Sin parametrizar 
    wandb.init(
        project="clasificacion_generos_knn",
        name="knn_generos_v0",
        config={
            "tfidf_titulo_max_features": 2000,
            "tfidf_descripcion_max_features": 4000,
            "tfidf_tags_max_features": 2000,
            "tfidf_subtitulos_max_features": 5000,
            "ngram_range": (1,2),
            "svd_components": 300,
            "metric": "cosine",
            "k_values": list(range(3,16,2)),
            "cv_folds": 5
        }
    )

    config = wandb.config

    preprocess = ColumnTransformer(
        transformers=[
            ("Titulo", TfidfVectorizer(max_features=config.tfidf_titulo_max_features, ngram_range=tuple(config.ngram_range)), "Titulo"),
            ("Descripcion", TfidfVectorizer(max_features=config.tfidf_descripcion_max_features, ngram_range=tuple(config.ngram_range)), "Descripcion"),
            ("Tags", TfidfVectorizer(max_features=config.tfidf_tags_max_features, ngram_range=tuple(config.ngram_range)), "Tags"),
            ("Subtitulos", TfidfVectorizer(max_features=config.tfidf_subtitulos_max_features, ngram_range=tuple(config.ngram_range)), "Subtitulos"),
            ("Rango_edad", OneHotEncoder(), ["Rango_edad"]),
            ("Duracion", StandardScaler(), ["Duracion"])
        ]
    )
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

    results_table = wandb.Table(columns=["k", "cv_accuracy"])

    for k in config.k_values:

        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=config.svd_components)),
            ("model", KNeighborsClassifier(n_neighbors=k, metric=config.metric))
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=config.cv_folds,
            scoring="accuracy",
            n_jobs=-1
        )

        acc = scores.mean()

        print(f"K={k} -> CV accuracy: {acc:.4f}")

        results_table.add_data(k, acc)

        if acc > best_acc:
            best_acc = acc
            best_k = k

    wandb.log({"cv_results": results_table})

    best_model = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=config.svd_components)),
        ("model", KNeighborsClassifier(n_neighbors=best_k, metric=config.metric))
    ])

    best_model.fit(X_train, y_train)
    print(f"\nMejor K encontrado: {best_k}")
    print(f"Validation accuracy: {best_acc:.4f}")

    wandb.summary["best_k"] = best_k
    wandb.summary["best_cv_accuracy"] = best_acc
    # Evaluación final con test
    pred_test = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, pred_test)

    print("\n--- RESULTADOS EN TEST ---")
    print("Accuracy:", accuracy_score(y_test, pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))
    
    wandb.log({
        "test_accuracy": test_acc
    })
    # Matriz de confusión
    wandb.sklearn.plot_confusion_matrix(y_test, pred_test)

    # Guardar modelo
    joblib.dump(best_model, "knn_generos_model.joblib")
    wandb.save("knn_generos_model.joblib")

    wandb.finish()

    return best_model

if __name__ == '__main__':
    entramiento_modelo_knn_generos()
