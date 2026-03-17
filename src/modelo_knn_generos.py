from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import wandb
import joblib
import pandas as pd
from preprocess_utils import download_model_dfs, preprocess_tfidf

def entramiento_modelo_knn_generos(
        df_train, df_validation, df_test,
        project_ = "Clasificacion_generos_knn",
        name_ = "knn_generos_v0",
        titulo_ = 2000, 
        descripcion_ = 4000, 
        tags_ = 2000, 
        subtitulos_ = 5000, 
        ngram_range_ = (1,2), 
        svd_ = 300, 
        metric_ = "cosine", 
        k_values_ = [3,5,7,11],
        cv_folds_ = 5, 
        preprocess = preprocess_tfidf

): #Sin parametrizar 
    wandb.init(
        project= project_,
        name= name_,
        config={
            "tfidf_titulo_max_features": titulo_,
            "tfidf_descripcion_max_features": descripcion_,
            "tfidf_tags_max_features": tags_,
            "tfidf_subtitulos_max_features": subtitulos_,
            "ngram_range": ngram_range_,
            "svd_components": svd_,
            "metric": metric_,
            "k_values": k_values_,
            "cv_folds": cv_folds_
        }
    )

    config = wandb.config

    best_k = None
    best_acc = 0
    best_model = None

    #df_train, df_validation, df_test = download_model_dfs()
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
