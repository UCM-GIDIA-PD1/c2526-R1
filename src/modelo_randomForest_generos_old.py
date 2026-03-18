from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import wandb
import joblib
import pandas as pd
from preprocess_utils import download_model_dfs


def entramiento_modelo_randomForest_generos(): 
    wandb.init(
        project="clasificacion_generos_rf",
        name="rf_generos_v0",
        config={
            "tfidf_titulo_max_features": 2000,
            "tfidf_descripcion_max_features": 4000,
            "tfidf_tags_max_features": 2000,
            "tfidf_subtitulos_max_features": 5000,
            "ngram_range": (1,2),
            "svd_components": 300,
            "cv_folds": 5,
            "criterion": "entropy",
            "n_estimators": [10, 20, 30],
            "max_depth":[10, 15, 20],
            "max_features":	[0.5]
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
    best_md = None
    best_ne = None
    best_mf = None
    best_acc = 0
    best_model = None

    df_train, df_validation, df_test = download_model_dfs()

    
    #temporal
    generos_a_eliminar = [
        "Comedy", 
        "Autos & Vehicles", 
        "Travel & Events", 
        "Nonprofits & Activism", 
        "Pets & Animals"
    ]
    
    # Filtramos los tres datasets
    df_train = df_train[~df_train["Generos"].isin(generos_a_eliminar)]
    df_validation = df_validation[~df_validation["Generos"].isin(generos_a_eliminar)]
    df_test = df_test[~df_test["Generos"].isin(generos_a_eliminar)]

    

    df_train = pd.concat([df_train, df_validation])
    X_train = df_train.drop(columns=["Generos"])
    y_train = df_train["Generos"]
    #X_validation = df_validation.drop(columns=["Generos"])
    #y_validation = df_validation["Generos"]
    X_test = df_test.drop(columns=["Generos"])
    y_test = df_test["Generos"]

    results_table = wandb.Table(columns=["n_estimators", "max_depth", "max_features", "cv_accuracy"])

    for md in config.max_depth:
        for ne in config.n_estimators:
            for mf in config.max_features:
                pipeline = Pipeline([
                    ("preprocess", preprocess),
                    ("svd", TruncatedSVD(n_components=config.svd_components)),
                    ("model", RandomForestClassifier(n_estimators= ne, max_depth = md, criterion=config.criterion, max_features = mf))
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

                print(f"max_depth={md}, n_estimators={ne}, max_features={mf} -> CV accuracy: {acc:.4f}")

                results_table.add_data(ne, md, mf, acc)

                if acc > best_acc:
                    best_acc = acc
                    best_ne = ne
                    best_md = md
                    best_mf = mf

    wandb.log({"cv_results": results_table})

    best_model = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=config.svd_components)),
        ("model", RandomForestClassifier(n_estimators= best_ne, max_depth= best_md, criterion=config.criterion, max_features= best_mf))
    ])

    best_model.fit(X_train, y_train)
    print(f"\nMejor 'n_estimators' encontrado: {best_ne}")
    print(f"\nMejor 'max_depth' encontrado: {best_md}")
    print(f"\nMejor 'max_features' encontrado: {best_mf}")
    print(f"Validation accuracy: {best_acc:.4f}")

    wandb.summary["best_ne"] = best_ne
    wandb.summary["best_md"] = best_md
    wandb.summary["best_mf"] = best_mf
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
    joblib.dump(best_model, "rf_generos_model.joblib")
    wandb.save("rf_generos_model.joblib")

    wandb.finish()

    return best_model

if __name__ == '__main__':
    entramiento_modelo_randomForest_generos()
