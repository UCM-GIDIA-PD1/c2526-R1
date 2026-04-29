from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, recall_score, make_scorer
import wandb
import joblib
import pandas as pd
from preprocess_utils import download_model_dfs
import numpy as np


def entramiento_modelo_decission_tree_kids():

    version = "v1.0"#Cambiar en cada ejecución, hay que verlo

    wandb.init(
        entity="pd1-c2526-team1",
        project="clasificacion_kids_dt",
        name="dt_kids_"+version,
        config={
            "tfidf_titulo_max_features": 2000,
            "tfidf_descripcion_max_features": 4000,
            "tfidf_tags_max_features": 2000,
            "tfidf_subtitulos_max_features": 5000,
            "ngram_range": (1,2),
            "svd_components": 300,
            "depth_values": range(7,10),
            'criterion': 'gini',
            "cv_folds": 2
        }
    )

    config = wandb.config

    feature_cols = [
    "Titulo",
    "Descripcion",
    "Tags",
    "Subtitulos",
    "Generos",
    "Duracion"]

    preprocess = ColumnTransformer(
        transformers=[
            ("Titulo", TfidfVectorizer(max_features=config["tfidf_titulo_max_features"], ngram_range=tuple(config.ngram_range)), "Titulo"),
            ("Descripcion", TfidfVectorizer(max_features=config["tfidf_descripcion_max_features"], ngram_range=tuple(config.ngram_range)), "Descripcion"),
            ("Tags", TfidfVectorizer(max_features=config["tfidf_tags_max_features"], ngram_range=tuple(config.ngram_range)), "Tags"),
            ("Subtitulos", TfidfVectorizer(max_features=config["tfidf_subtitulos_max_features"], ngram_range=tuple(config.ngram_range)), "Subtitulos"),
            ("Generos", OneHotEncoder(), ["Generos"]),
            ("Duracion", StandardScaler(), ["Duracion"])
        ]
    )
    pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=config.svd_components))
        ])
    
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

    best_depth = None
    best_rec = -1
    best_model = None

    print('Downloading dataframes')
    df_train, df_validation, df_test = download_model_dfs()
    df_train = pd.concat([df_train, df_validation])
    print('Dataframes downloaded')

    x_train = df_train.drop(columns=["Rango_edad"])[feature_cols]
    x_test = df_test.drop(columns=["Rango_edad"])[feature_cols]

    y_train = df_train["Rango_edad"] != 'Adult'
    y_test = df_test["Rango_edad"] != 'Adult'

    results_table = wandb.Table(columns=['depth','cv_recall', 'cv_std'])
    scores_dict = {k: [] for k in config.depth_values}

    iteration = 1
    for train_idx, val_idx in kf.split(x_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_train_final, y_val_final = y_train.iloc[train_idx], y_train.iloc[val_idx]

        print(f'Starting preprocess and svd to data frame, iteration: {iteration}')
        x_train_final = pipe.fit_transform(x_tr, y_train_final)
        x_val_final = pipe.transform(x_val)
        print('Preprocess completed')

        for i in config.depth_values:
            print(f"{iteration}: Comenzando entrenamiento para el modelo con profundidad: {i}")
            model = DecisionTreeClassifier(
                max_depth=i, 
                criterion=config.criterion, 
                random_state=42
            )
            model.fit(x_train_final, y_train_final)

            preds = model.predict(x_val_final)
            rec = recall_score(y_val_final, preds)
            print(f'Recall: {rec}')

            scores_dict[i].append(rec)

        iteration += 1

    for depth, recalls in scores_dict.items():
        rec_mean = np.mean(recalls)
        rec_std = np.std(recalls)
        print(f"Depth = {depth} -> CV recall_mean: {rec_mean:.4f}")
        print(f"          -> CV recall_std: {rec_std:.4f}")

        results_table.add_data(depth, rec_mean, rec_std)

        if rec_mean > best_rec:
            best_rec = rec_mean
            best_depth = depth
    
    wandb.log({"cv_results":results_table})

    best_model = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=300)),
        ("model",DecisionTreeClassifier(max_depth=best_depth, criterion=config.criterion))
    ])

    best_model.fit(x_train, y_train)
    
    print(f"\nMejor depth encontrada: {best_depth}")
    print(f"Validation recall: {best_rec:.4f}")

    wandb.summary["best_depth"] = best_depth
    wandb.summary["best_cv_recall"] = best_rec

    pred_test = best_model.predict(x_test)
    test_rec = recall_score(y_test, pred_test)

    print("\n--- RESULTADOS EN TEST ---")
    print("Recall:", test_rec)
    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    wandb.log({
        "test_recall": test_rec
    })

    # Matriz de confusión fea
    wandb.log({
        f"confusion_matrix_{version}_depth{best_depth}": wandb.plot.confusion_matrix(
        y_true=y_test,
        preds=pred_test)
        })

    # Guardar modelo
    joblib.dump(best_model, "dt_kids_model.joblib")
    joblib.dump(preprocess, "preprocess.joblib")
    wandb.save("dt_kids_model.joblib")

    wandb.finish()

    return best_model

if __name__ == '__main__':
    entramiento_modelo_decission_tree_kids()