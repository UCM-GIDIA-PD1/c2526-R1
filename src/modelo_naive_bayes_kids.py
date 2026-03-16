from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, recall_score
import wandb
import joblib
import pandas as pd
import numpy as np
from preprocess_utils import download_model_dfs

def entrenamiento_modelo_naive_bayes_kids():

    version = "v1.0"

    wandb.init(
        entity="pd1-c2526-team1",
        project="clasificacion_kids_nb",
        name="nb_kids_"+version,
        config={
            "tfidf_titulo_max_features": 2000,
            "tfidf_descripcion_max_features": 4000,
            "tfidf_tags_max_features": 2000,
            "tfidf_subtitulos_max_features": 5000,
            "ngram_range": (1,2),
            "svd_components": 300,
            "alpha_values": [0.01, 0.1, 0.5, 1, 2, 5],
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
        "Duracion"
    ]

    # Preprocesamiento
    preprocess = ColumnTransformer(
        transformers=[
            ("Titulo", TfidfVectorizer(max_features=config["tfidf_titulo_max_features"], ngram_range=tuple(config.ngram_range)), "Titulo"),
            ("Descripcion", TfidfVectorizer(max_features=config["tfidf_descripcion_max_features"], ngram_range=tuple(config.ngram_range)), "Descripcion"),
            ("Tags", TfidfVectorizer(max_features=config["tfidf_tags_max_features"], ngram_range=tuple(config.ngram_range)), "Tags"),
            ("Subtitulos", TfidfVectorizer(max_features=config["tfidf_subtitulos_max_features"], ngram_range=tuple(config.ngram_range)), "Subtitulos"),
            ("Generos", OneHotEncoder(), ["Generos"]),
            ("Duracion", "passthrough", ["Duracion"])  # NB no maneja bien features negativas, así que se deja "Duracion" sin escalar
        ]
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=config.svd_components))
    ])

    print("Descargando dataframes...")
    df_train, df_validation, df_test = download_model_dfs()
    df_train = pd.concat([df_train, df_validation])
    print("Dataframes descargados")

    x_train = df_train[feature_cols]
    x_test = df_test[feature_cols]

    # Transformamos Rango_edad en 'kids' vs 'adult'
    y_train = np.where(df_train["Rango_edad"] != "Adult", "kids", "adult")
    y_test = np.where(df_test["Rango_edad"] != "Adult", "kids", "adult")

    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

    best_alpha = None
    best_rec = -1
    scores_dict = {alpha: [] for alpha in config.alpha_values}
    results_table = wandb.Table(columns=['alpha','cv_recall','cv_std'])

    iteration = 1
    for train_idx, val_idx in kf.split(x_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        print(f"Iteración {iteration}: preprocesando y aplicando SVD...")
        x_tr_trans = pipe.fit_transform(x_tr, y_tr)
        x_val_trans = pipe.transform(x_val)
        print("Preprocesamiento completado")

        for alpha in config.alpha_values:
            print(f"Entrenando MultinomialNB con alpha={alpha}")
            model = MultinomialNB(alpha=alpha)
            model.fit(x_tr_trans, y_tr)
            preds = model.predict(x_val_trans)
            rec = recall_score(y_val, preds, pos_label="kids")
            scores_dict[alpha].append(rec)
            print(f"Recall: {rec:.4f}")

        iteration += 1

    # Elegir el mejor alpha
    for alpha, recs in scores_dict.items():
        mean_rec = np.mean(recs)
        std_rec = np.std(recs)
        results_table.add_data(alpha, mean_rec, std_rec)
        print(f"Alpha={alpha} -> CV recall_mean={mean_rec:.4f}, std={std_rec:.4f}")

        if mean_rec > best_rec:
            best_rec = mean_rec
            best_alpha = alpha

    wandb.log({"cv_results": results_table})

    # Entrenar el modelo final con todo el train
    best_model = Pipeline([
        ("preprocess", preprocess),
        ("model", MultinomialNB(alpha=best_alpha))
    ])

    best_model.fit(x_train, y_train)
    print(f"Mejor alpha: {best_alpha}, CV recall: {best_rec:.4f}")

    wandb.summary["best_alpha"] = best_alpha
    wandb.summary["best_cv_recall"] = best_rec

    # Evaluación en test
    pred_test = best_model.predict(x_test)
    test_rec = recall_score(y_test, pred_test, pos_label="kids")
    print("\n--- RESULTADOS EN TEST ---")
    print("Recall:", test_rec)
    print(classification_report(y_test, pred_test))

    wandb.log({"test_recall": test_rec})
    wandb.log({
        f"confusion_matrix_{version}_alpha{best_alpha}": wandb.plot.confusion_matrix(
            y_true=y_test,
            preds=pred_test
        )
    })

    # Guardar modelo
    joblib.dump(best_model, "nb_kids_model.joblib")
    joblib.dump(preprocess, "preprocess_nb.joblib")
    wandb.save("nb_kids_model.joblib")
    wandb.finish()

    return best_model


if __name__ == '__main__':
    entrenamiento_modelo_naive_bayes_kids()