from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, recall_score
import wandb
import joblib
import pandas as pd
import numpy as np

from comun.preprocess_utils import download_model_dfs_filtered


def entrenamiento_modelo_naive_bayes_kids():

    version = "v1.0"

    wandb.init(
        entity="pd1-c2526-team1",
        project="clasificacion_kids_nb",
        name="nb_kids_" + version,
        config={
            "tfidf_titulo_max_features": 2000,
            "tfidf_descripcion_max_features": 4000,
            "tfidf_tags_max_features": 2000,
            "tfidf_subtitulos_max_features": 5000,
            "ngram_range": (1,2),
            "alpha_values": [0.1, 0.5, 1.0],
            "cv_folds": 3
        }
    )

    config = wandb.config

    feature_cols = [
        "Titulo",
        "Descripcion",
        "Tags",
        "Subtitulos",
        "Generos"
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("Titulo",
             TfidfVectorizer(
                 max_features=config.tfidf_titulo_max_features,
                 ngram_range=tuple(config.ngram_range), min_df=3, #ignora palabras que aparecen en menos de 3 archivos
                 max_df=0.9, #ignora palabras que aparecen en más del 90% de los archivos
             ),
             "Titulo"),

            ("Descripcion",
             TfidfVectorizer(
                 max_features=config.tfidf_descripcion_max_features,
                 ngram_range=tuple(config.ngram_range), min_df=3),
             "Descripcion"),

            ("Tags",
             TfidfVectorizer(
                 max_features=config.tfidf_tags_max_features,
                 ngram_range=tuple(config.ngram_range), min_df=3, max_df=0.9),
             "Tags"),

            ("Subtitulos",
             TfidfVectorizer(
                 max_features=config.tfidf_subtitulos_max_features,
                 ngram_range=tuple(config.ngram_range), min_df=3),
             "Subtitulos"),

            ("Generos", OneHotEncoder(handle_unknown="ignore"), ["Generos"])
        ]
    )

    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

    best_alpha = None
    best_rec = -1

    print("Downloading dataframes")
    df_train, df_validation, df_test = download_model_dfs_filtered()
    df_train = pd.concat([df_train, df_validation])
    print("Dataframes downloaded")

    x_train = df_train[feature_cols]
    x_test = df_test[feature_cols]

    y_train = df_train["Rango_edad"] != 'Adult'
    y_test = df_test["Rango_edad"] != 'Adult'

    results_table = wandb.Table(columns=["alpha", "cv_recall", "cv_std"])

    scores_dict = {a: [] for a in config.alpha_values}

    iteration = 1

    for train_idx, val_idx in kf.split(x_train):

        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        print(f"Iteration {iteration}: preprocessing")

        x_train_final = preprocess.fit_transform(x_tr, y_tr)
        x_val_final = preprocess.transform(x_val)

        for alpha in config.alpha_values:

            print(f"{iteration}: training Naive Bayes alpha={alpha}")

            model = MultinomialNB(alpha=alpha)

            model.fit(x_train_final, y_tr)

            preds = model.predict(x_val_final)

            rec = recall_score(y_val, preds)

            print(f"Recall: {rec}")

            scores_dict[alpha].append(rec)

        iteration += 1

    for alpha, recalls in scores_dict.items():

        rec_mean = np.mean(recalls)
        rec_std = np.std(recalls)

        print(f"Alpha {alpha} -> CV recall_mean: {rec_mean:.4f}")
        print(f"          -> CV recall_std: {rec_std:.4f}")

        results_table.add_data(alpha, rec_mean, rec_std)

        if rec_mean > best_rec:
            best_rec = rec_mean
            best_alpha = alpha

    wandb.log({"cv_results": results_table})

    best_model = Pipeline([
        ("preprocess", preprocess),
        ("model", MultinomialNB(alpha=best_alpha))
    ])

    best_model.fit(x_train, y_train)

    print(f"\nBest alpha: {best_alpha}")
    print(f"Validation recall: {best_rec:.4f}")

    wandb.summary["best_alpha"] = best_alpha
    wandb.summary["best_cv_recall"] = best_rec

    pred_test = best_model.predict(x_test)

    test_rec = recall_score(y_test, pred_test)

    print("\nResultados en test:")
    print("Recall:", test_rec)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, pred_test))

    wandb.log({
        "test_recall": test_rec
    })

    wandb.log({
        f"confusion_matrix_{version}_alpha{best_alpha}": wandb.plot.confusion_matrix(
            y_true=y_test,
            preds=pred_test
        )
    })

    joblib.dump(best_model, "nb_kids_model.joblib")

    wandb.save("nb_kids_model.joblib")

    wandb.finish()

    return best_model


if __name__ == "__main__":
    entrenamiento_modelo_naive_bayes_kids()