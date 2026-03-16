from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, recall_score
import wandb
import joblib
import pandas as pd
from modelos_utils import download_model_dfs


def entrenamiento_modelo_naive_bayes_kids():

    wandb.init(
        entity="pd1-c2526-team1",
        project="clasificacion_kids_nb",
        name="nb_kids_v0",
        config={
            "tfidf_titulo_max_features": 2000,
            "tfidf_descripcion_max_features": 4000,
            "tfidf_tags_max_features": 2000,
            "tfidf_subtitulos_max_features": 5000,
            "ngram_range": (1,2),
            "svd_components": 300,
            "alpha_values": [0.01, 0.1, 0.5, 1, 2, 5],
            "cv_folds": 5
        }
    )

    config = wandb.config

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

    best_alpha = None
    best_rec = 0
    best_model = None

    df_train, df_validation, df_test = download_model_dfs()

    # unir train y validation
    df_train = pd.concat([df_train, df_validation])

    x_train = df_train.drop(columns=["Rango_edad"])
    y_train = ['kids' if i!='Adult' else i for i in df_train["Rango_edad"]]

    x_test = df_test.drop(columns=["Rango_edad"])
    y_test = ['kids' if i!='Adult' else i for i in df_test["Rango_edad"]]

    results_table = wandb.Table(columns=['alpha','cv_recall', 'cv_std'])

    for alpha in config.alpha_values:

        print(f"Entrenando modelo con alpha: {alpha}")

        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=config["svd_components"])),
            ("model", MultinomialNB(alpha=alpha))
        ])

        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=config["cv_folds"],
            scoring="recall",
            n_jobs=-1
        )

        rec_mean = scores.mean()
        rec_std = scores.std()

        print(f"Alpha = {alpha} -> CV recall_mean: {rec_mean:.4f}")
        print(f"            -> CV recall_std: {rec_std:.4f}")

        results_table.add_data(alpha, rec_mean, rec_std)

        if rec_mean > best_rec:
            best_rec = rec_mean
            best_alpha = alpha

    wandb.log({"cv_results": results_table})

    best_model = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=config["svd_components"])),
        ("model", MultinomialNB(alpha=best_alpha))
    ])

    best_model.fit(x_train, y_train)

    print(f"\nMejor alpha encontrado: {best_alpha}")
    print(f"Validation recall: {best_rec:.4f}")

    wandb.summary["best_alpha"] = best_alpha
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

    wandb.sklearn.plot_confusion_matrix(y_test, pred_test)

    joblib.dump(best_model, "nb_kids_model.joblib")
    wandb.save("nb_kids_model.joblib")

    wandb.finish()

    return best_model


if __name__ == '__main__':
    entrenamiento_modelo_naive_bayes_kids()