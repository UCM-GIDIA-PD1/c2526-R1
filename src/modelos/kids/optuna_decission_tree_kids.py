import optuna
import wandb
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from comun.preprocess_utils import download_model_dfs


print('Cargando dataframes para clasificación Kids...')
df_train, df_validation, df_test = download_model_dfs()
df_train_full = pd.concat([df_train, df_validation])

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    run = wandb.init(
        entity="pd1-c2526-team1",
        project="Modelo DT Kids Optuna",
        name=f"trial_dt_{trial.number}",
        reinit=True,
        config=params
    )

    feature_cols = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Generos", "Duracion"]
    
    preprocess = ColumnTransformer(
        transformers=[
            ("Titulo", TfidfVectorizer(max_features=2000, ngram_range=(1,2)), "Titulo"),
            ("Descripcion", TfidfVectorizer(max_features=4000, ngram_range=(1,2)), "Descripcion"),
            ("Tags", TfidfVectorizer(max_features=2000, ngram_range=(1,2)), "Tags"),
            ("Subtitulos", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), "Subtitulos"),
            ("Generos", OneHotEncoder(handle_unknown='ignore'), ["Generos"]),
            ("Duracion", StandardScaler(), ["Duracion"]),
            ("Subgeneros", OneHotEncoder(handle_unknown='ignore'), ["Subgeneros"]),
            ("Titulo_canal", TfidfVectorizer(max_features=2000, ngram_range=(1,2)), "Titulo_canal")
        ]
    )

    X = df_train_full[feature_cols]
    y = df_train_full["Rango_edad"] != 'Adult'

    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    recalls = []

    # 4. CROSS VALIDATION
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Pipeline de entrenamiento
        model_pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=300)),
            ("model", DecisionTreeClassifier(**params, random_state=42))
        ])

        model_pipe.fit(X_tr, y_tr)
        preds = model_pipe.predict(X_val)
        
        rec = recall_score(y_val, preds)
        recalls.append(rec)

    mean_recall = np.mean(recalls)
    wandb.log({"mean_cv_recall": mean_recall})
    run.finish()

    return mean_recall

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\n" + "="*50)
    print("MEJOR RESULTADO PARA KIDS - DECISION TREE")
    print(f"Mejor Recall: {study.best_value:.4f}")
    print(f"Parámetros: {study.best_params}")
    print("="*50)