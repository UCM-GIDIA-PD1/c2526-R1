from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import wandb
from preprocess_utils import build_preprocess, build_score
from filter_and_divide_data import get_data_models_train_test

def entrenamiento(project_, trial_name, modelo, to_predict, max_features, ngram, svd, preprocess_type, columns, params, score_metric, average, n_splits, filtrado):
    print("Starting data acquisition")
    X_train_full, _, y_train_full, _ = get_data_models_train_test(filtrado=filtrado, to_predict=to_predict)
    
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y_train_full))

    run = wandb.init(
        project=project_,
        name=trial_name,
        config={**params, "preprocess": preprocess_type, "svd": svd},
        reinit=True,
        group="Optuna_KNN_Search"
    )

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_train_full, y_encoded):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

        preprocess = build_preprocess(preprocess_type, columns, X_tr, max_features, ngram, svd)
        
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=svd, random_state=42)),
            ("model", modelo(**params))
        ])

        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        
        score_val = build_score(score_metric, y_val, preds, average)
        cv_scores.append(score_val)

    mean_score = np.mean(cv_scores)
    wandb.log({f"mean_cv_{score_metric}": mean_score})
    run.finish()
    
    return mean_score