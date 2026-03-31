from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from preprocess_utils import build_preprocess, unzip_params, build_score
from filter_and_divide_data import get_data_models_train_test
import pandas as pd
import wandb
from collections import defaultdict
import numpy as np

def run_cross_validation_nb(project_, name_, X_train, y_train, preprocess_type, columns, params, score, average, n_splits=5):
    wandb.init(
        project=project_,
        name=name_,
        config={
            "params": params,
            "columns": columns,
            "preprocess_type": preprocess_type,
            "modelo": "MultinomialNB",
            "score": score,
            "average": average,
            "cv_folds": n_splits
        }
    )

    best_alpha = None
    best_score_val = 0

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    params_ = unzip_params(params)
    scores_dict = []

    # Cross-validation
    for train_idx, val_idx in tqdm(kf.split(X_train, y_train), desc="CV Splits NB"):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        preprocess = build_preprocess(preprocess_type, columns, X_tr, 5000, (1,2), None)
        pipe = Pipeline([("preprocess", preprocess)])  # NB sin SVD

        X_tr_trans = pipe.fit_transform(X_tr, y_tr)
        X_val_trans = pipe.transform(X_val)

        for paramset in params_:
            model = MultinomialNB(**paramset)
            model.fit(X_tr_trans, y_tr)

            preds = model.predict(X_val_trans)
            score_val = build_score(score, y_val, preds, average)

            scores_dict.append((paramset, score_val))


    mean_scores = []
    scores_grouped = defaultdict(list)

    for paramset, score_val in scores_dict:
        key = tuple(sorted(paramset.items()))
        scores_grouped[key].append(score_val)

    for key, values in scores_grouped.items():
        avg_score = np.mean(values)
        paramset = dict(key)

        mean_scores.append((paramset, avg_score))

    for paramset, avg_score in mean_scores:
        print(f"{paramset} -> {avg_score:.4f}")

        if avg_score > best_score_val:
            best_score_val = avg_score
            best_alpha = paramset.get("alpha", None)

    # Log en wandb
    table = wandb.Table(columns=["params", "cv_score"])

    for paramset, avg_score in mean_scores:
        table.add_data(str(paramset), avg_score)

    wandb.log({"cv_results": table})

    wandb.summary["best_score_val"] = best_score_val
    wandb.summary["best_alpha"] = best_alpha

    print(f"\nBest alpha: {best_alpha}, CrossVal score: {best_score_val:.4f}")
    return best_alpha


def run_best_model_nb(preprocess_type, columns, X_train, y_train, X_test, y_test, alpha, score, average):
    preprocess = build_preprocess(preprocess_type, columns, X_train, 5000, (1,2), None)

    best_model = Pipeline([
        ("preprocess", preprocess),
        ("model", MultinomialNB(alpha=alpha))
    ])

    best_model.fit(X_train, y_train)

    pred_test = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    best_score_test = build_score(score, y_test, pred_test, average)
    print(f"Score: {best_score_test:.4f}")

    wandb.summary["best_score_test"] = best_score_test

    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    df_report = pd.DataFrame(classification_report(y_test, pred_test, output_dict=True)).transpose()
    wandb.log({"Classification_report": wandb.Table(dataframe=df_report)})

    df_preds = pd.DataFrame({
        "y_true": y_test,
        "y_pred": pred_test
    })

    wandb.log({"Predictions": wandb.Table(dataframe=df_preds)})

    wandb.finish()
    return best_model

def entrenamiento_nb(project_, name_, to_predict, preprocess_type, columns, params, score, average, n_splits=5, filtrado=False):

    print("Starting data acquisition")
    X_train, X_test, y_train, y_test = get_data_models_train_test(
        filtrado=filtrado, to_predict=to_predict
    )

    print("Finished data acquisition, starting cross-validation")

    best_alpha = run_cross_validation_nb(
        project_, name_, X_train, y_train,
        preprocess_type, columns, params,
        score, average, n_splits
    )

    print("Finished cross-validation, starting evaluating best model")

    run_best_model_nb(
        preprocess_type, columns,
        X_train, y_train,
        X_test, y_test,
        best_alpha, score, average
    )

    print("Ready!")