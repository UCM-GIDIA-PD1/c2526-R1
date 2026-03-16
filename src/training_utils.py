from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics import accuracy_score, classification_report
from preprocess_utils import build_preprocess, download_and_divide
import numpy as np
import pandas as pd

def run_cross_validation(X_train, y_train, preprocess_type, columns, parameter_name, parameter_vals, modelo, n_splits=5):
    best_param = None
    best_acc = 0
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) #n splits 5

    scores_dict = {k: [] for k in parameter_vals}
    for train_idx, val_idx in tqdm(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        preprocess = build_preprocess(preprocess_type, columns, X_tr)
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=300))
        ])

        X_tr_trans = pipe.fit_transform(X_tr, y_tr)
        X_val_trans = pipe.transform(X_val)

        for param_val in parameter_vals: #un array
            model = modelo(param_val)#parameter_name = param_val) #KNeighborsClassifier(n_neighbors=k, metric="cosine")
            model.fit(X_tr_trans, y_tr)

            preds = model.predict(X_val_trans)
            acc = accuracy_score(y_val, preds)

            scores_dict[param_val].append(acc)

    mean_acc_scores = {k: np.mean(v) for k, v in scores_dict.items()}
    for k in mean_acc_scores.keys():
        print(f"{parameter_name}={k} -> CV accuracy: {mean_acc_scores[k]:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_param = k

    print(f"\nMejor {parameter_name} encontrado: {best_param}")
    print(f"CrossVal accuracy: {best_acc:.4f}")

    return best_acc, best_param
    
def run_best_model(preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, param_name, param_value, metric_val):
    preprocess = build_preprocess(preprocess_type, columns, X_train)
    best_model = Pipeline([
    ("preprocess", preprocess),
    ("svd", TruncatedSVD(n_components=300)),
    ("model", modelo(param_value, metric=metric_val))
    ])

    best_model.fit(X_train, y_train)

    # Evaluación final con test
    pred_test = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    print("Accuracy:", accuracy_score(y_test, pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    return best_model

def entrenamiento(modelo, to_predict, preprocess_type, columns, param_name, param_vals, metric_val, n_splits):
    X_train, y_train, X_test, y_test = download_and_divide(to_predict=to_predict)

    best_acc, best_param = run_cross_validation(X_train, y_train, 
                                                preprocess_type=preprocess_type, 
                                                columns=columns, parameter_name=param_name, 
                                                parameter_vals=param_vals, 
                                                modelo=modelo, n_splits=n_splits)

    run_best_model(preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, param_name, best_param, metric_val)