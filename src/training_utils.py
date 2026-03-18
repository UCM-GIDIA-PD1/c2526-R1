from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics import accuracy_score, classification_report
from preprocess_utils import build_preprocess, unzip_params
from filter_and_divide_data import get_data_models_train_test
import numpy as np
import pandas as pd

def run_cross_validation(X_train, y_train, preprocess_type, columns, params, modelo, n_splits=5):
    best_param = None
    best_acc = 0
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) #n splits 5
    params_ = unzip_params(params=params)
    scores_dict = [] #{k.keys()[0]: [] for k in params_} #Revisad
    for train_idx, val_idx in tqdm(kf.split(X_train)):
        print("Iteracion")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        preprocess = build_preprocess(preprocess_type, columns, X_tr)
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=300))
        ])

        X_tr_trans = pipe.fit_transform(X_tr, y_tr)
        X_val_trans = pipe.transform(X_val)

        for paramset in params_: #un array
            model = modelo(**paramset)#parameter_name = param_val) #KNeighborsClassifier(n_neighbors=k, metric="cosine")
            model.fit(X_tr_trans, y_tr)

            preds = model.predict(X_val_trans)
            acc = accuracy_score(y_val, preds)

            scores_dict.append((paramset, acc))
            #scores_dict[param_val].append(acc)


    #mean_acc_scores = {k: np.mean(v) for k, v in scores_dict.items()}
    mean_acc_scores = []

    for i in range(len(scores_dict) // n_splits):
        total = 0
        paramset = scores_dict[i][0]
        total += scores_dict[i][1]
        for j in range(i+1, len(scores_dict)):
            if np.array_equal(scores_dict[j][0], paramset):
                total += scores_dict[j][1]
        total /= n_splits
        mean_acc_scores.append((paramset, total))


    for k in mean_acc_scores:
        print(f"Parameters set as {k[0]} -> CV accuracy: {k[1]:.4f}")

        if k[1] > best_acc:
            best_acc = k[1]
            best_param = k[0]

    print(f"\nMejor combinación de parámetros encontrada: {best_param}")
    print(f"CrossVal accuracy: {best_acc:.4f}")

    return best_acc, best_param
    
def run_best_model(preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, paramset):
    preprocess = build_preprocess(preprocess_type, columns, X_train)
    best_model = Pipeline([
    ("preprocess", preprocess),
    ("svd", TruncatedSVD(n_components=300)),
    ("model", modelo(**paramset))
    ])

    best_model.fit(X_train, y_train)

    # Evaluación final con test
    pred_test = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    print("Accuracy:", accuracy_score(y_test, pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    return best_model

#He añadido una nueva variable que es filtrado --> Dice si utilizar el dataframe filtrado o sin filtrar
def entrenamiento(modelo, to_predict, preprocess_type, columns, params, n_splits, filtrado = False):
    #He modificado esta parte del codigo porque download_and_divide sigue sin estratificar datos o accede a datos filtrados
    print("Starting data acquisition")
    X_train, X_test, y_train, y_test = get_data_models_train_test(filtrado = filtrado, to_predict=to_predict)

    print("Finished data acquisition, starting crossvalidation")

    #X_train, y_train, X_test, y_test = download_and_divide(to_predict=to_predict)
    best_acc, best_param = run_cross_validation(X_train, y_train, 
                                                preprocess_type=preprocess_type, 
                                                columns=columns, params=params, 
                                                modelo=modelo, n_splits=n_splits)

    print("Finished crossvalidation, starting evaluating best model")
    #print(best_param)
    run_best_model(preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, best_param)
    print("Ready!")