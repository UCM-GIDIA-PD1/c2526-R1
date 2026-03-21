from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics import accuracy_score, classification_report
from preprocess_utils import build_preprocess, unzip_params, build_score
from filter_and_divide_data import get_data_models_train_test
import numpy as np
import pandas as pd
import wandb

def run_cross_validation(project_, name_, X_train, y_train, preprocess_type, columns, params, modelo, score, average, n_splits=5):
    wandb.init(
        project= project_,
        name= name_,
        config={
            "max_features": 5000,
            "ngram_range": (1,2),
            "svd_components": 300,
            "cv_folds": n_splits,
            "params": params, 
            "columns": columns, 
            "preprocess_type": preprocess_type, 
            "modelo": modelo,
            "score": score, 
            "average": average
        }
    )
    best_param = None
    best_acc = 0
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) #n splits 5
    params_ = unzip_params(params=params)
    scores_dict = [] #{k.keys()[0]: [] for k in params_} #Revisad
    i = 0
    for train_idx, val_idx in tqdm(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        preprocess = build_preprocess(preprocess_type, columns, X_tr)
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=300, random_state=42))
        ])

        X_tr_trans = pipe.fit_transform(X_tr, y_tr)
        X_val_trans = pipe.transform(X_val)

        for paramset in params_: #un array
            model = modelo(**paramset)#parameter_name = param_val) #KNeighborsClassifier(n_neighbors=k, metric="cosine")
            model.fit(X_tr_trans, y_tr)

            preds = model.predict(X_val_trans)
            score_val = build_score(score, y_val, preds, average)

            scores_dict.append((paramset, score_val))
            i = i+1
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
        print(f"Parameters set as {k[0]} -> CV score: {k[1]:.4f}")

        if k[1] > best_acc:
            best_acc = k[1]
            best_param = k[0]

    print(f"\nMejor combinación de parámetros encontrada: {best_param}")
    print(f"CrossVal score: {best_acc:.4f}")

    #Creamos tabla
    table = wandb.Table(columns=["params", "cv_score"])

    for param, score in mean_acc_scores:
        table.add_data(str(param), score)

    wandb.log({"cv_results": table})

    wandb.summary["best_score_val"] = best_acc
    wandb.summary["best_params"] = best_param
    return best_acc, best_param
    
def run_best_model(preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, paramset, score, average):
    preprocess = build_preprocess(preprocess_type, columns, X_train)
    best_model = Pipeline([
    ("preprocess", preprocess),
    ("svd", TruncatedSVD(n_components=300, random_state= 42)),
    ("model", modelo(**paramset))
    ])

    best_model.fit(X_train, y_train)

    # Evaluación final con test
    pred_test = best_model.predict(X_test)

    print("\n--- RESULTADOS EN TEST ---")
    best_score_test = build_score(score, y_test, pred_test, average)
    print("Score:", build_score(score, y_test, pred_test, average))
    wandb.summary["best_score_test"] = best_score_test

    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))
    report = classification_report(y_test, pred_test, output_dict= True)
    df_report = pd.DataFrame(report).transpose()
    wandb.log({"Classification_report": wandb.Table(dataframe=df_report)})
    df_preds = pd.DataFrame({
    "y_true": y_test,
    "y_pred": pred_test
    })

    wandb.log({"Predictions": wandb.Table(dataframe=df_preds)})
    wandb.finish()
    return best_model

#He añadido una nueva variable que es filtrado --> Dice si utilizar el dataframe filtrado o sin filtrar
def entrenamiento(project_, name_, modelo, to_predict, preprocess_type, columns, params, score, average, n_splits, filtrado = False):
    """
    Ejecuta un modelo especificado, haz una run en wandb y devuelve por pantalla la mejor selección de parametros

    Parameters
    ----------
    project_: string
        Nombre del projecto
    
    name_: string
        nombre de la run
    
    modelo_: Modelo de skicit learn que se va a utilizar
        Ejemplos: KNeighborsClassifier
    
    to_predict: Columna a predecir
        Opciones: "Generos" o "Made for kids"
    
    preprocess_type: Tipo de preprocesamiento de palabras
        Opciones: "Bag of words", "TF-IDF" and "Word2Vec"

    columns: Columnas del dataframe
        Opciones: "Descripcion", "Subtitulos", "Titulo", etc...
    
    params: Diccionario de parametros a ejecutarse (El valor siempre dentro de corchetes)
        Ejemplo: {"n_neighbors": [3, 4, 5], "metric": ["cosine"]}

    score: string (Informa sobre la medida para elegir el mejor modelo)
        Opciones: "Accuracy", "Precision", "Recall", "F1"

    average: string (Distribución de los valores de score)
        Opciones: "binary" (no funciona con multiclase), "macro", "weighted", "macro", "micro"

    n_splits: int
        Numero de pruebas del cross validation

    filtrado: Bool
        Indica si quieres (True) o no quieres (False), utilzar un dataframe filtrado

    Returns
    -------
    Nada:
        No devuelve nada
    """ 
    #He modificado esta parte del codigo porque download_and_divide sigue sin estratificar datos o accede a datos filtrados
    print("Starting data acquisition")
    X_train, X_test, y_train, y_test = get_data_models_train_test(filtrado = filtrado, to_predict=to_predict)

    print("Finished data acquisition, starting crossvalidation")

    #X_train, y_train, X_test, y_test = download_and_divide(to_predict=to_predict)
    best_acc, best_param = run_cross_validation(project_, name_, X_train, y_train, 
                                                preprocess_type=preprocess_type, 
                                                columns=columns, params=params, 
                                                modelo=modelo, score = score, average = average, 
                                                n_splits=n_splits)

    print("Finished crossvalidation, starting evaluating best model")
    #print(best_param)
    run_best_model(preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, best_param, score, average)
    print("Ready!")