from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, f1_score, auc
from comun.preprocess_utils import build_preprocess, unzip_params, build_score
from comun.filter_and_divide_data import get_data_models_train_test, get_data_models_train_test_latest
from collections import defaultdict
import numpy as np
import pandas as pd
import wandb
def run_cross_validation(project_, name_, X_train, y_train, max_features, ngram, svd, preprocess_type, columns, params, modelo, score, average, n_splits=5):
    wandb.init(
        project= project_,
        name= name_,
        config={
            "max_features": max_features,
            "ngram_range": ngram,
            "svd_components": svd,
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
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #n splits 5
    params_ = unzip_params(params=params)
    scores_dict = [] #{k.keys()[0]: [] for k in params_} #Revisad
    i = 0
    for train_idx, val_idx in tqdm(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        preprocess = build_preprocess(preprocess_type, columns, X_tr, max_features, ngram, svd)
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=svd, random_state=42))
        ])

        X_tr_trans = pipe.fit_transform(X_tr, y_tr)
        X_val_trans = pipe.transform(X_val)

        for paramset in params_: #un array
            model = modelo(**paramset)#parameter_name = param_val) #KNeighborsClassifier(n_neighbors=k, metric="cosine")
            model.fit(X_tr_trans, y_tr)
            
            if score.lower() == "auc":
                y_scores = model.predict_proba(X_val_trans)[:, 1]
                precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
                score_val = auc(recalls, precisions)
                print(f"PR-AUC del modelo: {score_val}")
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                best_f1_val = np.max(f1_scores)
                idx_max_f1 = np.argmax(f1_scores)
                if idx_max_f1 < len(thresholds):
                    mejor_umbral = thresholds[idx_max_f1]
                else:
                    mejor_umbral = thresholds[-1]
                print(f"PR-AUC: {score_val:.4f} | Max F1: {best_f1_val:.4f} | Threshold: {mejor_umbral:.4f}")
            else:
                preds = model.predict(X_val_trans)
                score_val = build_score(score, y_val, preds, average)

            scores_dict.append((paramset, score_val))
            i = i+1
            #scores_dict[param_val].append(acc)

    #print(scores_dict)
    #mean_acc_scores = {k: np.mean(v) for k, v in scores_dict.items()}
    mean_acc_scores = []
    scores_grouped = defaultdict(list)
    for paramset, score_val in scores_dict:
        key = tuple(sorted(paramset.items()))
        scores_grouped[key].append(score_val)
    
    for key, values in scores_grouped.items():
        mean_score = np.mean(values)
        paramset = dict(key)
        mean_acc_scores.append((paramset, mean_score))

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
    
def run_best_model(max_features, ngram, svd, preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, paramset, score, average, le):
    preprocess = build_preprocess(preprocess_type, columns, X_train, max_features, ngram, svd)
    best_model = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=svd, random_state=42)),
        ("model", modelo(**paramset))
    ])

    best_model.fit(X_train, y_train)
    if score.lower() == "auc":
        y_probs_test = best_model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs_test)
        
        # Calculamos PR-AUC final
        best_score_test = auc(recalls, precisions)
        
        # Buscamos el umbral que maximiza el F1 en test (o podrías usar el de CV)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        idx_max = np.argmax(f1_scores)
        
        # Nota: thresholds tiene N-1 elementos respecto a precision/recall
        best_threshold = thresholds[idx_max] if idx_max < len(thresholds) else thresholds[-1]
        
        # Aplicamos el umbral para tener las predicciones finales
        raw_preds = (y_probs_test >= best_threshold).astype(int)
        
        print(f"\n--- OPTIMIZACIÓN PR-AUC ---")
        print(f"PR-AUC Test: {best_score_test:.4f}")
        print(f"Umbral óptimo seleccionado: {best_threshold:.4f}")
    else:
        raw_preds = best_model.predict(X_test)
        best_score_test = build_score(score, y_test, raw_preds, average)
    
    print("\n--- RESULTADOS EN TEST ---")
    print("Score:", best_score_test)
    wandb.summary["best_score_test"] = best_score_test
    if score.lower() == "auc": wandb.summary["optimal_threshold"] = best_threshold

    # Matriz de confusión
    class_names = le.classes_.tolist()
    y_test_text = le.inverse_transform(y_test)
    pred_test_text = le.inverse_transform(raw_preds)
    
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
                                probs=None, 
                                y_true=y_test, 
                                preds=raw_preds, 
                                class_names=class_names)})

    print("\nClassification Report:")
    report = classification_report(y_test_text, pred_test_text, output_dict= True)
    print(report)
    wandb.summary["accuracy"] = report["accuracy"]
    wandb.summary["precision_weighted"] = report["weighted avg"]["precision"]
    wandb.summary["recall_weighted"] = report["weighted avg"]["recall"]
    wandb.summary["f1_weighted"] = report["weighted avg"]["f1-score"]

    wandb.summary["precision_macro"] = report["macro avg"]["precision"]
    wandb.summary["recall_macro"] = report["macro avg"]["recall"]
    wandb.summary["f1_macro"] = report["macro avg"]["f1-score"]
   
    wandb.finish()
    return best_model

#He añadido una nueva variable que es filtrado --> Dice si utilizar el dataframe filtrado o sin filtrar
def entrenamiento(project_, name_, modelo, to_predict, max_features, ngram, svd, preprocess_type, columns, params, score, average, n_splits, filtrado = False, include_images=False):
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

    max_features: int
        Número máximo de dimensiones de cada columna textual 

    ngram: tuple (int1, int2) (int1< int 2)
        Rango del ngrama a utilizar (Combinaciones de palabras)

    svd: int
        Numero de dimensiones que quieres tener para entrenar el modelo
    
    preprocess_type: Tipo de preprocesamiento de palabras
        Opciones: "Bag of words", "TF-IDF" and "Word2Vec"

    columns: Columnas del dataframe
        Opciones: "Descripcion", "Subtitulos", "Titulo", etc...
    
    params: Diccionario de parametros a ejecutarse (El valor siempre dentro de corchetes)
        Ejemplo: {"n_neighbors": [3, 4, 5], "metric": ["cosine"]}

    score: string (Informa sobre la medida para elegir el mejor modelo)
        Opciones: "Accuracy", "Precision", "Recall", "F1"

    average: string (Distribución de los valores de score)
        Opciones: "binary" (no funciona con multiclase), "macro", "weighted", "micro"

    n_splits: int
        Numero de pruebas del cross validation

    filtrado: 0, 1 o 2
       0 para no filtrar, 1 para filtrar videos con longitud extrema o sin información textual,
       2 para filtrar videos con subtítulos a None, dejando un poco de estos videos como ruido

    include_images: Boolean
        True: incluye la columna con los embedings de las imagenes
        False: No incluye imagenes

    Returns
    -------
    Nada:
        No devuelve nada
    """ 
    #He modificado esta parte del codigo porque download_and_divide sigue sin estratificar datos o accede a datos filtrados
    print("Starting data acquisition")
    if include_images:
        X_train, X_test, y_train, y_test = get_data_models_train_test_latest(filtrado = 0, to_predict = to_predict, include_images = True)
    else:
        X_train, X_test, y_train, y_test = get_data_models_train_test(filtrado = filtrado, to_predict=to_predict)
    print("Finished data acquisition, starting crossvalidation")

    le = LabelEncoder()
    y_train_encoded = pd.Series(le.fit_transform(y_train))

    #X_train, y_train, X_test, y_test = download_and_divide(to_predict=to_predict)
    
    best_acc, best_param = run_cross_validation(project_, name_, X_train, y_train_encoded, max_features, ngram, svd,
                                                preprocess_type=preprocess_type, 
                                                columns=columns, params=params, 
                                                modelo=modelo, score = score, average = average, 
                                                n_splits=n_splits)
    
    print("Finished crossvalidation, starting evaluating best model")
    #print(best_param)

    #PRUEBAS PARA MATRIZ DE CONFUSION
    # CODIFICA TAMBIÉN EL TEST AQUÍ
    y_test_encoded = pd.Series(le.transform(y_test))
    #FIN PRUEBAS

    best_model = run_best_model(max_features, ngram, svd, preprocess_type, columns, X_train, y_train_encoded, X_test, y_test_encoded, modelo, best_param, score, average, le)
    print("Ready!")
    return best_model
    