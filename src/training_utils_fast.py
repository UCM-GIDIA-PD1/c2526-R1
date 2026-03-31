from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  classification_report, confusion_matrix
from preprocess_utils import  build_score


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
    best_param = {'learning_rate': 0.3, 'max_depth': 10, 'n_estimators': 50}
    best_acc = 0
    return best_acc, best_param
    
def run_best_model(max_features, ngram, svd, preprocess_type, columns, X_train, y_train, X_test, y_test, modelo, paramset, score, average, le):
    
    best_model=None
    raw_preds = pd.read_csv("data/pred_test.csv", header=None)[0].values
    print(raw_preds)

    print("\n--- RESULTADOS EN TEST ---")
    best_score_test = build_score(score, y_test, raw_preds, average)
    print("Score:", best_score_test)
    wandb.summary["best_score_test"] = best_score_test

    # Matriz de confusión
    class_names = le.classes_.tolist()
    y_test_text = le.inverse_transform(y_test)
    pred_test_text = le.inverse_transform(raw_preds)
    cm = confusion_matrix(y_test_text, pred_test_text, labels=class_names)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.insert(0, "Real / Predicho", class_names)
    wandb.log({"matriz_confusion": wandb.Table(dataframe=df_cm)})
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(probs=None, 
                                y_true=y_test, 
                                preds=raw_preds, 
                                class_names=class_names)})

    print("\nClassification Report:")
    report = classification_report(y_test_text, pred_test_text, output_dict= True)
    print(report)
    df_report = pd.DataFrame(report).transpose()
    wandb.log({"Classification_report": wandb.Table(dataframe=df_report)})
    
    df_preds = pd.DataFrame({
    "y_true": y_test_text,
    "y_pred": pred_test_text
    })

    wandb.log({"Predictions": wandb.Table(dataframe=df_preds)})
    wandb.finish()
    return best_model

#He añadido una nueva variable que es filtrado --> Dice si utilizar el dataframe filtrado o sin filtrar
def entrenamiento(project_, name_, modelo, to_predict, max_features, ngram, svd, preprocess_type, columns, params, score, average, n_splits, filtrado = False):
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

    filtrado: Bool
        Indica si quieres (True) o no quieres (False), utilzar un dataframe filtrado

    Returns
    -------
    Nada:
        No devuelve nada
    """ 
    #He modificado esta parte del codigo porque download_and_divide sigue sin estratificar datos o accede a datos filtrados
    print("Starting data acquisition")
    X_train = pd.read_csv("data/X_train.csv", header=None)
    y_train = pd.read_csv("data/y_train.csv", header=None)
    X_test = pd.read_csv("data/X_test.csv", header=None)
    y_test = pd.read_csv("data/y_test.csv", header=None)
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
    le = LabelEncoder()
    y_train_encoded = pd.Series(le.fit_transform(y_train))
    # CODIFICA TAMBIÉN EL TEST AQUÍ
    y_test_encoded = pd.Series(le.transform(y_test))
    #FIN PRUEBAS
    run_best_model(max_features, ngram, svd, preprocess_type, columns, X_train, y_train_encoded, X_test, y_test_encoded, modelo, best_param, score, average, le)
    print("Ready!")