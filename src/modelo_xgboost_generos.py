from training_utils import entrenamiento
from xgboost import XGBClassifier


if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion", "Titulo_canal"]

    max_features = 3000
    ngram = (1,3)
    svd = 150

    project = "Modelo XGB Generos"
    name = "V1.0"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {
        "n_estimators": [3, 4, 5], 
        "max_depth": range(1,15),
        'learning_rate':[0.01, 0.1, 0.5, 1.0]
        }

    score = 'F1'
    average = 'weighted'
    n_fold = 5

    filtrado = True

    entrenamiento(project, name, XGBClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                    params, score, average, n_fold, filtrado)
