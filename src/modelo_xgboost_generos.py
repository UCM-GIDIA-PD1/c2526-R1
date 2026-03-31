from training_utils import entrenamiento
from xgboost import XGBClassifier

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion", "Titulo_canal"]
    
    max_features = 3000
    ngram = (1,3)
    svd = 150

    project = "Modelo XGB Generos"
    name = "V1.1"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {
        "n_estimators": range(30, 101), 
        "max_depth": [9,10,11],
        'learning_rate':[0.3, 0.5, 0.7]
        }

    score = 'F1'
    average = 'weighted'
    n_fold = 5

    filtrado = True

    entrenamiento(project, name, XGBClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)
