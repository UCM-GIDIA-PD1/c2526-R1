from training_utils import entrenamiento
from xgboost import XGBClassifier

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion"]
    
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {
        "n_estimators": [3, 4, 5], 
        "max_depth": range(1,15),
        'learning_rate':[0.01, 0.1, 0.5, 1.0]
        }

    n_fold = 5

    filtrado = False

    entrenamiento(XGBClassifier, to_predict, preprocess_type, columns, 
                  params, n_fold, filtrado)
