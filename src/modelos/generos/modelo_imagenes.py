
from comun.training_utils import entrenamiento
from xgboost import XGBClassifier


if __name__ == '__main__':
    to_predict = "Generos"
    #Añadimos la columna de las imagenes
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion", "Titulo_canal", "img_embedding"]

    max_features = 3000
    ngram = (1,3)
    svd = 150

    project = "Procesamiento imagenes ()"
    name = "V0"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {
        "n_estimators": [190], 
        "max_depth": [10],
        'learning_rate':[0.3]
        }

    score = 'F1'
    average = 'weighted'
    n_fold = 5

    filtrado = True

    entrenamiento(project, name, XGBClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                    params, score, average, n_fold, filtrado, include_images=True)
