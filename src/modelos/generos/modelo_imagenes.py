
from training_utils import entrenamiento
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    to_predict = "Generos"
    #Añadimos la columna de las imagenes
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion", "Titulo_canal"] #, "img_embedding"

    max_features = 3000
    ngram = (1,3)
    svd = 150

    project = "Procesamiento imagenes ()"
    name = "Prueba con imagenes XGboost y DL"
    preprocess_type = "Deep Learning"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"
    """
    params = {
        "max_iter": [10000], 
        "hidden_layer_sizes": [10], 
        "random_state": [42]
    }
    """
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
                    params, score, average, n_fold, filtrado, include_images=False)
    #entrenamiento(project, name, MLPClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
    #            params, score, average, n_fold, filtrado, include_images=False)
