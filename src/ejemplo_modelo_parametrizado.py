from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion"]
    
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {"n_neighbours": [3, 4, 5], "metric": "cosine"}

    n_fold = 2

    filtrado = False

    entrenamiento(KNeighborsClassifier, to_predict, preprocess_type, columns, 
                  params, n_fold, filtrado)
