from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento
from sklearn.metrics import accuracy_score, precision_score


if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for kids", "Duracion"]
    
    project = "Prueba generica"
    name = "V0"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {"n_neighbors": [3, 4, 5], "metric": ["cosine"]}
    n_fold = 2

    filtrado = True

    entrenamiento(project, name, KNeighborsClassifier, to_predict, preprocess_type, columns, 
                  params, n_fold, filtrado)
