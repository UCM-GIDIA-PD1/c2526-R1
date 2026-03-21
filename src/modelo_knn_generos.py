from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento
import numpy as np

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for kids", "Duracion"]
    
    project = "Modelo Knn Generos"
    name = "V1.1"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine", "minkowski", "euclidean"
    #weights = ["uniform", "distance"]

    params = {"n_neighbors": np.arange(1,15), "metric": ["cosine"],
               "weights":["distance"], "n_jobs": [-1]}
    
    score = "F1"
    average = "weighted"
    n_fold = 10
    filtrado = False
    for i in ["V1.2", "V1.3", "V1.4", "V1.5", "V1.6", "V1.7", "V1.8", "V1.9", "V1.10"]:
        entrenamiento(project, i, KNeighborsClassifier, to_predict, preprocess_type, columns, 
                      params, score, average, n_fold, filtrado)

