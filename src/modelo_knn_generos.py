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
    # metric = "cosine"

    params = {"n_neighbors": np.arange(1,15), "metric": ["cosine", "minkowski", "euclidean"],
               "weights":["uniform", "distance"], "n_jobs": [-1]}
    
    score = "Precision"
    average = "weighted"
    n_fold = 2
    filtrado = False
    for i in ["V1.0", "V1.1"]:
        entrenamiento(project, i, KNeighborsClassifier, to_predict, preprocess_type, columns, 
                      params, score, average, n_fold, filtrado)

