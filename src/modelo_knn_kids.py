from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento
import numpy as np

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for kids", "Duracion"]
    
    project = "Modelo Knn Generos"
    name = "V0"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {"n_neighbors": np.arange(1,20), "metric": ["cosine", "minkowski", "euclidean"],
               "weights":["uniform", "distance"], "n_jobs": [-1]}
    score = "Precision"
    average = "weighted"
    n_fold = 2
    filtrado = True

    entrenamiento(project, name, KNeighborsClassifier, to_predict, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)

