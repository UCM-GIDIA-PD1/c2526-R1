from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento
import numpy as np

if __name__ == '__main__':
    #1. preprocess type, 2. Filtrado, 3. Columnas utilizadas
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Made for kids", "Duracion", "Titulo_canal"]
    
    project = "Modelo Knn Generos"
    name = "Sin subgeneros"
    preprocess_type = "DeepLearning"
    max = 5000
    ngram = (1,3)
    svd = 100

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine", "minkowski", "euclidean"
    #weights = ["uniform", "distance"]

    params = {"n_neighbors": [6], "metric": ["cosine", "minkowski", "euclidean"],
               "weights":["distance", "uniform"], "n_jobs": [-1]}
    
    score = "F1"
    average = "weighted"
    n_fold = 2
    filtrado = False

    entrenamiento(project, name, KNeighborsClassifier, to_predict, max, ngram, svd, preprocess_type, columns, 
                      params, score, average, n_fold, 2)

