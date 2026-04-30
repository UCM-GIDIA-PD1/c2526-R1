from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento
from Server_PD import upload_model_minio
import numpy as np
import json

if __name__ == '__main__':
    #1. preprocess type, 2. Filtrado, 3. Columnas utilizadas
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Generos", "Duracion", "Subgeneros", "Titulo_canal"]
    
    project = "Modelo Knn Kids"
    name = "V1.1"
    preprocess_type = "Word2Vec"
    max = 5000
    ngram = (1,3)
    svd = 100

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine", "minkowski", "euclidean"
    #weights = ["uniform", "distance"]

    
    params = {"n_neighbors": np.arange(1,15), "metric": ["cosine", "minkowski", "euclidean"],
               "weights":["distance", "uniform"], "n_jobs": [-1]}
    

    score = "Precision"
    average = "weighted"
    n_fold = 2
    filtrado = False

    for i in [("V0.0.2", False), ("V0.1.2", True)]:
        entrenamiento(project, i[0], KNeighborsClassifier, to_predict, max, ngram, svd, preprocess_type, columns, 
                      params, score, average, n_fold, i[1])


    