from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento
import numpy as np

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for kids", "Duracion"]
    
    project = "Modelo Knn Generos"
    name = "V1.1"
    preprocess_type = "Bag of words"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine", "minkowski", "euclidean"
    #weights = ["uniform", "distance"]

    params = {"n_neighbors": np.arange(1,15), "metric": ["cosine", "minkowski", "euclidean"],
               "weights":["distance", "uniform"], "n_jobs": [-1]}
    
    score = "F1"
    average = "weighted"
    n_fold = 5
    filtrado = False
    for i in [("V3.1", False), ("V3.2", True)]:
        entrenamiento(project, i[0], KNeighborsClassifier, to_predict, preprocess_type, columns, 
                      params, score, average, n_fold, i[1])

