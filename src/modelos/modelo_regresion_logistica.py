from sklearn.linear_model import LogisticRegression
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion"]
    
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [1000],
        "multi_class": "multinomial"
    }

    n_fold = 2

    filtrado = False

    entrenamiento(LogisticRegression, to_predict, preprocess_type, columns, 
                  params, n_fold, filtrado)
