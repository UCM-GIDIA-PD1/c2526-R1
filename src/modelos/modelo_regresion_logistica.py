from sklearn.linear_model import LogisticRegression
from training_utils import entrenamiento
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for kids", "Duracion", "Subgeneros", "Titulo_canal"]
    
    max_features = 5000
    ngram = (1,2)
    svd = 150

    project = "Regresion logística géneros"
    name = "V0.0.1"
    preprocess_type = "Word2Vec"

    params = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["saga", "lbfgs"],
        "max_iter": [1000, 2000, 3000]
    }
    
    score = "Precision"
    average = "macro"
    n_fold = 5
    filtrado = False

    entrenamiento(project, name, LogisticRegression, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                params, score, average, n_fold, filtrado)
