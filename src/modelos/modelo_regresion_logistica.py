from sklearn.linear_model import LogisticRegression
from training_utils_logistic import entrenamiento
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# No se puede sacar la matriz de confusión.

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for kids", "Duracion", "Subgeneros", "Titulo_canal"]
    
    max_features = 5000
    ngram = (1,2)
    svd = 150

    project = "Regresion logística géneros"
    name = "V1.0.1"
    preprocess_type = "TF-IDF"

    params = {
        "C": [0.1],
        "solver": ["saga"],
        "max_iter": [4000]
    }
    
    score = "Precision"
    average = "weighted"
    n_fold = 5
    filtrado = 0

    entrenamiento(project, name, LogisticRegression, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                params, score, average, n_fold, filtrado)
