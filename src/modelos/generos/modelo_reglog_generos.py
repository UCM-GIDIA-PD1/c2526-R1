from sklearn.linear_model import LogisticRegression
from comun.training_utils import entrenamiento
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for kids", "Duracion", "Subgeneros", "Titulo_canal"]
    
    max_features = 5000
    ngram = (1,2)
    svd = 150

    project = "Regresion logística géneros"
    name = "W1.w0"
    preprocess_type = "Word2Vec" # mejor usar Word2Vec

    params = {
        "C": [0.01, 0.1, 1, 10], # ideal: 0.1
        "solver": ["saga", "lbfgs"], # ideal: saga
        "max_iter": [3000, 4000] # usamos 3000, pero no parece afectar entre 3000 y 5000
    }
    
    score = "F1"
    average = "weighted"
    n_fold = 5
    filtrado = 2

    entrenamiento(project, name, LogisticRegression, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                params, score, average, n_fold, filtrado)
