from sklearn.neural_network import MLPClassifier
from optuna_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Duracion", "Subgeneros", "Titulo_canal"]
    
    max_features = 5000
    ngram = (1,2)
    svd = 150

    project = "Multilayer Perceptron kids"
    name = "W0.T0"
    preprocess_type = "TF-IDF"

    params = {
        "max_iter": [10000], 
        "hidden_layer_sizes": [1, 5, 10], 
        "random_state": [42]
    }
    
    score = "Precision"
    average = "weighted"
    n_fold = 5
    filtrado = 0

    entrenamiento(project, name, MLPClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                params, score, average, n_fold, filtrado)
