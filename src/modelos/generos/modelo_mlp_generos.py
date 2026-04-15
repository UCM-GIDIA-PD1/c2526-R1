from sklearn.neural_network import MLPClassifier
from comun.training_utils import entrenamiento

# No se puede sacar la matriz de confusión.

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Made for kids", "Duracion", "Titulo_canal"]
    
    max_features = 5000
    ngram = (1,2)
    svd = 150

    project = "Multilayer Perceptron generos"
    name = "W0.T0"
    preprocess_type = "Word2Vec"

    params = {
        "max_iter": [10000], 
        "hidden_layer_sizes": [1, 10, 30], 
        "random_state": [42]
    }
    
    score = "F1"
    average = "weighted"
    n_fold = 5
    filtrado = 0

    entrenamiento(project, name, MLPClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                params, score, average, n_fold, filtrado)
