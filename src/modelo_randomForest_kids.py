from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Duracion", "Subgeneros", "Titulo_canal"]
    
    project = "Prueba rf kids"
    name = "V0"

    preprocess_type = "Word2Vec"

    max_features = 3000
    ngram = (1,2)
    svd = 150

    params = { "criterion": ["gini"], "n_estimators": [30], "max_depth":[30], "max_features":["sqrt"]}

    n_fold = 5

    filtrado = 2

    score = "Precision"
    average = "weighted"

    entrenamiento(project, name, RandomForestClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)
