from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Duracion", "Subgeneros", "Titulo_canal"]
    
    project = "Prueba matriz confusion "
    name = "V0"

    preprocess_type = "TF-IDF"

    max_features = 3000
    ngram = (1,2)
    svd = 150

    params = { "criterion": ["gini"], "n_estimators": [3], "max_depth":[3], "max_features":["sqrt"]}

    n_fold = 5

    filtrado = False

    score = "Precision"
    average = "weighted"

    entrenamiento(project, name, RandomForestClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)
