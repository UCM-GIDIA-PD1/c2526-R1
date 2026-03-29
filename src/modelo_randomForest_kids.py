from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion", "Subgeneros", "Titulo_canal"]
    
    project = "Prueba matriz confusion "
    name = "V0"

    preprocess_type = "TF-IDF"

    max_features = 3000
    ngram = (1,2)
    svd = 150

    params = { "criterion": ["entropy"], "n_estimators": [5], "max_depth":[15], "max_features":["sqrt"]}

    n_fold = 2

    filtrado = False

    score = "Precision"
    average = "weighted"

    entrenamiento(project, name, RandomForestClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)
