from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento



if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion",  "Subgeneros", "Titulo_canal"]
    
    project = "Prueba rf generos"
    name = "V0"

    preprocess_type = "Word2Vec"

    max_features = 5000
    ngram = (1,3)
    svd = 200

    params = { "criterion": ["entropy"], "n_estimators": [100], "max_depth":[15], "max_features":["sqrt", 0.5]}

    n_fold = 5

    filtrado = True

    score = "F1"
    average = "weighted"

    entrenamiento(project, name, RandomForestClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)

