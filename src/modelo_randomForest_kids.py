from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion"]
    
    project = "Prueba rf kids"
    name = "V0"

    preprocess_type = "Word2Vec"

    params = { "criterion": ["entropy"], "n_estimators": [1, 2, 3, 4, 5], "max_depth":[10, 15, 20], "max_features":[0.5]}

    n_fold = 5

    filtrado = False

    score = "Precision"
    average = "weighted"

    entrenamiento(project, name, RandomForestClassifier, to_predict, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)
