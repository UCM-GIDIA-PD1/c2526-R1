from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion"]
    
    project = "Prueba rf generos"
    name = "V0"

    preprocess_type = "TF-IDF"

    params = { "criterion": ["entropy"], "n_estimators": [40], "max_depth":[15,20], "max_features":[0.8]}

    n_fold = 5

    filtrado = True

    score = "Precision"
    average = "weighted"

    entrenamiento(project, name, RandomForestClassifier, to_predict, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)
