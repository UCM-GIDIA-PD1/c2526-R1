from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion"]
    
    preprocess_type = "Word2Vec"

    params = { "criterion": ["entropy"], "n_estimators": [40], "max_depth":[15], "max_features":[0.5]}

    n_fold = 5

    filtrado = True

    entrenamiento(RandomForestClassifier, to_predict, preprocess_type, columns, 
                  params, n_fold, filtrado)
