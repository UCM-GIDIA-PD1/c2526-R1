from sklearn.ensemble import RandomForestClassifier
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Rango_edad", "Duracion"]
    
    preprocess_type = "Word2Vec"

    params = { "criterion": ["entropy"], "n_estimators": [10, 20, 30], "max_depth":[10, 15, 20], "max_features":[0.5]}

    n_fold = 5

    filtrado = False

    entrenamiento(RandomForestClassifier, to_predict, preprocess_type, columns, 
                  params, n_fold, filtrado)
