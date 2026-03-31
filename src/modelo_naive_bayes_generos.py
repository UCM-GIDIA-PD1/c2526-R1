from training_utils_naive_bayes import entrenamiento_nb

if __name__ == "__main__":
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos"]

    project = "Modelo Naive Bayes generos"
    name = "V0"
    preprocess_type = "TF-IDF"
    params = {"alpha": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]}
    n_fold = 5
    score = "F1" 
    average = "weighted"
    filtrado = False

    entrenamiento_nb(
        project, name, to_predict, preprocess_type, columns,
        params, score, average, n_fold, filtrado
    )