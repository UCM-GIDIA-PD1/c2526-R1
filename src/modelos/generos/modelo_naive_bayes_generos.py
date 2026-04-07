from comun.training_utils_naive_bayes import entrenamiento_nb

if __name__ == "__main__":
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos"]

    project = "Modelos definitivos generos"
    name = "Naive Bayes Generos 2"
    preprocess_type = "TF-IDF"
    params = {"alpha": [0.005]}
    n_fold = 2
    score = "F1" 
    average = "weighted"
    filtrado = 2

    entrenamiento_nb(
        project, name, to_predict, preprocess_type, columns,
        params, score, average, n_fold, filtrado
    )