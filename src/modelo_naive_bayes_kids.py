from training_utils_naive_bayes import entrenamiento_nb

if __name__ == "__main__":
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos"] #Se elimina "Generos" para evitar data leakage.

    project = "Modelos definitivos kids"
    name = "Naive Bayes Kids 2"
    preprocess_type = "TF-IDF"
    params = {"alpha": [0.1]}
    n_fold = 2
    ngram = (1,2)
    score = "Precision"
    average = "weighted"
    filtrado = 2

    entrenamiento_nb(project, name, to_predict, preprocess_type, columns, params, score, average, 
                     n_fold, filtrado)