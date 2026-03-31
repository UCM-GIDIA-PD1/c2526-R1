from sklearn.neighbors import KNeighborsClassifier
from training_utils import entrenamiento
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Duracion", "Subgeneros", "Titulo_canal"]
    
    max_features = 3000
    ngram = (1,2)
    svd = 150

    project = "Prueba generica"
    name = "V0.0.0"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {"n_neighbors": [2,3], "metric": ["cosine"]}
    score = "Precision"
    average = "weighted"
    n_fold = 2

    # Opciones posibles de filtrado: 0 para no filtrar, 
    # 1 para filtrar videos con longitud extrema o sin información textual 
    # 2 para filtrar videos con subtítulos a None, dejando un poco de estos videos como ruido
    filtrado = 2 
    for i in ["Probar gráficas 2"]:
        entrenamiento(project, i, KNeighborsClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)
