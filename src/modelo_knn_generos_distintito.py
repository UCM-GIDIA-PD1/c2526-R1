from sklearn.neighbors import KNeighborsClassifier
from modelos_utils import download_and_divide, run_cross_validation, run_best_model

def entramiento_modelo_knn_generos():     
    X_train, y_train, X_test, y_test = download_and_divide(to_predict="Generos")
    best_acc, best_param = run_cross_validation(X_train, y_train, preprocess_type="Word2Vec", parameter_name="n_neighbors", parameter_vals=range(3, 4), modelo=KNeighborsClassifier, n_splits=2)
    run_best_model("Word2Vec", X_train, y_train, X_test, y_test, KNeighborsClassifier, "n_neighbors", best_param, "cosine")

if __name__ == '__main__':
    entramiento_modelo_knn_generos()
