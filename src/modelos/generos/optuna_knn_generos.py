import optuna
from sklearn.neighbors import KNeighborsClassifier
from comun.optuna_utils import entrenamiento_para_optuna

def objective(trial):
    # Espacio de búsqueda de hiperparámetros para KNN
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["cosine", "minkowski", "euclidean"]),
        "n_jobs": -1
    }

    score = entrenamiento_para_optuna(
        project_="Modelo Knn Generos Optuna",
        trial_name=f"knn_trial_{trial.number}",
        modelo=KNeighborsClassifier,
        to_predict="Generos",
        max_features=5000,
        ngram=(1, 3),
        svd=100,
        preprocess_type="Word2Vec", # Cambiar a "TF-IDF" si Word2Vec tarda demasiado
        columns=["Titulo", "Descripcion", "Tags", "Made for kids", "Duracion", "Titulo_canal"],
        params=params,
        score_metric="F1",
        average="weighted",
        n_splits=2, 
        filtrado=2
    )
    
    return score

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=20)

    # 3. Resultados finales
    print("\n" + "=" * 30)
    print("MEJOR RESULTADO ENCONTRADO")
    print(f"F1-Score: {study.best_value:.4f}")
    print(f"Parámetros: {study.best_params}")
    print("=" * 30)