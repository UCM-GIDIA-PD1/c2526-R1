import optuna
from sklearn.neighbors import KNeighborsClassifier
from optuna_utils import entrenamiento

def objective(trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
        "metric": trial.suggest_categorical("metric", ["cosine", "minkowski", "euclidean"]),
        "weights": trial.suggest_categorical("weights", ["distance", "uniform"]),
        "n_jobs": -1
    }

    score = entrenamiento(
        project_="Modelo Knn Kids Optuna",
        trial_name=f"knn_kids_trial_{trial.number}",
        modelo=KNeighborsClassifier,
        to_predict="Made for kids",
        max_features=5000,
        ngram=(1, 3),
        svd=100,
        preprocess_type="Word2Vec",
        columns=["Titulo", "Descripcion", "Tags", "Subtitulos", "Generos", "Duracion", "Subgeneros", "Titulo_canal"],
        params=params,
        score_metric="Precision",
        average="weighted",
        n_splits=2,
        filtrado=False
    )
    
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    print("Iniciando optimización para KNN (Kids)...")
    study.optimize(objective, n_trials=5)

    print("\n" + "="*50)
    print("MEJOR RESULTADO KNN - MADE FOR KIDS")
    print("="*50)
    print(f"Mejor Recall: {study.best_value:.4f}")
    print("-" * 50)
    print("MEJORES PARÁMETROS:")
    for key, value in study.best_params.items():
        print(f" > {key:20}: {value}")
    print("="*50)
