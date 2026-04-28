import optuna
from sklearn.ensemble import RandomForestClassifier
from comun.optuna_utils import entrenamiento

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 10, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "n_jobs": -1,
        "random_state": 42
    }

    project_name = "Modelo randomForest Generos Optuna"
    trial_name = f"rf_trial_{trial.number}"
    
    score = entrenamiento(
        project_=project_name,
        trial_name=trial_name,
        modelo=RandomForestClassifier,
        to_predict="Generos",
        max_features=3000,
        ngram=(1, 2),
        svd=150,
        preprocess_type="Word2Vec",
        columns=["Titulo", "Descripcion", "Tags", "Made for kids","Duracion", "Subgeneros", "Titulo_canal"],
        params=params,
        score_metric="F1",
        average="weighted",
        n_splits=2,
        filtrado=2
    )
    
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    study.optimize(objective, n_trials=10)

    print("\n" + "="*50)
    print("RESUMEN DE OPTIMIZACIÓN - RANDOM FOREST")
    print("="*50)
    print(f"Mejor F1-Score:  {study.best_value:.4f}")
    print("-"*50)
    print("MEJORES HIPERPARÁMETROS:")
    for key, value in study.best_params.items():
        print(f" > {key:20}: {value}")
    print("="*50)