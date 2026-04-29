import optuna
from sklearn.ensemble import RandomForestClassifier
from optuna_utils import entrenamiento

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
        "max_depth": trial.suggest_int("max_depth", 10, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "n_jobs": -1,
        "random_state": 42
    }

    score = entrenamiento(
        project_="Prueba rf kids Optuna",
        trial_name=f"rf_kids_trial_{trial.number}",
        modelo=RandomForestClassifier,
        to_predict="Made for kids",
        max_features=3000,
        ngram=(1, 2),
        svd=150,
        preprocess_type="Word2Vec",
        columns=["Titulo", "Descripcion", "Tags", "Subtitulos", "Duracion", "Subgeneros", "Titulo_canal"],
        params=params,
        score_metric="Precision", 
        average="weighted",
        n_splits=2,
        filtrado=2
    )
    
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    print("Iniciando optimización para Random Forest (Kids)...")
    study.optimize(objective, n_trials=10)

    print("\n" + "="*50)
    print("MEJOR RESULTADO RANDOM FOREST - MADE FOR KIDS")
    print("="*50)
    print(f"Mejor Recall: {study.best_value:.4f}")
    print("-" * 50)
    print("MEJORES PARÁMETROS:")
    for key, value in study.best_params.items():
        print(f" > {key:20}: {value}")
    print("="*50)