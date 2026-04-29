import optuna
from sklearn.linear_model import LogisticRegression
from comun.optuna_utils import entrenamiento

def objective(trial):
    params = {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 2000, 5000, step=500),
        "n_jobs": -1,
        "random_state": 42
    }

    project_name = "Modelo RegLog Generos Optuna"
    trial_name = f"reglog_trial_{trial.number}"
    
    score = entrenamiento(
        project_=project_name,
        trial_name=trial_name,
        modelo=LogisticRegression,
        to_predict="Generos",
        max_features=5000,
        ngram=(1, 2),
        svd=150,
        preprocess_type="Word2Vec",
        columns=["Titulo", "Descripcion", "Tags", "Made for kids", "Duracion", "Subtitulos","Titulo_canal"],
        params=params,
        score_metric="F1",
        average="weighted",
        n_splits=3, 
        filtrado=2
    )
    
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    study.optimize(objective, n_trials=10)

    print("\n" + "="*50)
    print("RESUMEN DE OPTIMIZACIÓN - REGRESIÓN LOGÍSTICA")
    print("="*50)
    print(f"Mejor F1-Score:  {study.best_value:.4f}")
    print("-"*50)
    print("MEJORES HIPERPARÁMETROS:")
    for key, value in study.best_params.items():
        print(f" > {key:20}: {value}")
    print("="*50)