import optuna
from sklearn.neural_network import MLPClassifier
from comun.optuna_utils import entrenamiento

def objective(trial):

    params = {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [
            (10,), (30,), (50,), (20, 10), (50, 25)
        ]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": "adam",
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "learning_rate": "adaptive",
        "max_iter": 2000,
        "random_state": 42
    }

    score = entrenamiento(
        project_="Modelo MLP Kids Optuna",
        trial_name=f"mlp_kids_trial_{trial.number}",
        modelo=MLPClassifier,
        to_predict="Made for kids",
        max_features=5000,
        ngram=(1, 2),
        svd=150,
        preprocess_type="TF-IDF",
        columns=["Titulo", "Descripcion", "Tags", "Subtitulos", "Duracion", "Subgeneros", "Titulo_canal"],
        params=params,
        score_metric="Precision", 
        average="weighted",
        n_splits=2,
        filtrado=0
    )
    
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    print("Iniciando optimización para MLP (Kids)...")
    study.optimize(objective, n_trials=12)

    print("\n" + "="*50)
    print("MEJOR RESULTADO MLP - MADE FOR KIDS")
    print("="*50)
    print(f"Mejor Recall: {study.best_value:.4f}")
    print("-" * 50)
    print("MEJOES PARÁMETROS:")
    for key, value in study.best_params.items():
        print(f" > {key:20}: {value}")
    print("="*50)
