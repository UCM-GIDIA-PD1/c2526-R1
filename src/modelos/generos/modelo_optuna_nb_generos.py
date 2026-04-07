from comun.filter_and_divide_data import get_data_models_train_test
from comun.training_utils_optuna import run_optuna_nb

if __name__ == "__main__":
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos"]
    
    X_train, X_test, y_train, y_test = get_data_models_train_test(
        filtrado=True, 
        to_predict=to_predict
    )

    # Ejecutar Optuna
    best_alpha = run_optuna_nb(
        project_="Modelo Optuna Naive Bayes Generos",
        name="V0",
        X_train=X_train,
        y_train=y_train,
        preprocess_type="TF-IDF",
        columns=columns,
        score_name="F1",
        n_trials=20
    )