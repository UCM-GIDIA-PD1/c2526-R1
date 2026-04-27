import os
import joblib  # Añadimos joblib para guardar/cargar en local
from joblib import load
import json
import wandb
import pandas as pd
from sklearn.inspection import permutation_importance
from comun.Server_PD import download_model_minio
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_curve, auc
from comun.filter_and_divide_data import extract_definitive_test, get_data_models_train_test_latest
import numpy as np

# --- NUEVA FUNCIÓN PARA GESTIÓN LOCAL ---
def obtener_modelo(bucket, ruta_minio, claves, nombre_local):
    """
    Si el modelo existe en local, lo carga. 
    Si no, lo descarga de MinIO y lo guarda en local.
    """
    if os.path.exists(nombre_local):
        print(f"Cargando {nombre_local} desde local...")
        return joblib.load(nombre_local)
    else:
        print(f"Descargando {ruta_minio} desde MinIO...")
        modelo = download_model_minio(bucket, ruta_minio, claves)
        joblib.dump(modelo, nombre_local)
        print(f"Guardado {nombre_local} en local para futuros usos.")
        return modelo

def evaluar_modelo_final(proyecto, nombre, model, X_test_trans, y_test, encoder = None, X_test=None):
    # (Código de evaluar_modelo_final se mantiene igual...)
    raw_preds = model.predict(X_test_trans)
    prob_preds = model.predict_proba(X_test_trans)

    if encoder:
        class_names = encoder.classes_.tolist()
        y_test_text = encoder.inverse_transform(y_test)
        pred_test_text = encoder.inverse_transform(raw_preds)
    else:
        class_names = ["Adult", "Kids"]
        y_test_text = y_test
        pred_test_text = raw_preds

    analizar_casos_error(X_test, y_test, raw_preds, prob_preds, encoder,)

    report = classification_report(y_test_text, pred_test_text, output_dict=True)
    if(encoder == None): 
        aucScore = auc_score(prob_preds, y_test)
    else: 
        aucScore = None
    
    print(f"Evaluación de {nombre} finalizada. F1-Score: {report['weighted avg']['f1-score']:.4f}")

def calcular_importancia(model, pipe, X_test, y_test):
    # (Código de calcular_importancia se mantiene igual...)
    class FullPipelineWrapper:
        def __init__(self, pipe, model):
            self.pipe = pipe
            self.model = model
        def predict(self, X):
            X_trans = self.pipe.transform(X)
            return self.model.predict(X_trans)
        def score(self, X, y):
            preds = self.predict(X)
            return f1_score(y, preds, average='weighted')
        def fit(self, X, y=None):
            return self

    full_model = FullPipelineWrapper(pipe, model)
    result = permutation_importance(full_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)

    importancia_df = pd.DataFrame({
        'Variable Real': X_test.columns,
        'Importancia Media': result.importances_mean
    }).sort_values(by='Importancia Media', ascending=False)

    print("\n--- IMPORTANCIA DE VARIABLES REALES (GÉNEROS) ---")
    print(importancia_df)
    
   # wandb.init(project="modelo_generos_definitivo", name="Importancia_Variables")
    #tabla = wandb.Table(dataframe=importancia_df)
    #wandb.log({"importancia_variables_reales": wandb.plot.bar(tabla, "Variable Real", "Importancia Media", title="Importancia por Columna Original")})
    #wandb.finish()

def auc_score(y_scores, y_val):
    # (Código de auc_score se mantiene igual...)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
    score_val = auc(recalls, precisions)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_val = np.max(f1_scores)
    idx_max_f1 = np.argmax(f1_scores)
    mejor_umbral = thresholds[idx_max_f1] if idx_max_f1 < len(thresholds) else thresholds[-1]
    print(f"PR-AUC: {score_val:.4f} | Max F1: {best_f1_val:.4f} | Threshold: {mejor_umbral:.4f}")
    return score_val

def analizar_casos_error(X_test, y_test, raw_preds, y_probs, le, n_casos=10):
    if y_probs.ndim == 1:
        confianzas = y_probs
    else:
        confianzas = np.max(y_probs, axis=1)

    posibles_nombres_id = ['video_id', 'id', 'id_video', 'ID', 'Id']
    col_id_real = next((c for c in posibles_nombres_id if c in X_test.columns), None)

    analisis_df = pd.DataFrame({
        'Texto': X_test['Titulo'].values if 'Titulo' in X_test.columns else X_test.iloc[:, 0].values,
        'Realidad': le.inverse_transform(y_test),
        'Prediccion': le.inverse_transform(raw_preds),
        'Confianza': confianzas
    })

    # Si encontramos la columna, la añadimos al dataframe de errores
    if col_id_real:
        analisis_df['ID_Video'] = X_test[col_id_real].values
    else:
        # Si no la encuentra, al menos guardamos el índice por si acaso
        analisis_df['Indice_Original'] = X_test.index

    if y_probs.ndim > 1:
        # Creamos un DataFrame con las probs y nombres de las clases
        probs_df = pd.DataFrame(y_probs, columns=[f"Prob_{c}" for c in le.classes_])
        # Concatenamos horizontalmente
        analisis_df = pd.concat([analisis_df, probs_df], axis=1)
        
    errores = analisis_df[analisis_df['Realidad'] != analisis_df['Prediccion']]
    errores_flagrantes = errores.sort_values(by='Confianza', ascending=False)

    print(f"\n--- ANÁLISIS DE {n_casos} ERRORES MÁS GRAVES ---")
    if not errores_flagrantes.empty:
        print(errores_flagrantes.head(n_casos).to_string(index=False))
        errores_flagrantes.to_csv("analisis_errores.csv", index=False)
    else:
        print("No se encontraron errores.")
        
if __name__ == '__main__':

    # Descarga de modelos (Ahora con lógica local/MinIO)
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)

    # Reemplazamos las llamadas directas por la nueva función
    model_genre = obtener_modelo("pd1", "grupo1/models/genres/genres_definitive", claves, "model_genre.joblib")
    encoder_genre = obtener_modelo("pd1", "grupo1/models/genres/encoder", claves, "encoder_genre.joblib")
    model_kids = obtener_modelo("pd1", "grupo1/models/kids/kids_definitive", claves, "model_kids.joblib")
    pipe_kids = obtener_modelo("pd1", "grupo1/models/kids/pipe_kids", claves, "pipe_kids.joblib")
    pipe_genres = obtener_modelo("pd1", "grupo1/models/genres/pipe_genres", claves, "pipe_genres.joblib")

    # (El resto del código de Kids y Generos se mantiene exactamente igual...)
    #X_test, y_test = extract_definitive_test()
    #X_test_trans_kids = pipe_kids.transform(X_test)
    
    X_test_gen, y_test_gen = extract_definitive_test(columna = "Generos")
    X_test_trans_genre = pipe_genres.transform(X_test_gen)
    y_test_encoded = encoder_genre.transform(y_test_gen)
    
    evaluar_modelo_final("modelo_generos_definitivo", "V0", model_genre, X_test_trans_genre, y_test_encoded, encoder_genre, X_test_gen)