import pandas as pd
import torch
import wandb
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess_utils import download_model_dfs

def clasificar_subtitulo(texto, classifier, categorias):
    res = classifier(str(texto)[:800], candidate_labels=categorias)
    mejor_clase = res['labels'][0]
    confianza = res['scores'][0]
    es_kids_llm = (mejor_clase == "Made for Kids")
    return mejor_clase, es_kids_llm, confianza

if __name__ == '__main__':
    project = "LLM Kids"
    name = "V0"
    to_predict = "Made for kids"
    
    wandb.init(project=project, name=name, config={
        "model": "facebook/bart-large-mnli",
        "method": "Zero-Shot Classification",
        "max_chars": 800
    })

    print("Descargando dataframes desde MinIO...")
    _, _, df_test = download_model_dfs()
    
    #Toma todos los videos de df_test
    df_eval = df_test[df_test["Subtitulos"].notna()].sample(1000).copy()

    #Conteo total
    total_videos = len(df_eval)
    print(f"--- Iniciando evaluación de {total_videos} videos ---")

    wandb.config.update({"total_videos": total_videos})

    #Carga del Modelo
    device = 0 if torch.cuda.is_available() else -1
    print(f"Cargando BART en {'GPU' if device == 0 else 'CPU'}...")
    
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    
    categorias = ["Made for Kids", "Entertainment", "Education", "News", "People & Blogs"]

    print(f"Analizando {total_videos} videos...")
    resultados = df_eval['Subtitulos'].apply(lambda x: pd.Series(clasificar_subtitulo(x, classifier, categorias)))
    df_eval[['LLM_Categoria', 'LLM_Is_Kids', 'Confianza']] = resultados 
    #El porcentaje de confianza aunque sea bajo, muestra una probabilidad total de 1.0 entre las 5 categorías asignadas.

    y_true = df_eval[to_predict].astype(bool)
    y_pred = df_eval['LLM_Is_Kids'].astype(bool)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted')
    }

    wandb.log(metrics)
    wandb.log({"resultados_tabla": wandb.Table(dataframe=df_eval[['Titulo', to_predict, 'LLM_Is_Kids', 'Confianza']])})
    print("Scoring completado y subido a W&B.")
    wandb.finish()