import pandas as pd
import torch
import numpy as np
from transformers import pipeline
# Importamos las herramientas de tu proyecto
from comun.preprocess_utils import download_model_dfs 

def analizar_kids_llm_rubrica(n_muestras=15):
    print("Descargando dataframes desde MinIO...")
    _, _, df_test = download_model_dfs() #Ignora los dfs de train y validation
    
    # Filtra los videos que tengan subtitulos y no sean nulos
    df_eval = df_test[
        df_test["Subtitulos"].notna() & 
        (df_test["Subtitulos"].str.strip() != "")
    ].copy()

    if len(df_eval) == 0:
        print("No se encontraron registros con subtítulos válidos.")
        return None

    #Usa unos pocos para probar y crea una copia por si hay errores
    df_eval = df_eval.sample(min(n_muestras, len(df_eval))).copy()
    

    categorias = ["News & Politics", "People & Blogs", "Entertainment", "Education", "Made for Kids"]
    
    #Verifica si hay GPU disponible para acelerar el proceso
    device = 0 if torch.cuda.is_available() else -1
    print(f"Cargando BART en {'GPU' if device == 0 else 'CPU'}...")
    
    #Convierte el texto a números, pasa por las capas de BART, convierte las probabilidades finales en categorías asociadas a las etiquetas.
    classifier = pipeline( 
        model="facebook/bart-large-mnli", #Natural Language Inference: Determinar la relación entre dos fragmentos de texto: una premisa y una hipótesis.
        device=device
    )

    def clasificar_subtitulo(texto):
        #Asigna valor de caracteres que bart lea para no saturar el modelo
        res = classifier(str(texto)[:800], candidate_labels=categorias) #res es un diccionario con las etiquetas y sus respectivas puntuaciones
        mejor_clase = res['labels'][0] #toma la etiqueta con mayor puntuación
        confianza = res['scores'][0] #toma el puntake de la etiqueta
        #Si es muy bajo, está "adivinando"
        es_kids_llm = (mejor_clase == "Made for Kids") #Si la mejor clase es "Made for Kids", entonces es un video para niños
        
        return mejor_clase, es_kids_llm, confianza

    print(f"Analizando {len(df_eval)} videos extraídos...")
    
    # Aplica lógica y expande el resultado en tres columnas de la tupla devuelta
    resultados = df_eval['Subtitulos'].apply(lambda x: pd.Series(clasificar_subtitulo(x)))
    df_eval[['LLM_Categoria', 'LLM_Is_Kids', 'Confianza']] = resultados

    return df_eval

if __name__ == "__main__":
    reporte = analizar_kids_llm_rubrica(n_muestras=10)
    
    if reporte is not None:
        print("\n" + "="*80)
        print("REPORTE SEMÁNTICO (BART) - DATOS DE MINIO")
        print("="*80)
        
        # Columnas clave para verificar la extracción
        cols = ['Titulo', 'Made for kids', 'LLM_Is_Kids', 'LLM_Categoria', 'Confianza']
        print(reporte[cols])
        
        # Cálculo de coincidencia con la etiqueta real
        coincidencias = (reporte['Made for kids'] == reporte['LLM_Is_Kids']).sum()
        print("-" * 80)
        print(f"Coincidencia con etiqueta del Dataset: {coincidencias}/{len(reporte)}")