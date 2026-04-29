import pandas as pd
import torch
from transformers import pipeline
from comun.preprocess_utils import download_model_dfs

def analizar_kids_llm_rubrica(n_muestras=15):
    print("Descargando dataframes desde MinIO...")
    df_test = download_model_dfs()
    
    # 2. DEFINICIÓN DE CATEGORÍAS Y TARGET
    # Usamos las categorías exactas solicitadas
    categorias = ["News & Politics", "People & Blogs", "Entertainment", "Education", "Made for Kids"]
    
    # Seleccionamos la muestra únicamente del set de TEST
    # Filtramos para asegurar que tengan subtítulos
    df_eval = df_test[df_test["Subtitulos"].notna()].sample(n_muestras).copy()

    # 3. CARGA DEL MODELO (Zero-Shot)
    print(f"Cargando LLM en {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

    def clasificar_subtitulo(texto):
        # El LLM analiza el contexto semántico del inicio del subtítulo
        # (Límite de tokens para eficiencia)
        res = classifier(str(texto)[:600], candidate_labels=categorias)
        
        # Obtenemos la etiqueta ganadora y su confianza
        mejor_clase = res['labels'][0]
        confianza = res['scores'][0]
        
        # Mapeo lógico: Si la clase ganadora es "Made for Kids", el predictor es True
        es_kids_llm = (mejor_clase == "Made for Kids")
        
        return mejor_clase, es_kids_llm, confianza

    # 4. EJECUCIÓN DEL ANÁLISIS SEMÁNTICO
    print(f"Evaluando {n_muestras} videos del set de TEST...")
    
    resultados = df_eval['Subtitulos'].apply(lambda x: pd.Series(clasificar_subtitulo(x)))
    df_eval[['LLM_Categoria', 'LLM_Is_Kids', 'Confianza']] = resultados

    return df_eval

if __name__ == "__main__":
    # Ejecución del reporte
    reporte_test = analizar_kids_llm_rubrica(n_muestras=10)
    
    print("\n" + "="*80)
    print("COMPARATIVA LLM - EVALUACIÓN SOBRE SET DE TEST (KIDS)")
    print("="*80)
    
    # Comparamos la predicción del LLM contra el valor real del set de Test
    columnas_comparar = ['Titulo', 'Made for kids', 'LLM_Is_Kids', 'LLM_Categoria', 'Confianza']
    print(reporte_test[columnas_comparar])
    
    # Cálculo de acierto simple en la muestra
    aciertos = (reporte_test['Made for kids'] == reporte_test['LLM_Is_Kids']).sum()
    print("-" * 80)
    print(f"Coincidencia LLM vs Dataset en Test: {aciertos}/{len(reporte_test)}")
    print("="*80)